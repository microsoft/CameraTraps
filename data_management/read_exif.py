#
# read_exif.py
#
# Given a folder of images, read relevant EXIF fields from all images, and write them to 
# a .json file.  Depends on having exiftool available, since every pure-Python approach
# we've tried fails on at least some fields.
#
# Does not currently support command-line operation.
#

#%% Imports and constants

import os
import subprocess
import multiprocessing
import time
import json

from multiprocessing.pool import ThreadPool as ThreadPool
from multiprocessing.pool import Pool as Pool
# from functools import partial

from tqdm import tqdm

# From ai4eutils
from path_utils import find_images

verbose = False
n_print = 500
debug_max_images = None
use_joblib = False
use_threads = False
n_threads = 50

tag_types_to_ignore = set(['File','ExifTool'])
# exiftool_command_name = r'c:\exiftool-12.13\exiftool(-k).exe'
exiftool_command_name = r'c:\exiftool\exiftool.exe'
assert os.path.isfile(exiftool_command_name)

input_base = 'j:\\'
output_base = 'j:\\exif_results'
os.makedirs(output_base,exist_ok=True)

file_list_cache_file = os.path.join(output_base,'file_list.txt')
exif_cache_file = os.path.join(output_base,'exif_results.json')


#%% Multiprocessing init

def pinit(c):
    
    global cnt
    cnt = c
    
class Counter(object):
    
    def __init__(self, total):
        # 'i' means integer
        self.val = multiprocessing.Value('i', 0)
        self.total = multiprocessing.Value('i', total)
        self.last_print = multiprocessing.Value('i', 0)

    def increment(self, n=1):
        b_print = False
        with self.val.get_lock():
            self.val.value += n
            if ((self.val.value - self.last_print.value) >= n_print):
                self.last_print.value = self.val.value
                b_print = True           
        if b_print:
            total_string = ''
            if self.total.value > 0:
                 total_string = ' of {}'.format(self.total.value)
            print('{}: iteration {}{}'.format(time.strftime("%Y-%m-%d %H:%M:%S"), 
                                                                 self.val.value,total_string),flush=True)
    @property
    def value(self):
        return self.val.value
    def last_print_value(self):
        return self.last_print.value
    
pinit(Counter(-1))


#%% Functions

def enumerate_files():
    
    # image_files will contain the *relative* paths to all image files in the input folder
    
    if (file_list_cache_file is None) or (not os.path.isfile(file_list_cache_file)):
        
        image_files = find_images(input_base,recursive=True)
        image_files = [os.path.relpath(s,input_base) for s in image_files]
        image_files = [s.replace('\\','/') for s in image_files]
        if file_list_cache_file is not None:
            with open(file_list_cache_file,'w') as f:
                for fn in image_files:
                    f.write(fn + '\n')
                    
        print('Enumerated {} files'.format(len(image_files)))
        
    else:               
        
        with open(file_list_cache_file,'r') as f:
            image_files = f.readlines()
            image_files = [s.strip() for s in image_files]
            
        print('Read a list of {} files'.format(len(image_files)))

    return image_files

# file_path = os.path.join(input_base,image_files[0])
def read_exif_tags(file_path):
    """
    Get relevant fields from EXIF data for an image
    """
    
    result = {'status':'unknown','tags':{}}
    
    # -G means "Print group name for each tag", e.g. print:
    #
    # [File]          Bits Per Sample                 : 8
    #
    # ...instead of:
    #
    # Bits Per Sample                 : 8
    proc = subprocess.Popen([exiftool_command_name, '-G', file_path], stdout=subprocess.PIPE, encoding='utf8')
    
    exif_lines = proc.stdout.readlines()    
    exif_lines = [s.strip() for s in exif_lines]
    if ( (exif_lines is None) or (len(exif_lines) == 0) or not any([s.lower().startswith('[exif]') for s in exif_lines])):
        result['status'] = 'failure'
        return results
    
    # A list of three-element lists (type/tag/value)
    exif_tags = []
    
    # line_raw = exif_lines[0]
    for line_raw in exif_lines:
        
        # A typical line:
        #
        # [ExifTool]      ExifTool Version Number         : 12.13
        
        line = line_raw.strip()
        
        # Split on the first occurrence of ":"
        tokens = line.split(':',1)
        assert(len(tokens) == 2)
        
        field_value = tokens[1].strip()        
        
        field_name_type = tokens[0].strip()        
        field_name_type_tokens = field_name_type.split(None,1)
        assert len(field_name_type_tokens) == 2
        
        field_type = field_name_type_tokens[0].strip()
        assert field_type.startswith('[') and field_type.endswith(']')
        field_type = field_type[1:-1]
        
        if field_type in tag_types_to_ignore:
            if verbose:
                print('Ignoring tag with type {}'.format(field_type))
            continue        
        
        field_tag = field_name_type_tokens[1].strip()
        
        tag = [field_type,field_tag,field_value]
        
        exif_tags.append(tag)
        
    # ...for each output line
        
    return exif_tags

# ...process_exif()
    
def add_exif_data(im, overwrite=True):
    
    if cnt is not None:
        cnt.increment(n=1)
        
    fn = im['file_name']
    if verbose:
        print('Processing {}'.format(fn))
    
    if ('exif_tags' in im) and (overwrite==False):
        return None
    
    try:
        file_path = os.path.join(input_base,fn)
        assert os.path.isfile(file_path)
        exif_tags = read_exif_tags(file_path)
        im['exif_tags'] = exif_tags
    except Exception as e:
        s = 'Error on {}: {}'.format(fn,str(e))
        print(s)
        return s    
    return im


def create_image_objects():
    
    image_files = enumerate_files()

    images = []
    for fn in tqdm(image_files):
        im = {}
        im['file_name'] = fn
        images.append(im)
    
    if debug_max_images is not None:
        print('Trimming input list to {} images'.format(debug_max_images))
        images = images[0:debug_max_images]
    
    return images


def read_exif_data(images):
    
    if use_joblib:
        
        from joblib import Parallel, delayed
        # results = Parallel(n_jobs=n_threads)(delayed(add_exif_data)(im) for im in images[0:10])
        results = Parallel(n_jobs=n_threads)(delayed(add_exif_data)(im) for im in images)
        
    if n_threads == 1:
      
        results = []
        for im in images:        
            results.append(add_exif_data[im])    
        
    else:
        
        if use_threads:
            pool = ThreadPool(n_threads)
        else:
            pool = Pool(n_threads)
    
        results = list(pool.map(add_exif_data,images))

    return results


def write_exif_results(results):
    
    with open(exif_cache_file,'w') as f:
        json.dump(results,f,indent=1)


#%% Interactive driver

if False:
    
    #%%
    
    images = create_image_objects()
    results = read_exif_data(images)
    write_exif_results(results)
    
#%% Command-line driver

if __name__ == '__main__':
    
    images = create_image_objects()
    results = read_exif_data(images)
    write_exif_results(results)


#%% Create image objects
    


#%% Write results



#%% Scrap

if False:
    
    pass

    #%%
    
    im = images[0]
    file_path = os.path.join(input_base,im['file_name'])
    exif_tags = read_exif_tags(file_path)
    
    #%%
    
    fn = r"J:\exif_results\exif_results.json"
    with open(fn,'r') as f:
        exif_results = json.load(f)
    

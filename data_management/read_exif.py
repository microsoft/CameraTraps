#
# read_exif.py
#
# Given a folder of images, read relevant EXIF fields from all images, and write them to 
# a .json or .csv file.  Depends on having exiftool available, since every pure-Python
# approach we've tried fails on at least some fields.
#

#%% Imports and constants

import os
import subprocess
import multiprocessing
import time
import json

from multiprocessing.pool import ThreadPool as ThreadPool
from multiprocessing.pool import Pool as Pool

from tqdm import tqdm

# From ai4eutils
from path_utils import find_images

verbose = False
n_print = 500
debug_max_images = None

# Number of concurrent workers
n_workers = 1

# Should we use threads (vs. processes) for parallelization?
#
# Not relevant if n_workers is 1.
use_threads = True

tag_types_to_ignore = set(['File','ExifTool'])

exiftool_command_name = 'exiftool'


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

def enumerate_files(input_folder):
    
    # image_files will contain the *relative* paths to all image files in the input folder
    
    image_files = find_images(input_folder,recursive=True)
    image_files = [os.path.relpath(s,input_folder) for s in image_files]
    image_files = [s.replace('\\','/') for s in image_files]
    print('Enumerated {} files'.format(len(image_files)))
    return image_files


def read_exif_tags_for_image(file_path):
    """
    Get relevant fields from EXIF data for an image
    
    Returns a list of lists, where each element is (type/tag/value)    
    """
    
    result = {'status':'unknown','tags':[]}
    
    # -G means "Print group name for each tag", e.g. print:
    #
    # [File]          Bits Per Sample                 : 8
    #
    # ...instead of:
    #
    # Bits Per Sample                 : 8
    proc = subprocess.Popen([exiftool_command_name, '-G', file_path],
                            stdout=subprocess.PIPE, encoding='utf8')
    
    exif_lines = proc.stdout.readlines()    
    exif_lines = [s.strip() for s in exif_lines]
    if ( (exif_lines is None) or (len(exif_lines) == 0) or not \
        any([s.lower().startswith('[exif]') for s in exif_lines])):
        result['status'] = 'failure'
        return result
    
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
        
    result['status'] = 'success'
    result['tags'] = exif_tags
    return result

# ...process_exif()


def populate_exif_data(im, input_folder, overwrite=True):
    """
    Populate EXIF data into the image object [im].
    
    Returns a modified version of [im].
    """
    
    if cnt is not None:
        cnt.increment(n=1)
        
    fn = im['file_name']
    if verbose:
        print('Processing {}'.format(fn))
    
    if ('exif_tags' in im) and (overwrite==False):
        return None
    
    try:
        file_path = os.path.join(input_folder,fn)
        assert os.path.isfile(file_path)
        result = read_exif_tags_for_image(file_path)
        if result['status'] == 'success':
            exif_tags = result['tags']            
            im['exif_tags'] = exif_tags
        else:
            print('Error reading EXIF data for {}'.format(file_path))
    except Exception as e:
        s = 'Error on {}: {}'.format(fn,str(e))
        print(s)
        return s    
    return im


def create_image_objects(input_folder):
    """
    Create empty image objects for every image in [input_folder].
    """
    
    image_files = enumerate_files(input_folder)

    images = []
    for fn in image_files:
        im = {}
        im['file_name'] = fn
        images.append(im)
    
    if debug_max_images is not None:
        print('Trimming input list to {} images'.format(debug_max_images))
        images = images[0:debug_max_images]
    
    return images


def populate_exif_for_images(input_folder,images):
    """
    Main worker loop: read EXIF data for each image object in [images] and 
    populate the image objects.
    """
    
    if n_workers == 1:
      
        results = []
        for im in tqdm(images):
            results.append(populate_exif_data(im,input_folder))
        
    else:
        
        from functools import partial
        if use_threads:
            print('Starting parallel thread pool with {} workers'.format(n_workers))
            pool = ThreadPool(n_workers)
        else:
            print('Starting parallel process pool with {} workers'.format(n_workers))
            pool = Pool(n_workers)
    
        results = list(pool.map(partial(populate_exif_data,input_folder=input_folder),images))

    return results


def write_exif_results(results,output_file):
    """
    Write EXIF information to [output_file].
    """
    if output_file.endswith('.json'):
        with open(output_file,'w') as f:
            json.dump(results,f,indent=1)
    else:
        
        assert output_file.endswith('.csv')
        
        # Find all EXIF tags that exist in any image
        all_keys = set()
        for im in results:
            
            keys_this_image = set()
            exif_tags = im['exif_tags']
            file_name = im['file_name']
            for tag in exif_tags:
                tag_name = tag[1]
                assert tag_name not in keys_this_image, \
                    'Error: tag {} appears twice in image {}'.format(
                        tag_name,file_name)
                all_keys.add(tag_name)
                
            # ...for each tag in this image
            
        # ...for each image
        
        all_keys = sorted(list(all_keys))
        
        header = ['File Name']
        header.extend(all_keys)
        
        import csv
        with open(output_file,'w') as csvfile:
            
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(header)
            
            for im in results:
                
                row = [im['file_name']]
                kvp_this_image = {tag[1]:tag[2] for tag in im['exif_tags']}
                
                for i_key,key in enumerate(all_keys):
                    value = ''
                    if key in kvp_this_image:
                        value = kvp_this_image[key]
                    row.append(value)                                        
                # ...for each key that *might* be present in this image
                
                assert len(row) == len(header)
                
                writer.writerow(row)
                
            # ...for each image
            
        # ...with open()
    
    print('Wrote results to {}'.format(output_file))

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    
    # https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script

    from shutil import which
    return which(name) is not None


def read_exif_from_folder(input_folder,output_file):
    
    assert os.path.isdir(input_folder)
    
    assert output_file.lower().endswith('.json') or output_file.lower().endswith('.csv'), \
        'I only know how to write results to .json or .csv'
        
    try:
        with open(output_file, 'a') as f:
            if not f.writable():
                raise IOError('File not writable')
    except Exception:
        print('Could not write to file {}'.format(output_file))
        raise
    
    assert is_tool(exiftool_command_name), 'exiftool not available'

    images = create_image_objects(input_folder)
    results = populate_exif_for_images(input_folder,images)
    write_exif_results(results,output_file)
    
    
#%% Interactive driver

if False:
    
    #%%
    
    input_folder = os.path.expanduser('~/data/KRU-test')
    # input_folder = os.path.expanduser('~/data/KRU')
    # output_file = os.path.expanduser('~/data/test-exif.json')
    output_file = os.path.expanduser('~/data/test-exif.csv')
    
    read_exif_from_folder(input_folder,output_file)

    #%%
    
    with open(output_file,'r') as f:
        d = json.load(f)
        

#%% Command-line driver

import argparse

def main():

    global use_threads,n_workers
    
    parser = argparse.ArgumentParser(description=('Read EXIF information from all images in' + \
                                                  ' a folder, and write the results to .csv or .json'))

    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of concurrent workers to use (defaults to 1)')
    parser.add_argument('--use_threads', action='store_true',
                        help='Use threads (instead of processes) for multitasking')
    
    args = parser.parse_args()
    use_threads = args.use_threads
    n_workers = args.n_workers
    
    read_exif_from_folder(args.input_folder,args.output_file)
    
if __name__ == '__main__':
    main()

#
# read_exif.py
#
# Given a folder of images, read relevant metadata (EXIF/IPTC/XMP) fields from all images, 
# and write them to  a .json or .csv file.  
#
# This module can use either PIL (which can only reliably read EXIF data) or exiftool (which
# can read everything).  The latter approach expects that exiftool is available on the system
# path.  No attempt is made to be consistent in format across the two approaches.
#

#%% Imports and constants

import os
import subprocess
import json

from multiprocessing.pool import ThreadPool as ThreadPool
from multiprocessing.pool import Pool as Pool

from tqdm import tqdm
from PIL import Image, ExifTags

# From ai4eutils
from path_utils import find_images

from ct_utils import args_to_object

debug_max_images = None


#%% Options

class ReadExifOptions:
    
    verbose = False
    
    # Number of concurrent workers
    n_workers = 1
    
    # Should we use threads (vs. processes) for parallelization?
    #
    # Not relevant if n_workers is 1.
    use_threads = True
    
    tag_types_to_ignore = set(['File','ExifTool'])
    
    exiftool_command_name = 'exiftool'
    
    # Should we use exiftool or pil?
    processing_library = 'exiftool' # 'exiftool','pil'


#%% Functions

def enumerate_files(input_folder):
    """
    Enumerates all image files in input_folder, returning relative paths
    """
    
    image_files = find_images(input_folder,recursive=True)
    image_files = [os.path.relpath(s,input_folder) for s in image_files]
    image_files = [s.replace('\\','/') for s in image_files]
    print('Enumerated {} files'.format(len(image_files)))
    return image_files


def read_exif_tags_for_image(file_path,options=None):
    """
    Get relevant fields from EXIF data for an image
    
    Returns a dict with fields 'status' (str) and 'tags'
    
    The exact format of 'tags' depends on options.processing_library
    
    For exiftool, 'tags' is a list of lists, where each element is (type/tag/value)
    
    For pil, 'tags' is a dict (str:str)
    """
    
    if options is None:
        options = ReadExifOptions()
    
    result = {'status':'unknown','tags':[]}
    
    if options.processing_library == 'pil':
        
        try:
            img = Image.open(file_path)
            # exif_tags = img.info['exif'] if ('exif' in img.info) else None
            exif_info = img.getexif()
            if exif_info is None:
                exif_tags = None
            else:
                exif_tags = {}
                for k, v in exif_info.items():
                    assert isinstance(k,str) or isinstance(k,int), \
                        'Invalid EXIF key {}'.format(str(k))
                    if k in ExifTags.TAGS:
                        exif_tags[ExifTags.TAGS[k]] = str(v)
                    else:
                        # print('Warning: unrecognized EXIF tag: {}'.format(k))
                        exif_tags[k] = str(v)

        except Exception as e:
            print('Read failure for image {}: {}'.format(
                file_path,str(e)))
            result['status'] = 'read_failure'
            result['error'] = str(e)
        
        if result['status'] == 'unknown':
            if exif_tags is None:            
                result['status'] = 'empty_read'
            else:
                result['status'] = 'success'
                result['tags'] = exif_tags
                
        return result
        
    elif options.processing_library == 'exiftool':
        
        # -G means "Print group name for each tag", e.g. print:
        #
        # [File]          Bits Per Sample                 : 8
        #
        # ...instead of:
        #
        # Bits Per Sample                 : 8
        proc = subprocess.Popen([options.exiftool_command_name, '-G', file_path],
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
            assert(len(tokens) == 2), 'EXIF tokenization failure ({} tokens, expected 2)'.format(
                len(tokens))
            
            field_value = tokens[1].strip()        
            
            field_name_type = tokens[0].strip()        
            field_name_type_tokens = field_name_type.split(None,1)
            assert len(field_name_type_tokens) == 2, 'EXIF tokenization failure'
            
            field_type = field_name_type_tokens[0].strip()
            assert field_type.startswith('[') and field_type.endswith(']'), \
                'Invalid EXIF field {}'.format(field_type)
            field_type = field_type[1:-1]
            
            if field_type in options.tag_types_to_ignore:
                if options.verbose:
                    print('Ignoring tag with type {}'.format(field_type))
                continue        
            
            field_tag = field_name_type_tokens[1].strip()
            
            tag = [field_type,field_tag,field_value]
            
            exif_tags.append(tag)
            
        # ...for each output line
            
        result['status'] = 'success'
        result['tags'] = exif_tags
        return result
    
    else:
        
        raise ValueError('Unknown processing library {}'.format(
            options.processing_library))

    # ...which processing library are we using?
    
# ...read_exif_tags_for_image()


def populate_exif_data(im, image_base, options=None):
    """
    Populate EXIF data into the image object [im].
    
    im['file_name'] is relative to image_base.
    
    Returns a modified version of [im].
    """
    
    if options is None:
        options = ReadExifOptions()

    fn = im['file_name']
    if options.verbose:
        print('Processing {}'.format(fn))
    
    try:
        file_path = os.path.join(image_base,fn)
        assert os.path.isfile(file_path), 'Could not find file {}'.format(file_path)
        result = read_exif_tags_for_image(file_path,options)
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

# ...populate_exif_data()


def create_image_objects(image_files):
    """
    Create empty image objects for every image in [image_files], which can be a 
    list of relative paths (which will get stored without processing, so the base 
    path doesn't matter here), or a folder name.
    
    Returns a list of dicts with field 'file_name' (a relative path).
    """
    
    # Enumerate *relative* paths
    if isinstance(image_files,str):
        assert os.path.isdir(image_files), 'Invalid image folder {}'.format(image_files)
        image_files = enumerate_files(image_files)
        
    images = []
    for fn in image_files:
        im = {}
        im['file_name'] = fn
        images.append(im)
    
    if debug_max_images is not None:
        print('Trimming input list to {} images'.format(debug_max_images))
        images = images[0:debug_max_images]
    
    return images


def populate_exif_for_images(image_base,images,options=None):
    """
    Main worker loop: read EXIF data for each image object in [images] and 
    populate the image objects.
    
    'images' should be a list of dicts with the field 'file_name' containing
    a relative path (relative to 'image_base').    
    """
    
    if options is None:
        options = ReadExifOptions()

    if options.n_workers == 1:
      
        results = []
        for im in tqdm(images):
            results.append(populate_exif_data(im,image_base,options))
        
    else:
        
        from functools import partial
        if options.use_threads:
            print('Starting parallel thread pool with {} workers'.format(options.n_workers))
            pool = ThreadPool(options.n_workers)
        else:
            print('Starting parallel process pool with {} workers'.format(options.n_workers))
            pool = Pool(options.n_workers)
    
        results = list(tqdm(pool.imap(partial(populate_exif_data,image_base=image_base,
                                        options=options),images),total=len(images)))

    return results


def write_exif_results(results,output_file):
    """
    Write EXIF information to [output_file].
    
    'results' is a list of dicts with fields 'exif_tags' and 'file_name'.

    Writes to .csv or .json depending on the extension of 'output_file'.         
    """
    
    if output_file.endswith('.json'):
        
        with open(output_file,'w') as f:
            json.dump(results,f,indent=1)
            
    elif output_file.endswith('.csv'):
        
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
    
    else:
        
        raise ValueError('Could not determine output type from file {}'.format(
            output_file))
        
    # ...if we're writing to .json/.csv
    
    print('Wrote results to {}'.format(output_file))


def is_executable(name):
    
    """Check whether `name` is on PATH and marked as executable."""
    
    # https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script

    from shutil import which
    return which(name) is not None


def read_exif_from_folder(input_folder,output_file=None,options=None,filenames=None):
    """
    Read EXIF for all images in input_folder.
    
    If filenames is not None, it should be a list of relative filenames; only those files will 
    be processed.
    
    input_folder can be None or '', in which case filenames should be a list of absolute paths.
    """
    
    if options is None:
        options = ReadExifOptions()
    
    if input_folder is None:
        input_folder = ''
    if len(input_folder) > 0:
        assert os.path.isdir(input_folder), \
            '{} is not a valid folder'.format(input_folder)

    assert (len(input_folder) > 0) or (filenames is not None), \
        'Must specify either a folder or a list of files'
        
    if output_file is not None:    
        
        assert output_file.lower().endswith('.json') or output_file.lower().endswith('.csv'), \
            'I only know how to write results to .json or .csv'
            
        try:
            with open(output_file, 'a') as f:
                if not f.writable():
                    raise IOError('File not writable')
        except Exception:
            print('Could not write to file {}'.format(output_file))
            raise
        
    if options.processing_library == 'exif':
        assert is_executable(options.exiftool_command_name), 'exiftool not available'

    if filenames is None:
        images = create_image_objects(input_folder)
    else:
        assert isinstance(filenames,list)
        images = create_image_objects(filenames)
        
    results = populate_exif_for_images(input_folder,images,options)
    
    if output_file is not None:
        write_exif_results(results,output_file)
        
    return results

    
#%% Interactive driver

if False:
    
    #%%
    
    input_folder = os.path.expanduser('~/data/KRU-test')
    output_file = os.path.expanduser('~/data/test-exif.json')
    # output_file = os.path.expanduser('~/data/test-exif.csv')
    options = ReadExifOptions()
    options.verbose = False
    options.n_workers = 10
    options.use_threads = False
    options.processing_library = 'exiftool'
    # options.processing_library = 'pil'

    # file_path = os.path.join(input_folder,'KRU_S1_11_R1_IMAG0148.JPG')
    
    output_file = None
    results = read_exif_from_folder(input_folder,output_file,options)

    #%%
    
    with open(output_file,'r') as f:
        d = json.load(f)
        

#%% Command-line driver

import argparse
import sys

def main():

    options = ReadExifOptions()
    
    parser = argparse.ArgumentParser(description=('Read EXIF information from all images in' + \
                                                  ' a folder, and write the results to .csv or .json'))

    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of concurrent workers to use (defaults to 1)')
    parser.add_argument('--use_threads', action='store_true',
                        help='Use threads (instead of processes) for multitasking')
    parser.add_argument('--processing_library', type=str, default=options.processing_library,
                        help='Processing library (exif or pil)')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()    
    args_to_object(args, options)
    options.processing_library = options.processing_library.lower()
    
    read_exif_from_folder(args.input_folder,args.output_file,options)
    
if __name__ == '__main__':
    main()

#
# subset_json_detector_output.py
#
# Creates one or more subsets of a detector API output file (.json), doing either
# or both of the following (if both are requested, they happen in this order):
#
# 1) Retrieve all elements where filenames contain a specified query string, 
#    optionally replacing that query with a replacement token. If the query is blank, 
#    can also be used to prepend content to all filenames.
#
#    Does not support regex's, but supports a special case of ^string to indicate "must start with
#    to match".
#
# 2) Create separate .jsons for each unique path, optionally making the filenames 
#    in those .json's relative paths.  In this case, you specify an output directory, 
#    rather than an output path.  All images in the folder blah\foo\bar will end up 
#    in a .json file called blah_foo_bar.json.
#
###
#
# Sample invocations (splitting into multiple json's):
#
# Read from "1800_idfg_statewide_wolf_detections_w_classifications.json", split up into 
# individual .jsons in 'd:\temp\idfg\output', making filenames relative to their individual
# folders:
#
# python subset_json_detector_output.py "d:\temp\idfg\1800_idfg_statewide_wolf_detections_w_classifications.json" "d:\temp\idfg\output" --split_folders --make_folder_relative
#
# Now do the same thing, but instead of writing .json's to d:\temp\idfg\output, write them to *subfolders*
# corresponding to the subfolders for each .json file.
#
# python subset_json_detector_output.py "d:\temp\idfg\1800_detections_S2.json" "d:\temp\idfg\output_to_folders" --split_folders --make_folder_relative --copy_jsons_to_folders
#
###
#
# Sample invocations (creating a single subset matching a query):
#
# Read from "1800_detections.json", write to "1800_detections_2017.json"
#
# Include only images matching "2017", and change "2017" to "blah"
#
# python subset_json_detector_output.py "d:\temp\1800_detections.json" "d:\temp\1800_detections_2017_blah.json" --query 2017 --replacement blah
#
# Include all images, prepend with "prefix/"
#
# python subset_json_detector_output.py "d:\temp\1800_detections.json" "d:\temp\1800_detections_prefix.json" --replacement "prefix/"
#
###
#
# To subset a COCO Camera Traps .json database, see subset_json_db.py
#

#%% Constants and imports

import argparse
import sys
import copy
import json
import os
import re

from tqdm import tqdm

from ct_utils import args_to_object


#%% Helper classes

class SubsetJsonDetectorOutputOptions:
    
    # Only process files containing the token 'query'
    query = None
    
    # Replace 'query' with 'replacement' if 'replacement' is not None.  If 'query' is None,
    # prepend 'replacement'
    replacement = None
    
    # Should we split output into individual .json files for each folder?
    split_folders = False
    
    # Folder level to use for splitting ['bottom','top','n_from_bottom','n_from_top','dict']
    #
    # 'dict' requires 'split_folder_param' to be a dictionary mapping each filename
    # to a token.
    split_folder_mode = 'bottom'  # 'top'
    
    # When using the 'n_from_bottom' parameter to define folder splitting, this
    # defines the number of directories from the bottom.  'n_from_bottom' with
    # a parameter of zero is the same as 'bottom'.
    #
    # Same story with 'n_from_top'.
    #
    # When 'split_folder_mode' is 'dict', this should be a dictionary mapping each filename
    # to a token.
    split_folder_param = 0
    
    # Only meaningful if split_folders is True: should we convert pathnames to be relative
    # the folder for each .json file?
    make_folder_relative = False
    
    # Only meaningful if split_folders and make_folder_relative are True: if not None, 
    # will copy .json files to their corresponding output directories, relative to 
    # output_filename
    copy_jsons_to_folders = False
    
    # Should we over-write .json files?
    overwrite_json_files = False
    
    # If copy_jsons_to_folders is true, do we require that directories already exist?
    copy_jsons_to_folders_directories_must_exist = True
    
    # Threshold on confidence
    confidence_threshold = None
    
    # Should we remove failed images?
    remove_failed_images = False
    
    debug_max_images = -1
    
    
#%% Main function

def write_detection_results(data, output_filename, options):
    """
    Write the detector-output-formatted dict *data* to *output_filename*.
    """
    
    if (not options.overwrite_json_files) and os.path.isfile(output_filename):
        raise ValueError('File {} exists'.format(output_filename))
    
    basedir = os.path.dirname(output_filename)
    
    if options.copy_jsons_to_folders and options.copy_jsons_to_folders_directories_must_exist:
        if not os.path.isdir(basedir):
            raise ValueError('Directory {} does not exist'.format(basedir))
    else:
        os.makedirs(basedir, exist_ok=True)
    
    print('Serializing to {}...'.format(output_filename), end='')
    s = json.dumps(data, indent=1)
    print(' ...done')
    print('Writing output file...', end='')
    with open(output_filename, "w") as f:
        f.write(s)
    print(' ...done')

# ...write_detection_results()


def subset_json_detector_output_by_confidence(data, options):
    """
    Remove all detections below options.confidence_threshold, update max confidences accordingly.
    """
    
    if not options.confidence_threshold:
        return data
    
    images_in = data['images']
    images_out = []    
    
    print('Subsetting by confidence >= {}'.format(options.confidence_threshold))
    
    n_max_changes = 0
    
    # iImage = 0; im = images_in[0]
    for iImage, im in tqdm(enumerate(images_in), total=len(images_in)):
        
        if ('detections' not in im) or (im['detections'] is None):
            continue
        
        p_orig = im['max_detection_conf']

        # Find all detections above threshold for this image
        detections = [d for d in im['detections'] if d['conf'] >= options.confidence_threshold]

        # If there are no detections above threshold, set the max probability
        # to -1, unless it already had a negative probability.
        if len(detections) == 0:
            if p_orig <= 0:                
                p = p_orig
            else:
                p = -1

        # Otherwise find the max confidence
        else:
            p = max(d['conf'] for d in detections)
        
        im['detections'] = detections

        # Did this thresholding result in a max-confidence change?
        if abs(p_orig - p) > 0.00001:

            # We should only be *lowering* max confidence values (i.e., making them negative)
            assert (p_orig <= 0) or (p < p_orig), 'Confidence changed from {} to {}'.format(p_orig, p)
            n_max_changes += 1
        im['max_detection_conf'] = p
        images_out.append(im)
        
    # ...for each image        
    
    data['images'] = images_out    
    print('done, found {} matches (of {}), {} max conf changes'.format(
            len(data['images']),len(images_in),n_max_changes))
    
    return data

# ...subset_json_detector_output_by_confidence()


def remove_failed_images(data,options):
    """
    Removed failed images from [data]
    """
    images_in = data['images']
    images_out = []    
    
    if not options.remove_failed_images:
        return data
        
    print('Removing failed images...', end='')
    
    # i_image = 0; im = images_in[0]
    for i_image, im in tqdm(enumerate(images_in), total=len(images_in)):
        
        if 'failure' in im and isinstance(im['failure'],str):
            continue
        else:
            images_out.append(im)
        
    # ...for each image        
    
    data['images'] = images_out    
    n_removed = len(images_in) - len(data['images'])
    print('Done, removed {} of {}'.format(n_removed, len(images_in)))
    
    return data

# ...remove_failed_images()


def subset_json_detector_output_by_query(data, options):
    """
    Subset to images whose filename matches options.query; replace all instances of 
    options.query with options.replacement.
    """
    
    images_in = data['images']
    images_out = []    
    
    print('Subsetting by query {}, replacement {}...'.format(options.query, options.replacement), end='')
    
    query_string = options.query
    query_starts_with = False
    
    # Support a special case regex-like notation for "starts with"
    if query_string is not None and query_string.startswith('^'):
        query_string = query_string[1:]
        query_starts_with = True
        
    # i_image = 0; im = images_in[0]
    for i_image, im in tqdm(enumerate(images_in), total=len(images_in)):
        
        fn = im['file']
        
        # Only take images that match the query
        if query_string is not None:
            if query_starts_with:
                if (not fn.startswith(query_string)):
                    continue
            else:
                if query_string not in fn:
                    continue
        
        if options.replacement is not None:
            if query_string is not None:
                fn = fn.replace(query_string, options.replacement)
            else:
                fn = options.replacement + fn
            
        im['file'] = fn
        
        images_out.append(im)
        
    # ...for each image        
    
    data['images'] = images_out    
    print('done, found {} matches (of {})'.format(len(data['images']), len(images_in)))
    
    return data

# ...subset_json_detector_output_by_query()


def split_path(path, maxdepth=100):
    """
    Splits [path] into all its constituent tokens, e.g.:
    
    c:\blah\boo\goo.txt
    
    ...becomes:
        
    ['c:\\', 'blah', 'boo', 'goo.txt']
    
    http://nicks-liquid-soapbox.blogspot.com/2011/03/splitting-path-to-list-in-python.html
    """
    (head, tail) = os.path.split(path)
    return split_path(head, maxdepth - 1) + [tail] \
        if maxdepth and head and head != path \
        else [head or tail]

# ...split_path()

    
def top_level_folder(p):
    """
    Gets the top-level folder from the path *p*; on Windows, will use the top-level folder
    that isn't the drive.  E.g., top_level_folder(r"c:\blah\foo") returns "c:\blah".  Does not
    include the leaf node, i.e. top_level_folder('/blah/foo') returns '/blah'.
    """
    if p == '':
        return ''
    
    # Path('/blah').parts is ('/','blah')
    parts = split_path(p)
    
    if len(parts) == 1:
        return parts[0]

    # Handle paths like:
    #
    # /, \, /stuff, c:, c:\stuff
    drive = os.path.splitdrive(p)[0]
    if parts[0] == drive or parts[0] == drive + '/' or parts[0] == drive + '\\' or parts[0] in ['\\', '/']:
        return os.path.join(parts[0], parts[1])
    else:
        return parts[0]

# ...top_level_folder()

    
if False:  
      
    p = 'blah/foo/bar'; s = top_level_folder(p); print(s); assert s == 'blah'
    p = '/blah/foo/bar'; s = top_level_folder(p); print(s); assert s == '/blah'
    p = 'bar'; s = top_level_folder(p); print(s); assert s == 'bar'
    p = ''; s = top_level_folder(p); print(s); assert s == ''
    p = 'c:\\'; s = top_level_folder(p); print(s); assert s == 'c:\\'
    p = r'c:\blah'; s = top_level_folder(p); print(s); assert s == 'c:\\blah'
    p = r'c:\foo'; s = top_level_folder(p); print(s); assert s == 'c:\\foo'
    p = r'c:/foo'; s = top_level_folder(p); print(s); assert s == 'c:/foo'
    p = r'c:\foo/bar'; s = top_level_folder(p); print(s); assert s == 'c:\\foo'
    
    
def subset_json_detector_output(input_filename, output_filename, options, data=None):
    """
    Main internal entry point
        
    Makes a copy of [data] before modifying if a data dictionary is supplied.
    """
    
    if options is None:    
        options = SubsetJsonDetectorOutputOptions()
            
    # Input validation        
    if options.copy_jsons_to_folders:
        assert options.split_folders and options.make_folder_relative, \
            'copy_jsons_to_folders set without make_folder_relative and split_folders'
                
    if options.split_folders:
        if os.path.isfile(output_filename):
            raise ValueError('When splitting by folders, output must be a valid directory name, you specified an existing file')
            
    if data is None:
        print('Reading json...', end='')
        with open(input_filename) as f:
            data = json.load(f)
        print(' ...done, read {} images'.format(len(data['images'])))
        if options.debug_max_images > 0:
            print('Trimming to {} images'.format(options.debug_max_images))
            data['images'] = data['images'][:options.debug_max_images]
    else:
        print('Copying data')
        data = copy.deepcopy(data)
        print('...done')
        
    if options.query is not None:
        
        data = subset_json_detector_output_by_query(data, options)
    
    if options.remove_failed_images:
        
        data = remove_failed_images(data, options)
        
    if options.confidence_threshold is not None:
        
        data = subset_json_detector_output_by_confidence(data, options)
        
    if not options.split_folders:
        
        write_detection_results(data, output_filename, options)
        return data
    
    else:
        
        # Map images to unique folders
        print('Finding unique folders')    
        
        folders_to_images = {}
        
        # im = data['images'][0]
        for im in tqdm(data['images']):
            
            fn = im['file']
            
            if options.split_folder_mode == 'bottom':
                                
                dirname = os.path.dirname(fn)
                
            elif options.split_folder_mode == 'n_from_bottom':
                
                dirname = os.path.dirname(fn)
                for n in range(0, options.split_folder_param):
                    dirname = os.path.dirname(dirname)
                    
            elif options.split_folder_mode == 'n_from_top':
                
                # Split string into folders, keeping delimiters
                
                # Don't use this, it removes delimiters
                # tokens = split_path(fn)
                tokens = re.split(r'([\\/])',fn)
                
                n_tokens_to_keep = ((options.split_folder_param + 1) * 2) - 1;
                
                if n_tokens_to_keep > len(tokens):
                    raise ValueError('Cannot walk {} folders from the top in path {}'.format(
                                options.split_folder_param, fn))
                dirname = ''.join(tokens[0:n_tokens_to_keep])
                
            elif options.split_folder_mode == 'top':
                
                dirname = top_level_folder(fn)                
                
            elif options.split_folder_mode == 'dict':
                
                assert isinstance(options.split_folder_param, dict)
                dirname = options.split_folder_param[fn]
                
            else:
                
                raise ValueError('Unrecognized folder split mode {}'.format(options.split_folder_mode))
                
            folders_to_images.setdefault(dirname, []).append(im)
        
        # ...for each image
                
        print('Found {} unique folders'.format(len(folders_to_images)))
        
        # Optionally make paths relative
        # dirname = list(folders_to_images.keys())[0]
        if options.make_folder_relative:
            
            print('Converting database-relative paths to individual-json-relative paths...')
        
            for dirname in tqdm(folders_to_images):
                # im = folders_to_images[dirname][0]
                for im in folders_to_images[dirname]:
                    fn = im['file']
                    relfn = os.path.relpath(fn, dirname).replace('\\', '/')
                    im['file'] = relfn
        
        # ...if we need to convert paths to be folder-relative
        
        print('Finished converting to json-relative paths, writing output')
                       
        os.makedirs(output_filename, exist_ok=True)
        all_images = data['images']
        
        # dirname = list(folders_to_images.keys())[0]
        for dirname in tqdm(folders_to_images):
                        
            json_fn = dirname.replace('/', '_').replace('\\', '_') + '.json'
            
            if options.copy_jsons_to_folders:
                json_fn = os.path.join(output_filename, dirname, json_fn)
            else:
                json_fn = os.path.join(output_filename, json_fn)
            
            # Recycle the 'data' struct, replacing 'images' every time... medium-hacky, but 
            # forward-compatible in that I don't take dependencies on the other fields
            dir_data = data
            dir_data['images'] = folders_to_images[dirname]
            write_detection_results(dir_data, json_fn, options)
            print('Wrote {} images to {}'.format(len(dir_data['images']), json_fn))
            
        # ...for each directory
        
        data['images'] = all_images
        
        return data
    
    # ...if we're splitting folders

# ...subset_json_detector_output()

    
#%% Interactive driver
                
if False:

    #%%
    
    #%% Subset a file without splitting
    
    input_filename = r"c:\temp\sample.json"
    output_filename = r"c:\temp\output.json"
     
    options = SubsetJsonDetectorOutputOptions()
    options.replacement = None
    options.query = 'S2'
        
    data = subset_json_detector_output(input_filename,output_filename,options,None)
    

    #%% Subset and split, but don't copy to individual folders

    input_filename = r"C:\temp\xxx-20201028_detections.filtered_rde_0.60_0.85_10_0.05_r2_export\xxx-20201028_detections.filtered_rde_0.60_0.85_10_0.05_r2_export.json"
    output_filename = r"c:\temp\out"
    
    options = SubsetJsonDetectorOutputOptions()
    options.split_folders = True    
    options.make_folder_relative = True
    options.split_folder_mode = 'n_from_top'
    options.split_folder_param = 1
    
    data = subset_json_detector_output(input_filename,output_filename,options,None)
    
    
    #%% Subset and split, copying to individual folders
    
    input_filename = r"c:\temp\sample.json"
    output_filename = r"c:\temp\out"
     
    options = SubsetJsonDetectorOutputOptions()
    options.split_folders = True    
    options.make_folder_relative = True
    options.copy_jsons_to_folders = True
    
    data = subset_json_detector_output(input_filename,output_filename,options,data)
    

#%% Command-line driver


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input .json filename')
    parser.add_argument('output_file', type=str, help='Output .json filename')
    parser.add_argument('--query', type=str, default=None, help='Query string to search for (omitting this matches all)')
    parser.add_argument('--replacement', type=str, default=None, help='Replace [query] with this')
    parser.add_argument('--confidence_threshold', type=float, default=None, help='Remove detections below this confidence level')
    parser.add_argument('--split_folders', action='store_true', help='Split .json files by leaf-node folder')
    parser.add_argument('--split_folder_param', type=int, help='Directory level count for n_from_bottom and n_from_top splitting')
    parser.add_argument('--split_folder_mode', type=str, help='Folder level to use for splitting ("top" or "bottom")')
    parser.add_argument('--make_folder_relative', action='store_true', help='Make image paths relative to their containing folder (only meaningful with split_folders)')
    parser.add_argument('--overwrite_json_files', action='store_true', help='Overwrite output files')
    parser.add_argument('--copy_jsons_to_folders', action='store_true', help='When using split_folders and make_folder_relative, copy jsons to their corresponding folders (relative to output_file)')
    parser.add_argument('--create_folders', action='store_true', help='When using copy_jsons_to_folders, create folders that don''t exist')    
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()    
    
    # Convert to an options object
    options = SubsetJsonDetectorOutputOptions()
    if args.create_folders:
        options.copy_jsons_to_folders_directories_must_exist = False
        
    args_to_object(args, options)
    
    subset_json_detector_output(args.input_file, args.output_file, options)
    

if __name__ == '__main__':
    
    main()

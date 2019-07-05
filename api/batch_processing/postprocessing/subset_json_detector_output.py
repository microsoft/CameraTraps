#
# subset_json_detector_output.py
#
# Creates one or more subsets of a detector API output file (.json).  Can operate in two
# modes:
#
# 1) Retrieve all elements where filenames contain a specified query string, optionally
#    replacing that prefix with a replacement token. If the query is blank, can also be 
#    used to prepend content to all filenames.
#
# 2) Create separate .jsons for each unique path, optionally making the filenames in those .json's relative
#    paths.  In this case, you specify an output directory, rather than an output path.  All images in the
#    folder blah\foo\bar will end up in a .json file called blah_foo_bar.json.
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

#%% Constants and imports

import json
from tqdm import tqdm
import os


#%% Helper classes

class SubsetJsonDetectorOutputOptions:
    
    replacement = None
    query = None
    
    split_folders = False
    make_folder_relative = False
    
    
#%% Main function

def write_images(data, output_filename):
    
    basedir = os.path.dirname(output_filename)
    os.makedirs(basedir,exist_ok=True)
    
    print('Serializing to {}...'.format(output_filename), end = '')    
    s = json.dumps(data, indent=1)
    print(' ...done')
    print('Writing output file...', end = '')    
    with open(output_filename, "w") as f:
        f.write(s)
    print(' ...done')


def subset_json_detector_output_by_query(data,output_filename,options):
    
    images_in = data['images']

    images_out = []    
    
    print('Searching json...',end='')
    
    # iImage = 0; im = images_in[0]
    for iImage,im in tqdm(enumerate(images_in),total=len(images_in)):
        
        fn = im['file']
        
        # Only take images that match the query
        if (options.query is not None) and (options.query not in fn):
            continue
        
        if options.replacement is not None:
            if options.query is not None:
                fn = fn.replace(options.query,options.replacement)
            else:
                fn = options.replacement + fn
            
        im['file'] = fn
        
        images_out.append(im)
        
    # ...for each image        
    
    print(' ...done')
    
    data['images'] = images_out
    
    write_images(data,output_filename)
    
    print('Done, found {} matches (of {})'.format(len(data['images']),len(images_in)))
    
    return data


def subset_json_detector_output(input_filename,output_filename,options):

    if options is None:    
        options = SubsetJsonDetectorOutputOptions()
        
    # Input validation        
    if (options.query is not None or options.replacement is not None) and options.split_folders:
        raise ValueError('Query/replacement strings and splitting by folders are mutually exclusive')
        
    if options.split_folders:
        if os.path.isfile(output_filename):
            raise ValueError('When splitting by folders, output must be a valid directory name, you specified an existing file')
            
    print('Reading json...',end='')
    with open(input_filename) as f:
        data = json.load(f)
    print(' ...done')
    
    if options.split_folders:
        
        # Map images to unique folders
        print('Finding unique folders')    
        
        folders_to_images = {}
        
        # im = data['images'][0]
        for im in tqdm(data['images']):
            fn = im['file']
            dirname = os.path.dirname(fn)            
            folders_to_images.setdefault(dirname,[]).append(im)
        
        print('Found {} unique folders'.format(len(folders_to_images)))
        
        # Optionally make paths relative
        # dirname = list(folders_to_images.keys())[0]
        if options.make_folder_relative:
            
            print('Converting absolute paths to relative paths...')
        
            for dirname in folders_to_images:
                # im = folders_to_images[dirname][0]
                for im in folders_to_images[dirname]:
                    fn = im['file']
                    relfn = os.path.relpath(fn,dirname)
                    im['file'] = relfn
                    
        print('Finished converting to relative paths, writing output')
                       
        os.makedirs(output_filename,exist_ok=True)
        all_images = data['images']
        # dirname = list(folders_to_images.keys())[0]
        for dirname in folders_to_images:
            json_fn = os.path.join(output_filename,dirname.replace('/','_').replace('\\','_') + '.json')
            
            # Recycle the 'data' struct, replacing 'images' every time... medium-hacky, but 
            # forward-compatible in that I don't take dependencies on the other fields
            dir_data = data
            dir_data['images'] = folders_to_images[dirname]
            write_images(dir_data, json_fn)
            print('Wrote {} images to {}'.format(len(dir_data['images']),json_fn))
        # ...for each directory
        
        data['images'] = all_images
        return data
    
    else:
        return subset_json_detector_output_by_query(data,output_filename,options)
    
    
#%% Interactive driver
                
if False:

    #%%   
    
    input_filename = r"D:\temp\idfg\1800_idfg_statewide_wolf_detections_w_classifications.json"
    output_filename = r"D:\temp\idfg\output\1800_detections_2017.json"
     
    options = SubsetJsonDetectorOutputOptions()
    options.replacement = 'blah'
    options.query = '2017'
        
    data = subset_json_detector_output(input_filename,output_filename,options)
    print('Done, found {} matches'.format(len(data['images'])))

    #%%
    
    input_filename = r"D:\temp\idfg\1800_idfg_statewide_wolf_detections_w_classifications.json"
    output_filename = r"D:\temp\idfg\output"
     
    options = SubsetJsonDetectorOutputOptions()
    options.split_folders = True
    options.make_folder_relative = True
    
    data = subset_json_detector_output(input_filename,output_filename,options)
    print('Done')


#%% Command-line driver

import argparse
import inspect
import sys

# Copy all fields from a Namespace (i.e., the output from parse_args) to an object.  
#
# Skips fields starting with _.  Does not check field existence in the target object.
def argsToObject(args, obj):
    
    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v);

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input .json filename')
    parser.add_argument('output_file', type=str, help='Output .json filename')
    parser.add_argument('--query', type=str, default=None, help='Query string to search for (omitting this matches all)')
    parser.add_argument('--replacement', type=str, default=None, help='Replace [query] with this')
    parser.add_argument('--split_folders', action='store_true', help='Split .json files by leaf-node folder')
    parser.add_argument('--make_folder_relative', action='store_true', help='Make image paths relative to their containing folder (only meaningful with split_folders)')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()    
    
    # Convert to an options object
    options = SubsetJsonDetectorOutputOptions()
    argsToObject(args,options)
    
    subset_json_detector_output(args.input_file,args.output_file,options)
    
if __name__ == '__main__':
    
    main()

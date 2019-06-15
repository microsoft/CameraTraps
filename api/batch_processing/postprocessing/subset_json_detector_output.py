#
# subset_json_detector_output.py
#
# Pulls a subset of a detector API output file (.json) where filenames match 
# a specified query (prefix), optionally replacing that prefix with a replacement token.  
# If the query is blank, can also be used to prepend content to all filenames.
#
# Sample invocation:
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


#%% Helper classes

class SubsetJsonDetectorOutputOptions:
    
    replacement = None
    query = ''
    
    
#%% Main function
                
def subset_json_detector_output(input_filename,output_filename,options):

    if options is None:    
        options = SubsetJsonDetectorOutputOptions()
            
    print('Reading json...',end='')
    with open(input_filename) as f:
        data = json.load(f)
    print(' ...done')
    
    images_in = data['images']
    
    images_out = []
    
    print('Searching json...',end='')
    
    # iImage = 0; im = images_in[0]
    for iImage,im in tqdm(enumerate(images_in),total=len(images_in)):
        
        fn = im['file']
        
        # Only take images that match the query
        if not ((len(options.query) == 0) or (fn.startswith(options.query))):
            continue
        
        if options.replacement is not None:
            if len(options.query) > 0:
                fn = fn.replace(options.query,options.replacement)
            else:
                fn = options.replacement + fn
            
        im['file'] = fn
        
        images_out.append(im)
        
    # ...for each image        
    
    print(' ...done')
    
    data['images'] = images_out
    
    print('Serializing back to .json...', end = '')    
    s = json.dumps(data, indent=1)
    print(' ...done')
    print('Writing output file...', end = '')    
    with open(output_filename, "w") as f:
        f.write(s)
    print(' ...done')

    return data

    print('Done, found {} matches (of {})'.format(len(data['images']),len(images_in)))
    
    
#%% Interactive driver
                
if False:

    #%%   
    
    input_filename = r"D:\temp\1800_detections.json"
    output_filename = r"D:\temp\1800_detections_2017.json"
     
    options = SubsetJsonDetectorOutputOptions()
    options.replacement = 'blah'
    options.query = '2017'
        
    data = subset_json_detector_output(input_filename,output_filename,options)
    print('Done, found {} matches'.format(len(data['images'])))

    
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
    parser.add_argument('--query', type=str, default='', help='Prefix to search for (omitting this matches all)')
    parser.add_argument('--replacement', type=str, default=None, help='Replace [query] with this')
    
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

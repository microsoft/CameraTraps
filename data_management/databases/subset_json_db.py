#
# subset_json_db.py
#
# Select a subset of images (and associated annotations) from a .json file
# in COCO Camera Traps format.
#
# To subset the .json files produced by our batch processing API, see 
# subset_json_detector_output.py
#
# Sample invocation:
#
# python subset_json_db.py "E:\Statewide_wolf_container\idfg_20190409.json" "E:\Statewide_wolf_container\idfg_20190409_clearcreek.json" "clearcreek" --ignore_case
#   
    
#%% Constants and imports

import json
import argparse

from tqdm import tqdm


#%% Functions

def subset_json_db(input_json, query, output_json=None, ignore_case=False):
    """
    Given a json file (or dictionary already loaded from a json file), produce a new 
    database containing only the images whose filenames contain the string 'query', 
    optionally writing that DB output to a new json file.
    """
    
    if ignore_case:
        query = query.lower()
        
    # Load the input file if necessary
    if isinstance(input_json,str):
        print('Loading input .json...')
        with open(input_json, 'r') as f:
            data = json.load(f)
    else:
        data = input_json

    # Find images matching the query
    images = []
    image_ids = set()
    
    for im in tqdm(data['images']):
        fn = im['file_name']
        if ignore_case:
            fn = fn.lower()
        if query in fn:
            images.append(im)
            image_ids.add(im['id'])        
    
    # Find annotations referring to those images
    annotations = []
    
    for ann in tqdm(data['annotations']):
        if ann['image_id'] in image_ids:
            annotations.append(ann)
    
    output_data = data
    output_data['images'] = images
    output_data['annotations'] = annotations
    
    # Write the output file if requested
    if output_json is not None:
        print('Writing output .json...')
        json.dump(output_data,open(output_json,'w'),indent=4)
        
    return output_data


#%% Interactive driver

if False:
    
    #%%
    
    input_json = r"E:\Statewide_wolf_container\idfg_20190409.json"
    output_json = r"E:\Statewide_wolf_container\idfg_20190409_clearcreek.json"
    query = 'clearcreek'
    ignore_case = True
    db = subset_json_db(input_json, query, output_json, ignore_case)
    

#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', type=str, help='Input file (a COCO Camera Traps .json file)')
    parser.add_argument('output_json', type=str, help='Output file')    
    parser.add_argument('query', type=str, help='Filename query')    
    parser.add_argument('--ignore_case', action='store_true')
    args = parser.parse_args()    
    
    subset_json_db(args.input_json,args.query,args.output_json,args.ignore_case)

if __name__ == '__main__':
    
    main()

#
# cct_to_csv.py
#
# "Converts" a COCO Camera Traps .json file to .csv, in quotes because 
# all kinds of assumptions are made here, and if you have a particular .csv
# format in mind, YMMV.  Most notably, does not include any bounding box information
# or any non-standard fields that may be present in the .json file.  Does not
# propagate information about sequence-level vs. image-level annotations.
#
# Does not assume access to the images, therefore does not open .jpg files to find
# datetime information if it's not in the metadata, just writes datetime as 'unknown'.
#

#%% Imports

import os
import json
from tqdm import tqdm
from collections import defaultdict 


#%% Main function

def cct_to_csv(input_file,output_file=None):

    if output_file is None:
        output_file = input_file + '.csv'

    ##%% Read input
    
    print('Loading input data')
    
    with open(input_file,'r') as f:
        input_data = json.load(f)
    
    
    ##%% Build internal mappings
    
    print('Processing input data')    
    
    images = input_data['images']
    
    category_id_to_name = {cat['id']:cat['name'] for cat in input_data['categories']}
    
    image_id_to_class_names = defaultdict(set)
    
    annotations = input_data['annotations']
                             
    # annotation = annotations[0]
    for annotation in tqdm(annotations):
        image_id = annotation['image_id']
        class_name = annotation['category_id']
        image_id_to_class_names[image_id].add(
            category_id_to_name[class_name])
        
    
    ##%% Write output file
    
    print('Writing output file')
    
    with open(output_file,'w') as f:
        
        f.write('relative_path,datetime,location,sequence_id,class_name\n')
        
        # im = images[0]
        for im in tqdm(images):
            
            file_name = im['file_name']
            class_names_set = image_id_to_class_names[im['id']]
            assert len(class_names_set) > 0
            
            if 'datetime' in im:
                datetime = im['datetime']
            else:
                datetime = 'unknown'
            
            if 'location' in im:
                location = im['location']
            else:
                location = 'unknown'
            
            if 'seq_id' in im:
                sequence_id = im['seq_id']
            else:
                sequence_id = 'unknown'
            
            # Write out one line per class:
            for class_name in class_names_set:
                f.write('{},{},{},{},{}\n'.format(file_name,
                   datetime,location,sequence_id,class_name))
            
            # ...for each class name
            
        # ...for each image
        
    # ...with open(output_file)    
            
# ...def cct_to_csv


#%% Interactive driver

if False:

    #%%
    
    input_dir = r"G:\temp\cct-to-csv"
    files = os.listdir(input_dir)
    files = [s for s in files if s.endswith('.json')]
    for fn in files:
        input_file = os.path.join(input_dir,fn)
        assert os.path.isfile(input_file)
        cct_to_csv(input_file)


#%% Command-line driver

import argparse

def main():

    parser = argparse.ArgumentParser(description=('"Convert" a COCO Camera Traps .json file to .csv (read code to see why "convert" is in quotes)'))

    parser.add_argument('input_file', type=str)                        
    parser.add_argument('--output_file', type=str, default=None)
    
    args = parser.parse_args()
    cct_to_csv(args.input_file,args.output_file)
    
if __name__ == '__main__':
    main()

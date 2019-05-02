#
# cct_json_to_filename_json.py
#
# Converts a .json file in COCO Camera Traps format to a .json-formatted list of
# relative file names.

#%% Constants and environment

import json
import os


#%% Main function

def convertJsonToStringList(inputFilename,outputFilename):

    assert os.path.isfile(inputFilename), '.json file {} does not exist'.format(inputFilename)
                         
    with open(inputFilename,'r') as f:
        data = json.load(f)
    
    images = data['images']
    
    filenames = [im['file_name'] for im in images]

    # json.dump(s,open(outputFilename,'w'))
        
    s =  json.dumps(filenames)    
    with open(outputFilename, 'w') as f:
        f.write(s)
        
    return s
    
    
#%% Command-line driver

import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFilename')
    parser.add_argument('outputFilename')
    
    args = parser.parse_args()    
    convertJsonToStringList(args.jsonFile,args)


if __name__ == '__main__':
    
    main()

#%% Interactive driver

if False:

    #%%    
    inputFilename = r"D:\wildlife_data\mcgill_test\mcgill_test.json"
    outputFilename = "D:\wildlife_data\mcgill_test\mcgill_test_list.json"
    convertJsonToStringList(inputFilename,outputFilename)
    print('Finished converting {} to {}'.format(inputFilename,outputFilename))

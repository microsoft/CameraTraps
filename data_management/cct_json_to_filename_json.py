#
# cct_json_to_filename_json.py
#
# Given a .json file in COCO Camera Traps format, outputs a .json-formatted list of
# relative file names present in the CCT file.
#

#%% Constants and environment

import json
import os
from itertools import compress


#%% Main function

def convertJsonToStringList(inputFilename,outputFilename=None,prepend='',bConfirmExists=False,
                            bForceForwardSlash=True,imageBase=''):

    assert os.path.isfile(inputFilename), '.json file {} does not exist'.format(inputFilename)
    if outputFilename is None:
        outputFilename = inputFilename + '_images.json'
        
    with open(inputFilename,'r') as f:
        data = json.load(f)
    
    images = data['images']
    
    filenames = [im['file_name'] for im in images]
    
    if bConfirmExists:
        bValid = [False] * len(filenames)
        for iFile,f in enumerate(filenames):
            fullPath = os.path.join(imageBase,f)
            if os.path.isfile(fullPath):
                bValid[iFile] = True
        nFilesTotal = len(filenames)
        filenames = list(compress(filenames, bValid))
        nFilesValid = len(filenames)
        print('Marking {} of {} as valid'.format(nFilesValid,nFilesTotal))

    filenames = [prepend + s for s in filenames]
    if bForceForwardSlash:
        filenames = [s.replace('\\','/') for s in filenames]
        
    # json.dump(s,open(outputFilename,'w'))
        
    s =  json.dumps(filenames)    
    with open(outputFilename, 'w') as f:
        f.write(s)
        
    return s,outputFilename
    
    
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
    prepend = '20190430cameratraps/'
    inputFilename = r"D:\wildlife_data\awc\awc_imageinfo.json"
    outputFilename = r"D:\wildlife_data\awc\awc_image_list.json"
    convertJsonToStringList(inputFilename,outputFilename,prepend=prepend,bConfirmExists=True,imageBase=r'D:\wildlife_data\awc')
    print('Finished converting {} to {}'.format(inputFilename,outputFilename))

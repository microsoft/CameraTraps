########
#
# remove_repeat_detections.py
#
# This script is a thin wrapper around find_repeat_detections.py, used to invoke
# that script a second time *after* you've manually deleted the true positives
# from the folder of images that it produces.
#
# If you want to use this script, we recommend that you read the user's guide:
#
# https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing/postprocessing/repeat_detection_elimination.ms
#
########

#%% Constants and imports

import argparse
import os

import find_repeat_detections
from api.batch_processing.postprocessing import find_repeat_detections
from api.batch_processing.postprocessing import RepeatDetectionOptions

#%% Main function

def remove_repeat_detections(inputFile,outputFile,filteringDir):
    
    options = RepeatDetectionOptions()
    options.filterFileToLoad = os.path.join(filteringDir,find_repeat_detections.DETECTION_INDEX_FILE_NAME)
    find_repeat_detections(inputFile, outputFile, options)


#%% Command-line driver

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile', help='.json file containing the original, unfiltered API results')
    parser.add_argument('outputFile', help='.json file to which you want to write the final, filtered API results')
    parser.add_argument('filteringDir', help='directory where you looked at lots of images and decided which ones were really false positives')
    
    args = parser.parse_args()
    remove_repeat_detections(args.inputFile, args.outputFile, args.filteringDir)


if __name__ == '__main__':
    main()

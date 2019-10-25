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
from api.batch_processing.postprocessing import find_repeat_detections


#%% Main function

def remove_repeat_detections(inputFile,outputFile,filteringDir):

    assert os.path.isfile(inputFile)    
    assert os.path.isdir(filteringDir)    
    options = find_repeat_detections.RepeatDetectionOptions()
    options.filterFileToLoad = os.path.join(filteringDir,find_repeat_detections.DETECTION_INDEX_FILE_NAME)
    options.bWriteFilteringFolder = False
    find_repeat_detections.find_repeat_detections(inputFile, outputFile, options)


#%% Interactive driver

# python remove_repeat_detections.py "F:\wpz\6714_detections_wpz_all_20191015233705.SUCP_subset.json" "F:\wpz\6714_detections_wpz_all_20191015233705.SUCP_subset_filtered.json" "F:\wpz\rde\filtering_2019.10.24.16.52.54"
inputFile = r"F:\wpz\6714_detections_wpz_all_20191015233705.SUCP_subset.json" "F:\wpz\6714_detections_wpz_all_20191015233705.SUCP_subset_filtered.json"
outputFile = r"F:\wpz\6714_detections_wpz_all_20191015233705.SUCP_subset_filtered.json"
filteringDir = "F:\wpz\rde\filtering_2019.10.24.16.52.54"
remove_repeat_detections(inputFile,outputFile,filteringDir)


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

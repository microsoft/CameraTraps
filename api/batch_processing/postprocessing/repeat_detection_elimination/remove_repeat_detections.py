########
#
# remove_repeat_detections.py
#
# Used after running find_repeat_detections, then manually filtering the results,
# to create a final filtered output file.
#
# If you want to use this script, we recommend that you read the user's guide:
#
# https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing/postprocessing/repeat_detection_elimination.ms
#
########

#%% Constants and imports

import argparse
import os
from api.batch_processing.postprocessing.repeat_detection_elimination import repeat_detections_core


#%% Main function

def remove_repeat_detections(inputFile,outputFile,filteringDir,options=None):

    assert os.path.isfile(inputFile), "Can't find file {}".format(inputFile)
    assert os.path.isdir(filteringDir), "Can't find folder {}".format(filteringDir)
    if options is None:
        options = repeat_detections_core.RepeatDetectionOptions()
    options.filterFileToLoad = os.path.join(filteringDir,repeat_detections_core.DETECTION_INDEX_FILE_NAME)
    options.bWriteFilteringFolder = False
    repeat_detections_core.find_repeat_detections(inputFile, outputFile, options)


#%% Interactive driver

if False:
    
    #%%
    
    # python remove_repeat_detections.py "F:\wpz\6714_detections_wpz_all_20191015233705.SUCP_subset.json" "F:\wpz\6714_detections_wpz_all_20191015233705.SUCP_subset_filtered.json" "F:\wpz\rde\filtering_2019.10.24.16.52.54"
    inputFile = r''
    outputFile = r''
    filteringDir = r''
    remove_repeat_detections(inputFile,outputFile,filteringDir)


#%% Command-line driver

import sys

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile', help='.json file containing the original, unfiltered API results')
    parser.add_argument('outputFile', help='.json file to which you want to write the final, filtered API results')
    parser.add_argument('filteringDir', help='directory where you looked at lots of images and decided which ones were really false positives')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    remove_repeat_detections(args.inputFile, args.outputFile, args.filteringDir)


if __name__ == '__main__':
    main()

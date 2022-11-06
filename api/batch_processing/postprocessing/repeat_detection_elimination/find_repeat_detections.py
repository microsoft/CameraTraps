########
#
# find_repeat_detections.py
#
# If you want to use this script, we recommend that you read the user's guide:
#
# https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing/postprocessing/repeat_detection_elimination.ms
#
# Really, don't try to run this script without reading the user's guide, you'll think 
# it's more magical than it is. 
#
# This script looks through a sequence of detections in the API output json file, and finds 
# candidates that might be "repeated false positives", i.e. that random branch that the 
# detector thinks is an animal/person/vehicle.
#
# Typically after running this script, you would do a manual step to remove 
# true positives, then run remove_repeat_detections to produce a final output file.
#
# There's no way that statement was self-explanatory; see the user's guide.
#
########

#%% Constants and imports

import argparse
import os
import sys

import ct_utils
from api.batch_processing.postprocessing.repeat_detection_elimination import repeat_detections_core


#%% Interactive driver

if False:
    
    #%%

    baseDir = ''

    options = repeat_detections_core.RepeatDetectionOptions()
    options.bRenderHtml = True
    options.imageBase = baseDir
    options.outputBase = os.path.join(baseDir, 'repeat_detections')
    options.filenameReplacements = {}  # E.g., {'20190430cameratraps\\':''}

    options.confidenceMin = 0.85
    options.confidenceMax = 1.01
    options.iouThreshold = 0.85
    options.occurrenceThreshold = 8
    options.maxSuspiciousDetectionSize = 0.2

    options.filterFileToLoad = ''
    options.filterFileToLoad = os.path.join(baseDir,'...\detectionIndex.json')

    options.debugMaxDir = -1
    options.debugMaxRenderDir = -1
    options.debugMaxRenderDetection = -1
    options.debugMaxRenderInstance = -1

    options.bParallelizeComparisons = False
    options.bParallelizeRendering = False
    options.excludeClasses = [2]

    inputFilename = os.path.join(baseDir, 'blah.json')
    outputFilename = os.path.join(baseDir, 'blah.json')

    results = repeat_detections_core.find_repeat_detections(inputFilename, outputFilename, options)


#%% Command-line driver

def main():
    
    defaultOptions = repeat_detections_core.RepeatDetectionOptions()

    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile')
    parser.add_argument('--outputFile', action='store', type=str, default=None,
                        help=".json file to write filtered results to... don't use this " + \
                            "if you're going to do manual review of the repeat detection images")
    parser.add_argument('--imageBase', action='store', type=str, default='',
                        help='Image base dir, relevant if renderHtml is True or if " + \
                            "omitFilteringFolder is not set')
    parser.add_argument('--outputBase', action='store', type=str, default='',
                        help='HTML or filtering folder output dir')
    parser.add_argument('--filterFileToLoad', action='store', type=str, default='',  
                        help='Path to detectionIndex.json, which should be inside a ' + \
                            'folder of images that are manually verified to _not_ ' + \
                            'contain valid animals')

    parser.add_argument('--confidenceMax', action='store', type=float,
                        default=defaultOptions.confidenceMax,
                        help='Detection confidence threshold; don\'t process anything above this')
    parser.add_argument('--confidenceMin', action='store', type=float,
                        default=defaultOptions.confidenceMin,
                        help='Detection confidence threshold; don\'t process anything below this')
    parser.add_argument('--iouThreshold', action='store', type=float,
                        default=defaultOptions.iouThreshold,
                        help='Detections with IOUs greater than this are considered ' + \
                            '"the same detection"')
    parser.add_argument('--occurrenceThreshold', action='store', type=int,
                        default=defaultOptions.occurrenceThreshold,
                        help='More than this many near-identical detections in a group ' + \
                            '(e.g. a folder) is considered suspicious')
    parser.add_argument('--nWorkers', action='store', type=int,
                        default=defaultOptions.nWorkers,
                        help='Level of parallelism for rendering and IOU computation')
    parser.add_argument('--maxSuspiciousDetectionSize', action='store', type=float,
                        default=defaultOptions.maxSuspiciousDetectionSize,
                        help='Detections larger than this fraction of image area are not ' + \
                            'considered suspicious')

    parser.add_argument('--renderHtml', action='store_true',
                        dest='bRenderHtml', help='Should we render HTML output?')
    parser.add_argument('--omitFilteringFolder', action='store_false',
                        dest='bWriteFilteringFolder',
                        help='Should we create a folder of rendered detections for post-filtering?')
    parser.add_argument('--excludeClasses', action='store', nargs='+', type=int,
                        default=defaultOptions.excludeClasses,
                        help='List of classes (ints) to exclude from analysis, separated by spaces')

    parser.add_argument('--nDirLevelsFromLeaf', default=0, type=int,
                        help='Number of levels from the leaf folders to use for repeat ' + \
                            'detection (0 == leaves)')

    parser.add_argument('--debugMaxDir', action='store', type=int, default=-1)
    parser.add_argument('--debugMaxRenderDir', action='store', type=int, default=-1)
    parser.add_argument('--debugMaxRenderDetection', action='store', type=int, default=-1)
    parser.add_argument('--debugMaxRenderInstance', action='store', type=int, default=-1)

    parser.add_argument('--forceSerialComparisons', action='store_false',
                        dest='bParallelizeComparisons')
    parser.add_argument('--forceSerialRendering', action='store_false',
                        dest='bParallelizeRendering')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    # Convert to an options object
    options = repeat_detections_core.RepeatDetectionOptions()

    ct_utils.args_to_object(args, options)

    repeat_detections_core.find_repeat_detections(args.inputFile, args.outputFile, options)


if __name__ == '__main__':
    main()

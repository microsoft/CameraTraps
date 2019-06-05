#
# prepare_api_output_for_timelapse.py
#
# Takes output from the batch API and does some conversions to prepare 
# it for use in Timelapse.
#
# Specifically:
#
# * Removes the class field from each bounding box
# * Optionally does query-based subsetting of rows
# * Optionally does a search and replace on filenames
# * Replaces backslashes with forward slashes
# * Renames "detections" to "predicted_boxes"
#

#%% Constants and imports

# Python standard
import csv
import os

# pip-installable
from tqdm import tqdm

# AI4E repos, expected to be available on the path
from detection.detection_eval.load_api_results import load_api_results
import matlab_porting_tools as mpt


#%% Helper classes

class SubsetDetectorOutputOptions:
    
    replacement = None
    prepend = ''
    replacement = ''
    query = ''
    removeClassLabel = False
    nRows = None
    temporaryMatchColumn = '_bMatch'
    
    
#%% Helper functions

def process_row(row,options):
    
    if options.removeClassLabel:
            
        detections = row['detections']
        for iDetection,detection in enumerate(detections):
            detections[iDetection] = detection[0:5]
            
    # If there's no query, we're just pre-pending
    if len(options.query) == 0:
        
        row[options.temporaryMatchColumn] = True
        if len(options.prepend) > 0:
            row['image_path'] = options.prepend + row['image_path']
        
    else:
        
        fn = row['image_path']
        if options.query in os.path.normpath(fn):
            
            row[options.temporaryMatchColumn] = True
            
            if len(options.prepend) > 0:
                row['image_path'] = options.prepend + row['image_path']
            
            if options.replacement is not None:
                fn = fn.replace(options.query,options.replacement)
                row['image_path'] = fn
        
    return row


#%% Main function
                
def subset_detector_output(inputFilename,outputFilename,options):

    if options is None:    
        options = SubsetDetectorOutputOptions()
            
    options.query = os.path.normpath(options.query)
    
    detectionResults = load_api_results(inputFilename,nrows=options.nRows)
    nRowsLoaded = len(detectionResults)
    
    # Create a temporary column we'll use to mark the rows we want to keep
    detectionResults[options.temporaryMatchColumn] = False
    
    # This is the main loop over rows
    tqdm.pandas()
    detectionResults = detectionResults.progress_apply(lambda x: process_row(x,options), axis=1)
        
    print('Finished main loop, post-processing output')
    
    # Trim to matching rows
    detectionResults = detectionResults.loc[detectionResults[options.temporaryMatchColumn]]
    print('Trimmed to {} matching rows (from {})'.format(len(detectionResults),nRowsLoaded))
    
    detectionResults = detectionResults.drop(columns=options.temporaryMatchColumn)
    
    # Timelapse legacy issue; we used to call this column 'predicted_boxes'
    detectionResults.rename(columns={'detections':'predicted_boxes'},inplace=True)    
    detectionResults['image_path'] = detectionResults['image_path'].str.replace('\\','/')
    
    # Write output  
    # write_api_results(detectionResults,outputFilename)
    detectionResults.to_csv(outputFilename,index=False,quoting=csv.QUOTE_MINIMAL)
        
    return detectionResults


#%% Interactive driver
                
if False:

    #%%   
    
    inputFilename = r"D:\wildlife_data\idfg\idfg_7517_detections.refiltered_2019.05.17.15.31.28.csv"
    outputFilename = mpt.insert_before_extension(inputFilename,'for_timelapse_clearcreek')
    
    options = SubsetDetectorOutputOptions()
    options.prepend = ''
    options.replacement = None 
    options.query = 'ClearCreek_mustelids'
    options.nRows = None 
    options.removeClassLabel = True
        
    detectionResults = subset_detector_output(inputFilename,outputFilename,options)
    print('Done, found {} matches'.format(len(detectionResults)))

    
#%% Command-line driver (outdated)

import argparse
import inspect

# Copy all fields from a Namespace (i.e., the output from parse_args) to an object.  
#
# Skips fields starting with _.  Does not check existence in the target object.
def argsToObject(args, obj):
    
    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            # print('Setting {} to {}'.format(n,v))
            setattr(obj, n, v);

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile')
    parser.add_argument('outputFile')
    parser.add_argument('query')
    
    parser.add_argument('--replacement', action='store', type=str, default=None)
    args = parser.parse_args()    
    
    # Convert to an options object
    options = SubsetDetectorOutputOptions
    argsToObject(args,options)
    
    subset_detector_output(args.inputFile,args.outputFile,args.query,options)

if __name__ == '__main__':
    
    main()

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
# Note that "relative" paths as interpreted by Timelapse aren't strictly relative as
# of 6/5/2019.  If your project is in:
#
# c:\myproject
#
# ...and your .tdb file is:
#
# c:\myproject\blah.tdb
#
# ...and you have an image at:
#
# c:\myproject\imagefolder1\img.jpg
#
# The .csv that Timelapse sees should refer to this as:
#
# myproject/imagefolder1/img.jpg
# 
# ...*not* as:
#
# imagefolder1/img.jpg
#
# Hence all the search/replace functionality in this script.  It's very straightforward
# once you get this and doesn't take time, but it's easy to forget to do this.  This will
# be fixed in an upcoming release.
#

#%% Constants and imports

# Python standard
import csv
import os

# pip-installable
from tqdm import tqdm

# AI4E repos, expected to be available on the path
from api.batch_processing.load_api_results import load_api_results
import matlab_porting_tools as mpt


#%% Helper classes

class TimelapsePrepOptions:
    
    # Only process rows matching this query (if not None); this is processed
    # after applying os.normpath to filenames.
    query = None
    
    # If not none, replace the query token with this
    replacement = None
    
    # If not none, prepend matching filenames with this
    prepend = None
    
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
    if options.query is None:
        
        row[options.temporaryMatchColumn] = True
        if options.prepend is not None:
            row['image_path'] = options.prepend + row['image_path']
        
    else:
        
        fn = row['image_path']
        if options.query in os.path.normpath(fn):
            
            row[options.temporaryMatchColumn] = True
            
            if options.prepend is not None:
                row['image_path'] = options.prepend + row['image_path']
            
            if options.replacement is not None:
                fn = fn.replace(options.query,options.replacement)
                row['image_path'] = fn
        
    return row


#%% Main function
                
def prepare_api_output_for_timelapse(inputFilename,outputFilename,options):

    if options is None:    
        options = TimelapsePrepOptions()
    
    if options.query is not None:
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
    
    inputFilename = r"D:\temp\demo_images\snapshot_serengeti\detections.csv"
    outputFilename = mpt.insert_before_extension(inputFilename,'for_timelapse')
    
    options = TimelapsePrepOptions()
    options.prepend = ''
    options.replacement = 'snapshot_serengeti'
    options.query = r'd:\temp\demo_images\snapshot_serengeti'
    options.nRows = None 
    options.removeClassLabel = True
        
    detectionResults = prepare_api_output_for_timelapse(inputFilename,outputFilename,options)
    print('Done, found {} matches'.format(len(detectionResults)))

    
#%% Command-line driver (** outdated **)

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
    parser.add_argument('--query', action='store', type=str, default=None)
    parser.add_argument('--prepend', action='store', type=str, default=None)
    parser.add_argument('--replacement', action='store', type=str, default=None)
    args = parser.parse_args()    
    
    # Convert to an options object
    options = TimelapsePrepOptions()
    argsToObject(args,options)
    
    prepare_api_output_for_timelapse(args.inputFile,args.outputFile,args.query,options)

if __name__ == '__main__':
    
    main()

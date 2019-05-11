#
# load_api_results.py
#
# Loads the output of the batch processing API
#
# Currently just a simple table read, but since this format will likely grow more complex, 
# we should be doing all loading via this file
#

#%% Constants and imports

import pandas as pd
import json
import os

headers = ['image_path','max_confidence','detections']
    

#%% Functions

def load_api_results(filename,normalize_paths=True,filename_replacements={}):
    
    print('Loading API results from {}'.format(filename))
    
    detection_results = pd.read_csv(filename)
    
    print('De-serializing API results from {}'.format(filename))
    
    # Sanity-check that this is really a detector output file
    for s in ['image_path','max_confidence','detections']:
        assert s in detection_results.columns
    
    # Normalize paths to simplify comparisons later
    if normalize_paths:
        detection_results['image_path'] = detection_results['image_path'].apply(os.path.normpath)
        
    # De-serialize detections
    detection_results['detections'] = detection_results['detections'].apply(json.loads)
        
    # Optionally replace some path tokens to match local paths to the original blob structure
    # string_to_replace = list(options.detector_output_filename_replacements.keys())[0]
    for string_to_replace in filename_replacements:
        
        replacement_string = filename_replacements[string_to_replace]
        
        # TODO: hit some silly issues with vectorized str() and escaped characters, vectorize
        # this later.
        #
        # detection_results['image_path'].str.replace(string_to_replace,replacement_string)
        # iRow = 0
        for iRow in range(0,len(detection_results)):
            row = detection_results.iloc[iRow]
            fn = row['image_path']
            fn = fn.replace(string_to_replace,replacement_string)
            detection_results.at[iRow,'image_path'] = fn
    
    print('Finished loading and de-serializing API results from {}'.format(filename))    
    
    return detection_results

def write_api_results(detection_results,filename):
    
    print('Writing detection results to {}'.filename)
        
    # Write the output .csv
    with open(filename,'w')  as csvf:
        
        # Likely to get read in pandas, don't use '#'
        # headerString = '#' + ','.join(options.expectedHeaders)
        headerString = ','.join(headers)
        
        # Write the header
        csvf.write(headerString + '\n')
        
        for iRow,row in detection_results.iterrows():
            csvf.write('"' + row['image_path'] + '",' + str(row['max_confidence']) + ',"' + json.dumps(row['detections']) + '"\n')
    
    print('Finished writing detection results to {}'.filename)
    
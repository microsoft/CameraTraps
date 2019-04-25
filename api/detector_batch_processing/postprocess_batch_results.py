########
#
# postprocess_batch_results.py
#
# Given a .csv file representing the output from the batch API, do one or more of 
# the following:
#
# * Eliminate "suspicious detections", i.e. detections repeated numerous times with
#   unrealistically limited movement
#
# * Evaluate detector precision/recall, optionally rendering results (requires ground truth)
#
# * Sample true/false positives/negatives and render to html (requires ground truth)
#
# * Sample detections/non-detections 
#
########

#%% Constants and imports

import json
import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


#%% To be moved into options/inputs

detector_output_file = r'd:\temp\8471_detections.csv'
image_base_dir = r'd:\wildlife_data\mcgill_test'
ground_truth_json_file = r'd:\wildlife_data\mcgill_test\mcgill_test.json'
output_dir = r'd:\temp\postprocessing_tmp'


#%% Load ground truth if available

ground_truth_db = None

if len(ground_truth_json_file) > 0:
        
    ground_truth_db = json.load(open(ground_truth_json_file))
    
    # Normalize paths to simplify comparisons later
    for im in ground_truth_db['images']:
        im['file_name'] = os.path.normpath(im['file_name'])
    
    ### Build useful mappings to facilitate working with the DB
    
    # Category ID <--> name
    ground_truth_cat_id_to_name = {cat['id']: cat['name'] for cat in ground_truth_db['categories']}
    ground_truth_name_to_cat_id = {cat['name']: cat['id'] for cat in ground_truth_db['categories']}
    
    # Image filename --> ID
    ground_trith_filename_to_id = {im['file_name']: im['id'] for im in ground_truth_db['images']}
    
    ground_truth_image_id_to_category_ids = defaultdict(list)  # each image could potentially have multiple species, hence using lists

    # Image ID --> categories
    for ann in ground_truth_db['annotations']:
        ground_truth_image_id_to_category_ids[ann['image_id']].append(ann['category_id'])

    print('Finished loading and indexing ground truth')


#%% Load detection results

detection_results = pd.read_csv(detector_output_file)

# Sanity-check that this is really a detector output file
for s in ['image_path','max_confidence','detections']:
    assert s in detection_results.columns
    
    
#%% If ground truth is available, merge it into the detection results

# Error on any matching failures


#%% Find suspicious detections



#%% Evaluate precision/recall, optionally rendering results


#%% Sample true/false positives/negatives and render to html


#%% Sample detections/non-detections
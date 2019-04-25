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
from enum import Enum
from tqdm import tqdm
from collections import defaultdict


#%% To be moved into options/inputs

detector_output_file = r'd:\temp\8471_detections.csv'
image_base_dir = r'd:\wildlife_data\mcgill_test'
ground_truth_json_file = r'd:\wildlife_data\mcgill_test\mcgill_test.json'
output_dir = r'd:\temp\postprocessing_tmp'

negative_classes = ['empty']
confidence_threshold = 0.85


#%% Helper classes and functions

# Flags used to mark images as positive or negative for P/R analysis (according
# to ground truth and/or detector output)
class DetectionStatus(Enum):
    
    # This image is a negative
    DS_NEGATIVE = 0
    
    # This image is a positive
    DS_POSITIVE = 1
    
    # This image has annotations suggesting both negative and positive
    DS_AMBIGUOUS = 2
    
    # This image is not annotated
    DS_UNKNOWN = 3


class IndexedJsonDb:
    """
    Wrapper for a COCO Camera Traps database.
    
    Handles boilerplate dictionary creation that we do almost every time we load 
    a .json database.
    """
    
    # The underlying .json db
    db = None
    
    # Useful dictionaries
    cat_id_to_name = None
    cat_name_to_id = None
    filename_to_id = None
    image_id_to_annotations = None

    def __init__(self,jsonFilename,b_normalize_paths=False):
       
        self.db = json.load(open(jsonFilename))
    
        if b_normalize_paths:
            # Normalize paths to simplify comparisons later
            for im in self.db['images']:
                im['file_name'] = os.path.normpath(im['file_name'])
        
        ### Build useful mappings to facilitate working with the DB
        
        # Category ID <--> name
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.db['categories']}
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in self.db['categories']}
        
        # Image filename --> ID
        self.filename_to_id = {im['file_name']: im['id'] for im in self.db['images']}
        
        # Each image can potentially multiple annotations, hence using lists
        self.image_id_to_annotations = defaultdict(list)
        
        # Image ID --> image object
        self.image_id_to_image = {im['id'] : im for im in self.db['images']}
        
        # Image ID --> annotations
        for ann in self.db['annotations']:
            self.image_id_to_annotations[ann['image_id']].append(ann)
            
            
def mark_detection_status(indexed_db,negative_classes=['empty']):
    """
    For each image in indexed_db.db['images'], add a '_detection_status' field
    to indicate whether to treat this image as positive, negative, ambiguous,
    or unknown.
    
    Makes modifications in-place.
    
    returns (nNegative,nPositive,nUnknown,nAmbiguous)
    """
          
    nUnknown = 0
    nAmbiguous = 0
    nPositive = 0
    nNegative = 0
 
    db = indexed_db.db
    for im in db['images']:
        
        image_id = im['id']
        annotations = indexed_db.image_id_to_annotations[image_id]
        image_categories = [ann['category_id'] for ann in annotations]
        
        image_status = DetectionStatus.DS_UNKNOWN
        
        if len(image_categories) == 0:
            
            image_status = DetectionStatus.DS_UNKNOWN
            
        else:            
            
            for cat_id in image_categories:
                
                cat_name = indexed_db.cat_id_to_name[cat_id]            
                
                if cat_name in negative_classes:                    
                    if image_status == DetectionStatus.DS_UNKNOWN:                        
                        image_status = DetectionStatus.DS_NEGATIVE
                    elif image_status == DetectionStatus.DS_POSITIVE:
                        image_status = DetectionStatus.DS_AMBIGUOUS                    
                else:                    
                    if image_status == DetectionStatus.DS_UNKNOWN:                        
                        image_status = DetectionStatus.DS_POSITIVE
                    elif image_status == DetectionStatus.DS_NEGATIVE:
                        image_status = DetectionStatus.DS_AMBIGUOUS
        
        if image_status == DetectionStatus.DS_NEGATIVE:
            nNegative += 1
        elif image_status == DetectionStatus.DS_POSITIVE:
            nPositive += 1
        elif image_status == DetectionStatus.DS_UNKNOWN:
            nUnknown += 1
        elif image_status == DetectionStatus.DS_AMBIGUOUS:
            nAmbiguous += 1

    return (nNegative,nPositive,nUnknown,nAmbiguous)


#%% Load ground truth if available

ground_truth_indexed_db = None

if len(ground_truth_json_file) > 0:
        
    ground_truth_indexed_db = IndexedJsonDb(ground_truth_json_file,True)
    
    # Mark images in the ground truth as positive or negative
    (nNegative,nPositive,nUnknown,nAmbiguous) = mark_detection_status(ground_truth_indexed_db,
        negative_classes)
    print('Finished loading and indexing ground truth: {} negative, {} positive, {} unknown, {} ambiguous'.format(
            nNegative,nPositive,nUnknown,nAmbiguous))


#%% Load detection results

detection_results = pd.read_csv(detector_output_file)

# Sanity-check that this is really a detector output file
for s in ['image_path','max_confidence','detections']:
    assert s in detection_results.columns

# Normalize paths to simplify comparisons later
detection_results['image_path'] = detection_results['image_path'].apply(os.path.normpath)

# Add a column (pred_detection_label) to indicate predicted detection status
# detection_results['pred_detection_label'] = DetectionStatus.DS_UNKNOWN

import numpy as np
detection_results['pred_detection_label'] = \
    np.where(detection_results['max_confidence'] >= confidence_threshold,
             DetectionStatus.DS_POSITIVE, DetectionStatus.DS_NEGATIVE)

nPositives = sum(detection_results['pred_detection_label'] == DetectionStatus.DS_POSITIVE)
print('Finished loading and preprocessing {} rows from detector output, predicted {} positives'.format(
        len(detection_results),nPositives))


#%% Find suspicious detections



#%% If ground truth is available, match it to the detection results

class DetectionGroundTruth:

    gt_image_id = None
    gt_presence_label = None
    gt_class_label = None


# For now, error on any matching failures

# Add columns gt_image_id, gt_presence_label, gt_class_label



#%% Evaluate precision/recall, optionally rendering results


#%% Sample true/false positives/negatives and render to html


#%% Sample detections/non-detections
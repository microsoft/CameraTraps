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
# * Sample detections/non-detections and render to html (when ground truth isn't available)
#
########

#%% Constants and imports

import json
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score
from sklearn.utils.fixes import signature

#%% To be moved into options/inputs

detector_output_file = r'd:\temp\8471_detections.csv'
image_base_dir = r'd:\wildlife_data\mcgill_test'
ground_truth_json_file = r'd:\wildlife_data\mcgill_test\mcgill_test.json'
output_dir = r'd:\temp\postprocessing_tmp'

negative_classes = ['empty']
confidence_threshold = 0.85

# Used for summary statistics only
target_recall = 0.9


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

        im['_detection_status'] = image_status
        
    return (nNegative,nPositive,nUnknown,nAmbiguous)


#%% Prepare output dir
    
os.makedirs(output_dir,exist_ok=True)


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

    

#%% Fork here depending on whether or not ground truth is available
    
# If we have ground truth, we'll compute precision/recall and sample tp/fp/tn/fn.
#
# Otherwise we'll just visualize detections/non-detections.
    
if ground_truth_indexed_db is not None:

    #%% Make sure we can match ground truth to detection results

    detector_files = detection_results['image_path'].to_list()
        
    # For now, error on any matching failures, at some point we can decide 
    # how to handle "partial" ground truth.  All or none for now.
    for fn in detector_files:
        assert fn in ground_truth_indexed_db.filename_to_id    
    
    print('Confirmed filename matches to ground truth for {} files'.format(len(detector_files)))
    

    #%% Compute precision/recall
    
    # numpy array of detection probabilities
    p_detection = detection_results['max_confidence'].values
    n_detections = len(p_detection)
    
    # numpy array of bools (0.0/1.0)
    gt_detections = np.zeros(n_detections,dtype=float)
    
    for iDetection,fn in enumerate(detector_files):
        image_id = ground_truth_indexed_db.filename_to_id[fn]
        image = ground_truth_indexed_db.image_id_to_image[image_id]
        detection_status = image['_detection_status']
        
        if detection_status == DetectionStatus.DS_NEGATIVE:
            gt_detections[iDetection] = 0.0
        elif detection_status == DetectionStatus.DS_POSITIVE:
            gt_detections[iDetection] = 1.0
        else:
            gt_detections[iDetection] = -1.0
            
    # Don't include ambiguous/unknown ground truth in precision/recall analysis
    b_valid_ground_truth = gt_detections >= 0.0
    
    p_detection_pr = p_detection[b_valid_ground_truth]
    gt_detections_pr = gt_detections[b_valid_ground_truth]
    
    print('Including {} of {} values in p/r analysis'.format(np.sum(b_valid_ground_truth),
          len(b_valid_ground_truth)))
    
    precisions, recalls, thresholds = precision_recall_curve(gt_detections_pr, p_detection_pr)
    
    # For completeness, include the result at a confidence threshold of 1.0
    thresholds = np.append(thresholds, [1.0])

    precisions_recalls = pd.DataFrame(data={
            'confidence_threshold': thresholds,
            'precision': precisions,
            'recall': recalls
        })
    
    # Compute and print summary statistics
    average_precision = average_precision_score(gt_detections_pr, p_detection_pr)
    print('Average precision: {}'.format(average_precision))

    # Thresholds go up throughout precisions/recalls/thresholds; find the last
    # value where recall is at or above target.  That's our precision @ target recall.
    target_recall = 0.9
    b_above_target_recall = np.where(recalls >= target_recall)
    if not np.any(b_above_target_recall):
        precision_at_target_recall = 0.0
    else:
        i_target_recall = np.argmax(b_above_target_recall)
        precision_at_target_recall = precisions[i_target_recall]
    print('Precision at {} recall: {}'.format(target_recall,precision_at_target_recall))    
    
    cm = confusion_matrix(gt_detections_pr, np.array(p_detection_pr) > confidence_threshold)

    # Flatten the confusion matrix
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print('At a confidence threshold of {:.2f}, precision={:.2f}, recall={:.2f}, f1={:.2f}'.format(
            confidence_threshold,precision, recall, f1))
        
    
    #%% Render output
    
    # Write p/r table to .csv file in output directory
    pr_table_filename = os.path.join(output_dir, 'prec_recall.csv')
    precisions_recalls.to_csv(pr_table_filename, index=False)

    # Write precision/recall plot to .png file in output directory
    step_kwargs = ({'step': 'post'})
    plt.step(recalls, precisions, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recalls, precisions, alpha=0.2, color='b', **step_kwargs)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    t = 'Precision-Recall curve: AP={:0.2f}, P@{:0.2f}={:0.2f}'.format(
            average_precision, target_recall, precision_at_target_recall)
    plt.title(t)
    pr_figure_filename =os.path.join(output_dir, 'prec_recall.png')
    plt.savefig(pr_figure_filename)
    # plt.show()
        
        
    #%% Sample true/false positives/negatives and render to html
    

else:
    
    #%% Sample detections/non-detections
    
    pass
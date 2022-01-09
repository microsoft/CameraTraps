#
# umn-pr-analysis.py
#
# Precision/recall analysis for UMN data
#

#%% Imports and constants

import os
import json
import pandas as pd
import shutil

from tqdm import tqdm
from pathlib import Path

from sklearn.metrics import precision_recall_curve, confusion_matrix
from IPython.core.display import display
import visualization.plot_utils as plot_utils
import numpy as np
import matplotlib.pyplot as plt

analysis_base = "f:\\"
image_base = os.path.join(analysis_base,'2021.11.24-images\jan2020')
results_file_filtered = os.path.join(analysis_base,'umn-gomez-reprise-2021-11-272021.11.24-images_detections.filtered_rde_0.60_0.85_10_0.20.json')
# results_file = results_file_filtered
results_file = os.path.join(analysis_base,'umn-gomez-reprise-2021-11-272021.11.24-images_detections.json')
ground_truth_file = os.path.join(analysis_base,'images_hv_jan2020_reviewed_force_nonblank.csv')

# For two deployments, we're only processing imagse in the "detections" subfolder
detection_only_deployments = ['N23','N32']
deployments_to_ignore = ['N18','N28']

# String to remove from MegaDetector results
replacement_string = '2021.11.24-images/jan2020/'
alt_string = 'jul2020'

MISSING_COMMON_NAME_TOKEN = 'MISSING'
    
assert os.path.isfile(results_file)
assert os.path.isfile(ground_truth_file)
assert os.path.isdir(image_base)


#%% Enumerate deployment folders

deployment_folders = os.listdir(image_base)
deployment_folders = [s for s in deployment_folders if os.path.isdir(os.path.join(image_base,s))]
deployment_folders = set(deployment_folders)
print('Listed {} deployment folders'.format(len(deployment_folders)))


#%% Load MD results 

with open(results_file,'r') as f:
    md_results = json.load(f)
n_images_in_results = len(md_results['images'])

valid_images = []
n_invalid_images = 0

# im = md_results['images'][0]
for im in md_results['images']:  
    im['file'] = im['file'].replace('\\','/')
    if replacement_string not in im['file']:
        if alt_string is not None:
            assert alt_string in im['file']
        n_invalid_images += 1
        continue
    im['file'] = im['file'].replace(replacement_string,'')
    valid_images.append(im)

md_results['images'] = valid_images
print('Loaded {} MD results from a total of {} (ignored {})'.format(
    len(md_results['images']),n_images_in_results,n_invalid_images))

category_name_to_id = {}
category_id_to_name = md_results['detection_categories']
for category_id in md_results['detection_categories'].keys():
    category_name = md_results['detection_categories'][category_id]
    category_name_to_id[category_name] = category_id

    
#%% Load ground truth

ground_truth_df = pd.read_csv(ground_truth_file)

print('Loaded {} ground truth annotations'.format(
    len(ground_truth_df)))

# i_row = 0; row = ground_truth_df.iloc[i_row]
for i_row,row in tqdm(ground_truth_df.iterrows()):
    if not isinstance(row['common_name'],str):
        print('Warning: missing common name for {}'.format(row['filename']))
        row['common_name'] = MISSING_COMMON_NAME_TOKEN
    
    
#%% Create relative paths for ground truth data

# Some deployment folders have no subfolders, e.g. this is a valid file name:
# 
# M00/01010132.JPG
#
# But some deployment folders have subfolders, e.g. this is also a valid file name:
#
# N17/100EK113/07160020.JPG
#
# So we can't find files by just concatenating folder and file names, we have to enumerate and explicitly
# map what will appear in the ground truth as "folder/filename" to complete relative paths.

deployment_name_to_file_mappings = {}

n_filenames_ignored = 0
n_deployments_ignored = 0

# deployment_name = list(deployment_folders)[0]
for deployment_name in tqdm(deployment_folders):
    
    file_mappings = {}
    
    if deployment_name in deployments_to_ignore:
        print('Ignoring deployment {}'.format(deployment_name))
        n_deployments_ignored += 1
        continue
    
    # Enumerate all files in this folder
    absolute_deployment_folder = os.path.join(image_base,deployment_name)
    assert os.path.isdir(absolute_deployment_folder)
    
    files = list(Path(absolute_deployment_folder).rglob('*'))
    files = [p for p in files if not p.is_dir()]
    files = [str(s) for s in files]
    files = [s.replace('\\','/') for s in files]
    # print('Enumerated {} files for deployment {}'.format(len(files),deployment_name))
    
    # filename = files[100]
    for filename in files:
        
        if deployment_name in detection_only_deployments and 'detection' not in filename:
            n_filenames_ignored += 1
            continue
        
        if '.DS_Store' in filename:
            n_filenames_ignored += 1
            continue
        
        relative_path = os.path.relpath(filename,absolute_deployment_folder).replace('\\','/')
        image_name = relative_path.split('/')[-1]
        assert image_name not in file_mappings, 'Redundant image name {} in deployment {}'.format(
            image_name,deployment_name)
        assert '\\' not in relative_path
        file_mappings[image_name] = relative_path
    
    # ...for each file in this deployment
    
    deployment_name_to_file_mappings[deployment_name] = file_mappings

# ...for each deployment

print('Processed deployments, ignored {} deployments and {} files'.format(
    n_deployments_ignored,n_filenames_ignored))
    

#%% Map relative paths to MD results

relative_path_to_results = {}

for im in tqdm(md_results['images']):
    relative_path = im['file']
    assert relative_path not in relative_path_to_results
    relative_path_to_results[relative_path] = im


#%% Add relative paths to our ground truth table

ground_truth_df['relative_path'] = None

images_missing_results = []

# i_row = 0; row = ground_truth_df.iloc[i_row]
for i_row,row in tqdm(ground_truth_df.iterrows(),total=len(ground_truth_df)):
        
    # row['filename'] looks like, e.g. A01/01080001.JPG.  This is not actually a path, it's
    # just the deployment ID and the image name, separated by a slash.

    deployment_name = row['filename'].split('/')[0]
    
    assert deployment_name in deployment_folders, 'Could not find deployment folder {}'.format(deployment_name)
    assert deployment_name in deployment_name_to_file_mappings, 'Could not find deployment folder {}'.format(deployment_name)
    
    file_mappings = deployment_name_to_file_mappings[deployment_name]
            
    # Find the relative path for this image    
    image_name = row['filename'].split('/')[-1]
    assert image_name in file_mappings, 'No mappings for image {} in deployment {}'.format(
        image_name,deployment_name)    
    relative_path = os.path.join(deployment_name,file_mappings[image_name]).replace('\\','/')
    
    # Make sure we have MegaDetector results for this file
    if relative_path not in relative_path_to_results:
        print('Could not find MegaDetector results for {}'.format(relative_path))
        images_missing_results.append(relative_path)
        
    # Make sure this image file exists
    absolute_path = os.path.join(image_base,relative_path)
    assert os.path.isfile(absolute_path), 'Could not find file {}'.format(absolute_path)
    
    ground_truth_df.loc[i_row,'relative_path'] = relative_path

# ...for each row in the ground truth table

images_missing_results = set(images_missing_results)
print('{} images missing MD results'.format(len(images_missing_results)))


#%% Take everything out of Pandas

ground_truth_dicts = ground_truth_df.to_dict('records')


#%% Some additional error-checking of the ground truth

common_names = set()

# An early version of the data required consistency between common_name and is_blank
require_blank_consistency = False

check_blank_consistency = False

for d in tqdm(ground_truth_dicts):
    
    if check_blank_consistency:
        assert d['is_blank'] in [0,1]
        blank_is_consistent = ((d['is_blank'] == 1) and (d['common_name'] == 'Blank')) \
            or ((d['is_blank'] == 0) and (d['common_name'] != 'Blank'))
        if not blank_is_consistent:
            if require_blank_consistency:
                raise ValueError('Blank inconsistency at {}'.format(str(d)))
            else:
                print('Warning: blank inconsistency at {}, updating'.format(str(d)))
                d['is_blank'] = (1 if d['common_name'] == 'Blank' else 0)
    common_names.add(d['common_name'])
                 
    
#%% Combine MD and ground truth results
    
from copy import copy
merged_images = []
n_images_without_results = 0

# d = ground_truth_dicts[0]
for d in tqdm(ground_truth_dicts):
    
    d = copy(d)
    
    if d['relative_path'] not in relative_path_to_results:
        n_images_without_results += 1
        continue
    im = relative_path_to_results[d['relative_path']]
    
    if 'failure' in im and im['failure'] is not None:
        print('Skipping failed image {}'.format(im['filename']))
        continue
    
    # Find the maximum confidence for each category
    maximum_confidences = {}
    for s in category_name_to_id.keys():
        maximum_confidences[s] = 0.0
            
    for detection in im['detections']:
        conf = detection['conf']
        category_id = detection['category']
        category_name = category_id_to_name[category_id]
        if conf > maximum_confidences[category_name]:
            maximum_confidences[category_name] = conf
    # ...for each detection
    
    d['confidences'] = maximum_confidences
    
    merged_images.append(d)
    
# ...for each image    
    
print('Merged ground truth and MD results, {} images didn\'t have ground truth'.format(
    n_images_without_results))


#%% Precision/recall analysis

human_gt_labels = []
animal_gt_labels = []

human_prediction_probs = []
animal_prediction_probs = []
vehicle_prediction_probs = []

include_human_images_in_animal_analysis = False

for d in tqdm(merged_images):
    
    if d['common_name'] == MISSING_COMMON_NAME_TOKEN:
        print('Skipping image with missing ground truth')
        continue
    elif d['common_name'] == 'Blank':
        human_gt_label = 0
        animal_gt_label = 0
    else:
        label = d['common_name']
        if label == 'Human':
            human_gt_label = 1
            animal_gt_label = 0
        else:
            human_gt_label = 0
            animal_gt_label = 1
    
    human_gt_labels.append(human_gt_label)
    human_prediction_probs.append(d['confidences']['person'])

    if include_human_images_in_animal_analysis or human_gt_label == 0:
        animal_gt_labels.append(animal_gt_label)
        animal_prediction_probs.append(d['confidences']['animal'])
    
    vehicle_prediction_probs.append(d['confidences']['vehicle'])
    
# ...for each image


def precision_recall_analysis(gt_labels,prediction_probs,name,confidence_threshold=0.8,target_recall=0.9):

    precisions,recalls,thresholds = precision_recall_curve(gt_labels,prediction_probs)
    
    # Thresholds go up throughout precisions/recalls/thresholds; find the last
    # value where recall is at or above target.  That's our precision @ target recall.
    b_above_target_recall = np.where(recalls >= target_recall)
    if not np.any(b_above_target_recall):
        precision_at_target_recall = 0.0
    else:
        i_target_recall = np.argmax(b_above_target_recall)
        precision_at_target_recall = precisions[i_target_recall]
    print('Precision at {:.1%} recall: {:.1%}'.format(target_recall, precision_at_target_recall))
    
    cm = confusion_matrix(gt_labels, np.array(prediction_probs) > confidence_threshold)
    
    # Flatten the confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    precision_at_confidence_threshold = tp / (tp + fp)
    recall_at_confidence_threshold = tp / (tp + fn)
    f1 = 2.0 * (precision_at_confidence_threshold * recall_at_confidence_threshold) / \
        (precision_at_confidence_threshold + recall_at_confidence_threshold)
    
    print('At a confidence threshold of {:.1%}, precision={:.1%}, recall={:.1%}, f1={:.1%}'.format(
            confidence_threshold, precision_at_confidence_threshold, recall_at_confidence_threshold, f1))

    # Write precision/recall plot to .png file in output directory
    t = 'Precision-recall curve for {}: P@{:0.1%}={:0.1%}'.format(
        name, target_recall, precision_at_target_recall)
    fig = plot_utils.plot_precision_recall_curve(precisions, recalls, t)
    display(fig)
    
    if False:
        min_recall = 0.825
        indices = recalls > min_recall
        recalls_trimmed = recalls[indices]
        precisions_trimmed = precisions[indices]
        t = 'Precision-recall curve for {}: P@{:0.1%}={:0.1%}'.format(
            name, target_recall, precision_at_target_recall)
        fig = plot_utils.plot_precision_recall_curve(precisions_trimmed, recalls_trimmed, t, 
                                                     xlim=(min_recall,1.0),ylim=(0.875,1.0))
        display(fig)
            
    if False:
        # pr_figure_relative_filename = 'prec_recall.png'
        pr_figure_filename = ''
        # pr_figure_filename = os.path.join(output_dir, pr_figure_relative_filename)
        plt.savefig(pr_figure_filename)
        # plt.show(block=False)
        plt.close(fig)
                
# precision_recall_analysis(human_gt_labels,human_prediction_probs,'humans')

print('** Precision/recall analysis for animals **\n')
precision_recall_analysis(animal_gt_labels,animal_prediction_probs,'animals',confidence_threshold=0.8,target_recall=0.9)
precision_recall_analysis(animal_gt_labels,animal_prediction_probs,'animals',confidence_threshold=0.2,target_recall=0.9)


#%% Find and manually review all images of humans

if False:
    
    #%%
    
    human_image_filenames = []
    human_output_folder = os.path.join(analysis_base,'human_gt')
    os.makedirs(human_output_folder,exist_ok=True)
    
    for d in tqdm(merged_images):
        if d['common_name'] == 'Human':
            relative_path = d['relative_path']    
            human_image_filenames.append(relative_path)
            input_path_absolute = os.path.join(image_base,relative_path)
            output_relative_path = relative_path.replace('\\','/').replace('/','_')
            output_path_absolute = os.path.join(human_output_folder,output_relative_path)
            if not os.path.isfile(output_path_absolute):
                shutil.copyfile(input_path_absolute,output_path_absolute)
                
        # ...if this image is annotated as "human"
        
    # ...for each image
        
    print('Found {} human images'.format(len(human_image_filenames)))
    

#%% Find and manually review all MegaDetector animal misses

if False:
    
    #%%
    
    fp_threshold = 0.85
    fn_threshold = 0.6
    
    false_positives = []
    false_negatives = []
    
    fp_output_folder = os.path.join(analysis_base,'animal_fp')
    fn_output_folder = os.path.join(analysis_base,'animal_fn')
    os.makedirs(fp_output_folder,exist_ok=True)
    os.makedirs(fn_output_folder,exist_ok=True)
    
    skip_humans = True
    
    # im = merged_images[0]
    for im in tqdm(merged_images):
        
        if im['common_name'] == MISSING_COMMON_NAME_TOKEN:
            continue
    
        if skip_humans and im['common_name'] == 'Human':
            continue
        
        relative_path = im['relative_path'] 
        input_path_absolute = os.path.join(image_base,relative_path)
        output_relative_path = relative_path.replace('\\','/').replace('/','_')
        
        # GT says this is not an animal
        if im['common_name'] == 'Blank' or im['common_name'] == 'Human':
            if im['confidences']['animal'] > fp_threshold:
                false_positives.append(im)
                output_path_absolute = os.path.join(fp_output_folder,output_relative_path)
                if not os.path.isfile(output_path_absolute):
                    shutil.copyfile(input_path_absolute,output_path_absolute)                    
        
        # GT says this is an animal
        else:
            assert im['common_name'] != 'Human'
            if im['confidences']['animal'] < fn_threshold:
                false_negatives.append(im)
                output_path_absolute = os.path.join(fn_output_folder,output_relative_path)
                if not os.path.isfile(output_path_absolute):
                    shutil.copyfile(input_path_absolute,output_path_absolute)                    
    
    print('Found {} FPs and {} FNs'.format(len(false_positives),len(false_negatives)))
    


#%% Convert .json to .csv

if False:
    
    #%%
    
    from api.batch_processing.postprocessing import convert_output_format
    
    output_path = results_file.replace('.json','.csv')
    
    convert_output_format.convert_json_to_csv(input_path=results_file,
                                              output_path=output_path,
                                              min_confidence=None,
                                              omit_bounding_boxes=True,
                                              output_encoding=None)
    
    output_path_filtered = results_file_filtered.replace('.json','.csv')
    
    convert_output_format.convert_json_to_csv(input_path=results_file_filtered,
                                              output_path=output_path_filtered,
                                              min_confidence=None,
                                              omit_bounding_boxes=True,
                                              output_encoding=None)

#
# kga-pr-analysis.py
#
# Precision/recall analysis for KGA data
#

#%% Imports and constants

import os
import json

from tqdm import tqdm

if True:
    results_file_all = os.path.expanduser('~/postprocessing/' + \
        'snapshot-safari/snapshot-safari-2022-04-07/combined_api_outputs/snapshot-safari-2022-04-07_detections.json')

if False:
    results_file_all = os.path.expanduser('~/postprocessing/' + \
        'snapshot-safari/snapshot-safari-2022-04-07/combined_api_outputs/snapshot-safari-2022-04-07_detections.filtered_rde_0.60_0.85_10_0.20.json')
        
if False:
    results_file_all = os.path.expanduser('~/postprocessing/' + \
        'snapshot-safari/snapshot-safari-mdv5-camcocoinat-2022-05-02/combined_api_outputs/snapshot-safari-mdv5-camcocoinat-2022-05-02_detections.json'                                      )

if False:
    results_file_all = os.path.expanduser('~/postprocessing/' + \
                                          'snapshot-safari/snapshot-safari-mdv5-camonly-2022-05-02/combined_api_outputs/snapshot-safari-mdv5-camonly-2022-05-02_detections.json')

if True:    
    results_file_replacements = {'KGA/KGA_public/':''}
    required_prefix = 'KGA'
    ground_truth_file = os.path.expanduser('~/postprocessing/snapshot-safari/' + \
      'ground_truth/SnapshotKgalagadi_S1_v1.0.json')

if False:
    results_file_replacements = {'KRU/KRU_public/':''}
    required_prefix = 'KRU'
    ground_truth_file = os.path.expanduser('~/postprocessing/snapshot-safari/' + \
      'ground_truth/SnapshotKruger_S1_v1.0.json')

if False:
    results_file_replacements = {'KAR/KAR_public/':''}
    required_prefix = 'KAR'
    ground_truth_file = os.path.expanduser('~/postprocessing/snapshot-safari/' + \
      'ground_truth/SnapshotKaroo_S1_v1.0.json')

if False:
    results_file_replacements = {'ENO/ENO_public/':''}
    required_prefix = 'ENO'
    ground_truth_file = os.path.expanduser('~/postprocessing/snapshot-safari/' + \
      'ground_truth/SnapshotEnonkishu_S1_v1.0.json')

if False:
    results_file_replacements = {'MTZ/MTZ_public/':''}
    required_prefix = 'MTZ'
    ground_truth_file = os.path.expanduser('~/postprocessing/snapshot-safari/' + \
      'ground_truth/SnapshotMountainZebra_S1_v1.0.json')

empty_categories = ['empty']
human_categories = ['human']
other_categories = []

assert os.path.isfile(ground_truth_file)
assert os.path.isfile(results_file_all)


#%% Load and filter MD results

with open(results_file_all,'r') as f:
    results_all = json.load(f)

results_images_filtered = []

if required_prefix is None:
    results_images_filtered = results_all['images']
else:
    for im in tqdm(results_all['images']):
        if im['file'].startswith(required_prefix):
            results_images_filtered.append(im)
            
print('\nMatched {} of {} MD results'.format(len(results_images_filtered),
                                       len(results_all['images'])))    

md_results = results_all
md_results['images'] = results_images_filtered

assert md_results['detection_categories'] == {'1': 'animal', '2': 'person', '3': 'vehicle'}
detection_name_to_category_id = {}
for k in md_results['detection_categories']:
    v = md_results['detection_categories'][k]
    detection_name_to_category_id[v] = k
    
filename_to_md_result = {}
for im in results_images_filtered:
    for k in results_file_replacements:
        v = results_file_replacements[k]
        im['file'] = im['file'].replace(k,v)
    filename_to_md_result[im['file']] = im


#%% Load and filter ground truth

with open(ground_truth_file,'r') as f:
    ground_truth_all = json.load(f)

ground_truth_category_id_to_name = {c['id']:c['name'] for c in ground_truth_all['categories']}

ground_truth_category_id_to_detection_category = {}
    
for cat_id in ground_truth_category_id_to_name:
    
    category_name = ground_truth_category_id_to_name[cat_id]
    
    if category_name in empty_categories:
        print('Mapping {} to empty'.format(category_name))
        ground_truth_category_id_to_detection_category[cat_id] = 'empty'
        
    elif category_name in human_categories:
        print('Mapping {} to human'.format(category_name))
        ground_truth_category_id_to_detection_category[cat_id] = 'person'
        
    elif category_name in other_categories:
        print('Mapping {} to other'.format(category_name))
        ground_truth_category_id_to_detection_category[cat_id] = 'other'
        
    else:
        print('Mapping {} to animal'.format(category_name))
        ground_truth_category_id_to_detection_category[cat_id] = 'animal'

gt_images_filtered = []

if required_prefix is None:
    images_filtered = ground_truth_all['images']
else:
    for im in tqdm(ground_truth_all['images']):
        if im['file_name'].startswith(required_prefix):
            gt_images_filtered.append(im)

print('\nMatched {} of {} ground truth images'.format(len(gt_images_filtered),
                                       len(ground_truth_all['images'])))    


#%% Map images to image-level results

from collections import defaultdict

image_id_to_md_results = defaultdict(list)

failed_matches = []

for gt_im in gt_images_filtered:
    fn = gt_im['file_name']
    if fn not in filename_to_md_result:
        failed_matches.append(fn)
        image_id_to_md_results[gt_im['id']] = None
    else:
        image_id_to_md_results[gt_im['id']].append(filename_to_md_result[fn])

print('{} failed matches from GT to detection results'.format(len(failed_matches)))


#%% Map sequence IDs to images and annotations to images

sequence_id_to_images = defaultdict(list)
image_id_to_image = {}

for gt_im in gt_images_filtered:
    seq_id = gt_im['seq_id']    
    sequence_id_to_images[seq_id].append(gt_im)
    image_id = gt_im['id']
    assert image_id not in image_id_to_image
    image_id_to_image[image_id] = gt_im

sequence_id_to_annotations = defaultdict(list)

for ann in ground_truth_all['annotations']:
    
    assert ann['sequence_level_annotation']
    
    image_id = ann['image_id']
    seq_id = ann['seq_id']
    
    im = image_id_to_image[image_id]
    assert im['seq_id'] == seq_id
    
    sequence_id_to_annotations[seq_id].append(ann)
        
print('TODO: verify consistency of annotation within a sequence')

# Verify consistency of annotation within a sequence
for seq_id in sequence_id_to_annotations:
    
    annotations_this_sequence = sequence_id_to_annotations[seq_id]
    # TODO


#%% Find max confidence values for each category for each sequence

sequence_id_to_confidence_values = {}

n_failures = 0
n_missing = 0

# seq_id = list(sequence_id_to_images.keys())[1000]
for seq_id in sequence_id_to_images:
    
    assert seq_id not in sequence_id_to_confidence_values
    
    max_confidence_values = {'person':-1.0,'animal':-1.0,'vehicle':-1.0}
    found_image = False
    images_this_sequence = sequence_id_to_images[seq_id]
    
    # im = images_this_sequence[0]
    for im in images_this_sequence:

        fn = im['file_name']
        
        if fn not in filename_to_md_result:
            continue
       
        md_result = filename_to_md_result[fn]
        if 'detections' not in md_result or md_result['detections'] is None:
            assert 'failure' in md_result
            n_failures += 1
            continue
        
        found_image = True
        
        # det = md_result['detections'][]
        for det in md_result['detections']:
            category_id = det['category']
            category_name = md_results['detection_categories'][category_id]
            if det['conf'] > max_confidence_values[category_name]:
                max_confidence_values[category_name] = det['conf']
        
        # ...for each detection
        
    # ...for each image in this sequence
    
    if not found_image:
        sequence_id_to_confidence_values[seq_id] = None
        n_missing += 1
    else:
        sequence_id_to_confidence_values[seq_id] = max_confidence_values
    
# ...for each sequence

print('Found {} failures, {} missing'.format(n_failures, n_missing))


#%% Prepare for precision/recall analysis

human_gt_labels = []
animal_gt_labels = []

human_prediction_probs = []
animal_prediction_probs = []
vehicle_prediction_probs = []

pr_sequence_ids = []

# seq_id = list(sequence_id_to_images.keys())[1000]
for seq_id in sequence_id_to_images:
    
    if seq_id not in sequence_id_to_confidence_values or \
        sequence_id_to_confidence_values[seq_id] is None:
        continue
    
    confidence_values = sequence_id_to_confidence_values[seq_id]
    human_prediction_probs.append(confidence_values['person'])
    animal_prediction_probs.append(confidence_values['animal'])
    vehicle_prediction_probs.append(confidence_values['vehicle'])
    
    annotations_this_sequence = sequence_id_to_annotations[seq_id]
    assert len(annotations_this_sequence) > 0
    
    category_ids_this_sequence = set([ann['category_id'] for ann in annotations_this_sequence])
    
    animal = False
    human = False
    
    pr_sequence_ids.append(seq_id)
    
    # cat_id = list(category_ids_this_sequence)[0]
    for cat_id in category_ids_this_sequence:
        
        detection_category = ground_truth_category_id_to_detection_category[cat_id]
        
        if detection_category == 'animal':
            animal = True
        elif detection_category == 'person':
            human = True
            
    # ...for each category in this sequence
    
    human_gt_labels.append(human)
    animal_gt_labels.append(animal)
    
# ...for each sequence


#%% Precision/recall analysis

import numpy as np
import visualization.plot_utils as plot_utils
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, confusion_matrix
from IPython.core.display import display

def precision_recall_analysis(gt_labels,prediction_probs,name,confidence_threshold=0.8,target_recall=0.9):

    precisions,recalls,thresholds = precision_recall_curve(gt_labels,prediction_probs)
    
    # Confirm that thresholds are increasing, recall is decreasing
    assert np.all(thresholds[:-1] <= thresholds[1:])
    assert np.all(recalls[:-1] >= recalls[1:])
    
    # This is not necessarily true
    # assert np.all(precisions[:-1] <= precisions[1:])
    
    # Thresholds go up throughout precisions/recalls/thresholds; find the max
    # value where recall is at or above target.  That's our precision @ target recall.
    # This is very slightly optimistic in its handling of non-monotonic recall curves,
    # but is an easy scheme to deal with.
    b_above_target_recall = np.where(recalls >= target_recall)
    if not np.any(b_above_target_recall):
        precision_at_target_recall = 0.0
    else:
        precisions_above_target_recall = []
        for i_recall,recall in enumerate(recalls):
            if recall >= target_recall:
                precisions_above_target_recall.append(precisions[i_recall])
        assert len(precisions_above_target_recall) > 0
        precision_at_target_recall = max(precisions_above_target_recall)
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
        min_recall = 0.945
        indices = recalls > min_recall
        recalls_trimmed = recalls[indices]
        precisions_trimmed = precisions[indices]
        t = 'Precision-recall curve for {}: P@{:0.1%}={:0.1%}'.format(
            name, target_recall, precision_at_target_recall)
        fig = plot_utils.plot_precision_recall_curve(precisions_trimmed, recalls_trimmed, t, 
                                                     xlim=(min_recall,1.0),ylim=(0.6,1.0))
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
precision_recall_analysis(animal_gt_labels,animal_prediction_probs,'animals',
                          confidence_threshold=0.1,target_recall=0.9)
precision_recall_analysis(animal_gt_labels,animal_prediction_probs,'animals',
                          confidence_threshold=0.1,target_recall=0.95)


#%% Scrap

if False:
    
    #%% Find and manually review all sequence-level MegaDetector animal misses
    
    import shutil
    sequence_preview_dir = os.path.expanduser('~/tmp/sequence_preview')
    os.makedirs(sequence_preview_dir,exist_ok=True)    
    # input_base = '/media/user/lila-01/lila/snapshot-safari/KGA/KGA_public'
    input_base = '/media/user/lila-01/lila/snapshot-safari/MTZ/MTZ_public'
    
    fn_threshold = 0.15
    
    false_negative_sequences = []
    
    # i_sequence = 0; seq_id = pr_sequence_ids[i_sequence]
    for i_sequence,seq_id in enumerate(pr_sequence_ids):
        
        animal_gt_label = animal_gt_labels[i_sequence]
        p_animal = animal_prediction_probs[i_sequence]
        
        if (p_animal < fn_threshold) and (animal_gt_label == True):
            false_negative_sequences.append(seq_id)
            
    print('Found {} sequence-level false negatives'.format(len(false_negative_sequences)))
    
    # i_seq = 0; seq_id = false_negative_sequences[i_seq]
    for i_seq,seq_id in enumerate(false_negative_sequences):
        
        seq_images = sequence_id_to_images[seq_id]    
        image_files = [im['file_name'] for im in seq_images]
        
        # sequence_folder = os.path.join(sequence_preview_dir,'seq_{}'.format(str(i_seq).zfill(3)))
        sequence_folder = sequence_preview_dir
        os.makedirs(sequence_folder,exist_ok=True)
        
        # fn = image_files[0]
        for fn in image_files:
            
            input_path = os.path.join(input_base,fn)
            assert os.path.isfile(input_path)
            
            output_path = os.path.join(sequence_folder,fn.replace('/','_'))
            # print('Copying {} to {}'.format(input_path,output_path))
            shutil.copyfile(input_path, output_path)
        
        # ...for each file in this sequence.
        
    # ...for each sequence        
    

    #%% Image-level postprocessing
    
    import sys,subprocess
    
    def open_file(filename):
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])

    from api.batch_processing.postprocessing.postprocess_batch_results import (
        PostProcessingOptions, process_batch_results)

    input_base = '/media/user/lila-01/lila/snapshot-safari/MTZ/MTZ_public'
    
    temporary_results_file = os.path.expanduser('~/tmp/filtered_results.json')
    with open(temporary_results_file,'w') as f:
        json.dump(md_results,f,indent=2)
        
    output_base = os.path.expanduser('~/tmp/pr-image-level-preview')
    os.makedirs(output_base,exist_ok=True)
        
    options = PostProcessingOptions()
    options.image_base_dir = input_base
    options.parallelize_rendering = True
    options.include_almost_detections = True
    options.num_images_to_sample = 10000
    options.confidence_threshold = 0.75
    options.classification_confidence_threshold = 0.75
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = ground_truth_file
    options.separate_detections_by_category = True        
    options.api_output_file = temporary_results_file
    options.output_dir = output_base
    ppresults = process_batch_results(options)
    open_file(ppresults.output_html_file)

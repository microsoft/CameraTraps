########
#
# postprocess_batch_results.py
#
# Given a .csv file representing the output from the batch API, do one or more of 
# the following:
#
# * Evaluate detector precision/recall, optionally rendering results (requires ground truth)
#
# * Sample true/false positives/negatives and render to html (requires ground truth)
#
# * Sample detections/non-detections and render to html (when ground truth isn't available)
#
# Upcoming improvements:
#
# * Elimination of "suspicious detections", i.e. detections repeated numerous times with
#   unrealistically limited movement
# 
# * Support for accessing blob storage directly (currently images are accessed by
#   file paths, so images in Azure blobs should be accessed by mounting the 
#   containers)
#
########


#%% Constants and imports

import inspect
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from enum import IntEnum
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score
from detection.detection_eval.load_api_results import load_api_results

# Assumes the cameratraps repo root is on the path
from data_management.cct_json_utils import CameraTrapJsonUtils
from data_management.cct_json_utils import IndexedJsonDb
import visualization.visualization_utils as vis_utils

# Assumes ai4eutils is on the python path
#
# https://github.com/Microsoft/ai4eutils
from write_html_image_list import write_html_image_list


#%% Options

DEFAULT_NEGATIVE_CLASSES = ['empty']
DEFAULT_UNKNOWN_CLASSES = ['unknown','unlabeled']

class PostProcessingOptions:

    ### Required inputs

    detector_output_file = ''
    image_base_dir = ''
    ground_truth_json_file = ''
    output_dir = ''

    ### Options    
    
    negative_classes = DEFAULT_NEGATIVE_CLASSES
    unlabeled_classes = DEFAULT_UNKNOWN_CLASSES
    
    confidence_threshold = 0.85

    # Used for summary statistics only
    target_recall = 0.9    

    # Number of images to sample, -1 for "all images"
    num_images_to_sample = 500 # -1
    
    viz_target_width = 800
    
    sort_html_by_filename = True
    
    # Optionally replace one or more strings in filenames with other strings;
    # this is useful for taking a set of results generated for one folder structure
    # and applying them to a slightly different folder structure.
    detector_output_filename_replacements = {}
    ground_truth_filename_replacements = {}
    
    
#%% Helper classes and functions

# Flags used to mark images as positive or negative for P/R analysis (according
# to ground truth and/or detector output)
class DetectionStatus(IntEnum):
    
    # This image is a negative
    DS_NEGATIVE = 0
    
    # This image is a positive
    DS_POSITIVE = 1
    
    # Anything greater than this isn't clearly positive or negative
    DS_MAX_DEFINITIVE_VALUE = DS_POSITIVE
    
    # This image has annotations suggesting both negative and positive
    DS_AMBIGUOUS = 2
    
    # This image is not annotated or is annotated with 'unknown', 'unlabeled', ETC.
    DS_UNKNOWN = 3
    
    # This image has not yet been assigned a state
    DS_UNASSIGNED = 4


def mark_detection_status(indexed_db, negative_classes=DEFAULT_NEGATIVE_CLASSES,
                          unknown_classes=DEFAULT_UNKNOWN_CLASSES):
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
        
        image_status = DetectionStatus.DS_UNASSIGNED
        
        if len(image_categories) == 0:
            
            image_status = DetectionStatus.DS_UNKNOWN
            
        else:            
            
            for cat_id in image_categories:
                
                cat_name = indexed_db.cat_id_to_name[cat_id]            
                
                if cat_name in negative_classes:                    
                    
                    if image_status == DetectionStatus.DS_UNASSIGNED:
                        image_status = DetectionStatus.DS_NEGATIVE
                    elif image_status != DetectionStatus.DS_NEGATIVE:
                        image_status = DetectionStatus.DS_AMBIGUOUS
                
                elif cat_name in unknown_classes:
                    
                    if image_status == DetectionStatus.DS_UNASSIGNED:
                        image_status = DetectionStatus.DS_UNKNOWN
                    elif image_status != DetectionStatus.DS_UNKNOWN:
                        image_status = DetectionStatus.DS_AMBIGUOUS
                
                else:                    
                    
                    if image_status == DetectionStatus.DS_UNASSIGNED:
                        image_status = DetectionStatus.DS_POSITIVE
                    elif image_status != DetectionStatus.DS_POSITIVE:
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


def render_bounding_boxes(image_base_dir,image_relative_path,
                          display_name,boxes_and_scores,res,options=None):
    
        if options is None:
            options = PostProcessingOptions()
            
        # Leaving code in place for reading from blob storage, may support this
        # in the future.
        """
        stream = io.BytesIO()
        _ = blob_service.get_blob_to_stream(container_name, image_id, stream)
        image = Image.open(stream).resize(viz_size)  # resize is to display them in this notebook or in the HTML more quickly
        """
        
        image_full_path = os.path.join(image_base_dir,image_relative_path)
        if not os.path.isfile(image_full_path):
            print('Warning: could not find image file {}'.format(image_full_path))
            return ''
                        
        image = vis_utils.open_image(image_full_path)
        vis_utils.render_detection_bounding_boxes(boxes_and_scores, image, 
                                                  confidence_threshold=options.confidence_threshold,
                                                  thickness=6)
        
        image = vis_utils.resize_image(image, options.viz_target_width)
        
        # Render images to a flat folder... we can use os.sep here because we've
        # already normalized paths
        sample_name = res + '_' + image_relative_path.replace(os.sep, '~')
        
        image.save(os.path.join(options.output_dir, res, sample_name))
        
        # Use slashes regardless of os
        file_name = '{}/{}'.format(res, sample_name)
        
        return {
            'filename': file_name,
            'title': display_name,
            'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;text-align:left;margin-top:20;margin-bottom:5'
        }
    
    
def prepare_html_subpages(images_html,output_dir,options=None):
    
    if options is None:
            options = PostProcessingOptions()
            
    # Count items in each category
    image_counts = {}
    for res, array in images_html.items():
        image_counts[res] = len(array)

    # Optionally sort by filename before writing to html    
    if options.sort_html_by_filename:                
        images_html_sorted = {}
        for res, array in images_html.items():
            sorted_array = sorted(array, key=lambda x: x['filename'])
            images_html_sorted[res] = sorted_array        
        images_html = images_html_sorted
        
    # Write the individual HTML files
    for res, array in images_html.items():
        write_html_image_list(
            filename=os.path.join(output_dir, '{}.html'.format(res)), 
            images=array,
            options={
                'headerHtml': '<h1>{}</h1>'.format(res.upper())
            })
    
    return image_counts
    

#%% Main function
    
def process_batch_results(options):
    
    ##%% Expand some options for convenience
    
    output_dir = options.output_dir
    confidence_threshold = options.confidence_threshold
    
    
    ##%% Prepare output dir
        
    os.makedirs(output_dir,exist_ok=True)
    
    
    ##%% Load ground truth if available
    
    ground_truth_indexed_db = None
    
    if len(options.ground_truth_json_file) > 0:
            
        ground_truth_indexed_db = IndexedJsonDb(options.ground_truth_json_file,True)
        
        # Mark images in the ground truth as positive or negative
        (nNegative,nPositive,nUnknown,nAmbiguous) = mark_detection_status(ground_truth_indexed_db,
            options.negative_classes)
        print('Finished loading and indexing ground truth: {} negative, {} positive, {} unknown, {} ambiguous'.format(
                nNegative,nPositive,nUnknown,nAmbiguous))
    
    
    ##%% Load detection results
    
    detection_results = load_api_results(options.detector_output_file,normalize_paths=True,
                                         filename_replacements=options.detector_output_filename_replacements)
    
    # Add a column (pred_detection_label) to indicate predicted detection status
    import numpy as np
    detection_results['pred_detection_label'] = \
        np.where(detection_results['max_confidence'] >= options.confidence_threshold,
                 DetectionStatus.DS_POSITIVE, DetectionStatus.DS_NEGATIVE)
    
    nPositives = sum(detection_results['pred_detection_label'] == DetectionStatus.DS_POSITIVE)
    print('Finished loading and preprocessing {} rows from detector output, predicted {} positives'.format(
            len(detection_results),nPositives))
    
    
    ##%% Find suspicious detections
    
        
    ##%% If we have ground truth, remove images we can't match to ground truth
    
    if ground_truth_indexed_db is not None:
    
        b_match = [False] * len(detection_results)
        
        detector_files = detection_results['image_path'].to_list()
            
        for iFn,fn in enumerate(detector_files):
            
            # assert fn in ground_truth_indexed_db.filename_to_id, 'Could not find ground truth for row {} ({})'.format(iFn,fn)
            if fn in fn in ground_truth_indexed_db.filename_to_id:
                b_match[iFn] = True
                        
        print('Confirmed filename matches to ground truth for {} of {} files'.format(sum(b_match),len(detector_files)))
        
        detection_results = detection_results[b_match]
        detector_files = detection_results['image_path'].to_list()
        
        print('Trimmed detection results to {} files'.format(len(detector_files)))
        

    ##%% Sample images for visualization
    
    images_to_visualize = detection_results
        
    if options.num_images_to_sample > 0 and options.num_images_to_sample < len(detection_results):
        
        images_to_visualize = images_to_visualize.sample(options.num_images_to_sample)
    
        
    ##%% Fork here depending on whether or not ground truth is available
        
    # If we have ground truth, we'll compute precision/recall and sample tp/fp/tn/fn.
    #
    # Otherwise we'll just visualize detections/non-detections.
        
    if ground_truth_indexed_db is not None:
    
        ##%% Compute precision/recall
        
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
        print('Average precision: {:.2f}'.format(average_precision))
    
        # Thresholds go up throughout precisions/recalls/thresholds; find the last
        # value where recall is at or above target.  That's our precision @ target recall.
        target_recall = 0.9
        b_above_target_recall = np.where(recalls >= target_recall)
        if not np.any(b_above_target_recall):
            precision_at_target_recall = 0.0
        else:
            i_target_recall = np.argmax(b_above_target_recall)
            precision_at_target_recall = precisions[i_target_recall]
        print('Precision at {:.2f} recall: {:.2f}'.format(target_recall,precision_at_target_recall))    
        
        cm = confusion_matrix(gt_detections_pr, np.array(p_detection_pr) > confidence_threshold)
    
        # Flatten the confusion matrix
        tn, fp, fn, tp = cm.ravel()
    
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2.0 * (precision * recall) / (precision + recall)
        
        print('At a confidence threshold of {:.2f}, precision={:.2f}, recall={:.2f}, f1={:.2f}'.format(
                confidence_threshold,precision, recall, f1))
            
        
        ##%% Render output
        
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
        pr_figure_relative_filename = 'prec_recall.png'
        pr_figure_filename = os.path.join(output_dir, pr_figure_relative_filename)
        plt.savefig(pr_figure_filename)
        # plt.show()
            
            
        ##%% Sample true/false positives/negatives and render to html
        
        os.makedirs(os.path.join(output_dir, 'tp'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'fp'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tn'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'fn'), exist_ok=True)
    
        # Accumulate html image structs (in the format expected by write_html_image_lists) 
        # for each category
        images_html = {
            'tp': [],
            'fp': [],
            'tn': [],
            'fn': []
        }
            
        count = 0
            
        # i_row = 0; row = images_to_visualize.iloc[0]
        for i_row, row in tqdm(images_to_visualize.iterrows(), total=len(images_to_visualize)):
            
            image_relative_path = row['image_path']
            
            # This should already have been normalized to either '/' or '\'
            
            image_id = ground_truth_indexed_db.filename_to_id.get(image_relative_path,None)
            if image_id is None:
                print('Warning: couldn''t find ground truth for image {}'.format(image_relative_path))
                continue
    
            image_info = ground_truth_indexed_db.image_id_to_image[image_id]
            annotations = ground_truth_indexed_db.image_id_to_annotations[image_id]
            
            gt_status = image_info['_detection_status']
            
            if gt_status > DetectionStatus.DS_MAX_DEFINITIVE_VALUE:
                print('Skipping image {}, does not have a definitive ground truth status'.format(i_row,gt_status))
                continue
            
            gt_presence = bool(gt_status)
            
            gt_class_name = CameraTrapJsonUtils.annotationsToString(
                    annotations,ground_truth_indexed_db.cat_id_to_name)
            
            max_conf = row['max_confidence']
            boxes_and_scores = row['detections']
        
            detected = True if max_conf > confidence_threshold else False
            
            if gt_presence and detected:
                res = 'tp'
            elif not gt_presence and detected:
                res = 'fp'
            elif gt_presence and not detected:
                res = 'fn'
            else:
                res = 'tn'
            
            display_name = '<b>Result type</b>: {}, <b>Presence</b>: {}, <b>Class</b>: {}, <b>Image</b>: {}'.format(
                res.upper(),
                str(gt_presence),
                gt_class_name,
                image_relative_path)
    
            rendered_image_html_info = render_bounding_boxes(options.image_base_dir,image_relative_path,
                                                            display_name,boxes_and_scores,res,options)        
            if len(rendered_image_html_info) > 0:
                images_html[res].append(rendered_image_html_info)
                
            count += 1
            
        # ...for each image in our sample
        
        print('{} images rendered'.format(count))
            
        # Prepare the individual html image files
        image_counts = prepare_html_subpages(images_html,output_dir)
                
        # Write index.HTML    
        index_page = """<html><body>
        <p><strong>A sample of {} images, annotated with detections above {:.1f}% confidence.</strong></p>
        
        <a href="tp.html">True positives (tp)</a> ({})<br/>
        <a href="tn.html">True negatives (tn)</a> ({})<br/>
        <a href="fp.html">False positives (fp)</a> ({})<br/>
        <a href="fn.html">False negatives (fn)</a> ({})<br/>
        <br/><p><strong>Precision/recall summary for all {} images</strong></p><img src="{}"><br/>
        </body></html>""".format(
            count, confidence_threshold * 100,
            image_counts['tp'], image_counts['tn'], image_counts['fp'], image_counts['fn'],
            len(detection_results),pr_figure_relative_filename
        )
        output_html_file = os.path.join(output_dir, 'index.html')
        with open(output_html_file, 'w') as f:
            f.write(index_page)
        
        print('Finished writing html to {}'.format(output_html_file))
    
    
    ##%% Otherwise, if we don't have ground truth...
        
    else:
        
        ##%% Sample detections/non-detections
        
        os.makedirs(os.path.join(output_dir, 'detections'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'non_detections'), exist_ok=True)
        
        # Accumulate html image structs (in the format expected by write_html_image_lists) 
        # for each category
        images_html = {
            'detections': [],
            'non_detections': [],
        }
            
        count = 0
            
        # i_row = 0; row = images_to_visualize.iloc[0]
        for i_row, row in tqdm(images_to_visualize.iterrows(), total=len(images_to_visualize)):
            
            image_relative_path = row['image_path']
            
            # This should already have been normalized to either '/' or '\'
            max_conf = row['max_confidence']
            boxes_and_scores = row['detections']
            detected = True if max_conf > confidence_threshold else False
            
            if detected:
                res = 'detections'
            else:
                res = 'non_detections'
            
            display_name = '<b>Result type</b>: {}, <b>Image</b>: {}'.format(
                res.upper(),
                image_relative_path)
    
            rendered_image_html_info = render_bounding_boxes(options.image_base_dir,image_relative_path,
                                                            display_name,boxes_and_scores,res,options)        
            if len(rendered_image_html_info) > 0:
                images_html[res].append(rendered_image_html_info)
            
            count += 1
            
        # ...for each image in our sample
        
        print('{} images rendered'.format(count))
            
        # Prepare the individual html image files
        image_counts = prepare_html_subpages(images_html,output_dir)
            
        # Write index.HTML    
        index_page = """<html><body>
        <p><strong>A sample of {} images, annotated with detections above {:.1f}% confidence.</strong></p>
        
        <a href="detections.html">Detections</a> ({})<br/>
        <a href="non_detections.html">Non-detections</a> ({})<br/>
        </body></html>""".format(
            count, confidence_threshold * 100,
            image_counts['detections'], image_counts['non_detections']
        )
        output_html_file = os.path.join(output_dir, 'index.html')
        with open(output_html_file, 'w') as f:
            f.write(index_page)
        
        print('Finished writing html to {}'.format(output_html_file))
    
    # ...if we do/don't have ground truth

# ...process_batch_results

    
#%% Command-line driver
    
# Copy all fields from a Namespace (i.e., the output from parse_args) to an object.  
#
# Skips fields starting with _.  Does not check existence in the target object.
def argsToObject(args, obj):
    
    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v);


def main():
    
    # python postprocess_batch_results.py "d:\temp\8471_detections.csv" "d:\temp\postprocessing_tmp" --image_base_dir "d:\wildlife_data\mcgill_test" --ground_truth_json_file "d:\wildlife_data\mcgill_test\mcgill_test.json" --num_images_to_sample 100
    parser = argparse.ArgumentParser()
    parser.add_argument('detector_output_file', action='store', type=str, help='.csv file produced by the batch inference API (required)')
    parser.add_argument('output_dir', action='store', type=str, help='Base directory for output (required)')
    parser.add_argument('--image_base_dir', action='store', type=str, help='Base directory for images (optional, can compute statistics without images)')
    parser.add_argument('--ground_truth_json_file', action='store', type=str, help='Ground truth labels (optional, can render detections without ground truth)')
    
    parser.add_argument('--confidence_threshold', action='store', type=float, default=0.85, help='Confidence threshold for statistics and visualization')
    parser.add_argument('--target_recall', action='store', type=float, default=0.9, help='Target recall (for statistics only)')
    parser.add_argument('--num_images_to_sample', action='store', type=int, default=500, help='Number of images to visualize (defaults to 500) (-1 to include all images)')
    parser.add_argument('--viz_target_width', action='store', type=int, default=800, help='Output image width')
    parser.add_argument('--random_output_sort', action='store_true', help='Sort output randomly (defaults to sorting by filename)')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()    
    args.sort_html_by_filename = not args.random_output_sort
    
    options = PostProcessingOptions()
    argsToObject(args,options)
    
    process_batch_results(options)


if __name__ == '__main__':
    
    main()


#%% Interactive driver(s)

if False:
    
    #%%
    
    baseDir = r'e:\wildlife_data\rspb_gola_data'
    options = PostProcessingOptions()
    options.detector_output_file = os.path.join(baseDir,'RSPB_detections_old_format_mdv3.19.05.09.1612.csv')
    options.image_base_dir = os.path.join(baseDir,'gola_camtrapr_data')
    options.ground_truth_json_file = os.path.join(baseDir,'rspb_gola_v2.json')
    options.output_dir = os.path.join(baseDir,'postprocessing_output')
    options.detector_output_filename_replacements = {'gola\\gola_camtrapr_data\\':''}
    process_batch_results(options)        
         
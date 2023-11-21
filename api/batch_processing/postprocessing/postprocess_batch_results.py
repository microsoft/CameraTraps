"""

postprocess_batch_results.py

Given a .json or .csv file representing the output from the batch detection API,
do one or more of the following:

* Evaluate detector precision/recall, optionally rendering results (requires
    ground truth)
* Sample true/false positives/negatives and render to HTML (requires ground
    truth)
* Sample detections/non-detections and render to HTML (when ground truth isn't
    available)

Ground truth, if available, must be in the COCO Camera Traps format.

"""


#%% Constants and imports

import argparse
import collections
import copy
import errno
import io
import itertools
import os
import sys
import time
import uuid
import urllib
import warnings

from typing import Any, Dict, Iterable, Optional, Tuple
from enum import IntEnum
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import humanfriendly
import pandas as pd

from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score
from tqdm import tqdm

# Assumes ai4eutils is on the python path (https://github.com/Microsoft/ai4eutils)
from write_html_image_list import write_html_image_list
import path_utils

# Assumes the cameratraps repo root is on the path
import visualization.visualization_utils as vis_utils
import visualization.plot_utils as plot_utils
from data_management.cct_json_utils import (CameraTrapJsonUtils, IndexedJsonDb)
from api.batch_processing.postprocessing.load_api_results import load_api_results
from ct_utils import args_to_object

from detection.run_detector import get_typical_confidence_threshold_from_results

warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)


#%% Options

DEFAULT_NEGATIVE_CLASSES = ['empty']
DEFAULT_UNKNOWN_CLASSES = ['unknown', 'unlabeled', 'ambiguous']


def has_overlap(set1: Iterable, set2: Iterable) -> bool:
    """Check whether two sets overlap."""
    return not set(set1).isdisjoint(set(set2))


# Make sure there is no overlap between the two sets, because this will cause
# issues in the code
assert not has_overlap(DEFAULT_NEGATIVE_CLASSES, DEFAULT_UNKNOWN_CLASSES), (
        'Default negative and unknown classes cannot overlap.')


class PostProcessingOptions:

    ### Required inputs

    api_output_file = ''
    output_dir = ''

    ### Options

    # Can be a folder or a SAS URL
    image_base_dir = '.'

    ground_truth_json_file = ''

    ## These apply only when we're doing ground-truth comparisons
    
    # Classes we'll treat as negative
    #
    # Include the token "#NO_LABELS#" to indicate that an image with no annotations
    # should be considered empty.
    negative_classes = DEFAULT_NEGATIVE_CLASSES
    
    # Classes we'll treat as neither positive nor negative
    unlabeled_classes = DEFAULT_UNKNOWN_CLASSES

    # A list of output sets that we should count, but not render images for.
    #
    # Typically used to preview sets with lots of empties, where you don't want to
    # subset but also don't want to render 100,000 empty images.
    #
    # detections, non_detections
    # detections_animal, detections_person, detections_vehicle
    rendering_bypass_sets = []

    # By default, choose a confidence threshold based on the detector version
    confidence_threshold = None
    classification_confidence_threshold = 0.5

    # Used for summary statistics only
    target_recall = 0.9

    # Number of images to sample, -1 for "all images"
    num_images_to_sample = 500

    # Random seed for sampling, or None
    sample_seed: Optional[int] = 0 # None

    viz_target_width = 800

    line_thickness = 4
    box_expansion = 0

    job_name_string = None
    model_version_string = None
    
    sort_html_by_filename = True

    link_images_to_originals = True
    
    # Optionally separate detections into categories (animal/vehicle/human)
    # 
    # Currently only supported when ground truth is unavailable
    separate_detections_by_category = True

    # Optionally replace one or more strings in filenames with other strings;
    # useful for taking a set of results generated for one folder structure
    # and applying them to a slightly different folder structure.
    api_output_filename_replacements = {}
    ground_truth_filename_replacements = {}

    # Allow bypassing API output loading when operating on previously-loaded
    # results
    api_detection_results: Optional[pd.DataFrame] = None
    api_other_fields: Optional[Dict[str, Any]] = None

    # Should we also split out a separate report about the detections that were
    # just below our main confidence threshold?
    #
    # Currently only supported when ground truth is unavailable
    include_almost_detections = False
    almost_detection_confidence_threshold = None

    # Control rendering parallelization
    parallelize_rendering_n_cores: Optional[int] = 100
    parallelize_rendering_with_threads = True
    parallelize_rendering = False
    
    sort_classification_results_by_count = False    
    
# ...PostProcessingOptions


class PostProcessingResults:

    output_html_file = ''
    api_detection_results: Optional[pd.DataFrame] = None
    api_other_fields: Optional[Dict[str, Any]] = None


##%% Helper classes and functions

class DetectionStatus(IntEnum):
    """
    Flags used to mark images as positive or negative for P/R analysis
    (according to ground truth and/or detector output)
    """
    
    DS_NEGATIVE = 0
    DS_POSITIVE = 1

    # Anything greater than this isn't clearly positive or negative
    DS_MAX_DEFINITIVE_VALUE = DS_POSITIVE

    # image has annotations suggesting both negative and positive
    DS_AMBIGUOUS = 2

    # image is not annotated or is annotated with 'unknown', 'unlabeled', ETC.
    DS_UNKNOWN = 3

    # image has not yet been assigned a state
    DS_UNASSIGNED = 4

    # In some analyses, we add an additional class that lets us look at
    # detections just below our main confidence threshold
    DS_ALMOST = 5


def mark_detection_status(
        indexed_db: IndexedJsonDb,
        negative_classes: Iterable[str] = DEFAULT_NEGATIVE_CLASSES,
        unknown_classes: Iterable[str] = DEFAULT_UNKNOWN_CLASSES
        ) -> Tuple[int, int, int, int]:
    """
    For each image in indexed_db.db['images'], add a '_detection_status' field
    to indicate whether to treat this image as positive, negative, ambiguous,
    or unknown.

    Makes modifications in-place.

    returns (n_negative, n_positive, n_unknown, n_ambiguous)
    """
    
    negative_classes = set(negative_classes)
    unknown_classes = set(unknown_classes)

    # count the # of images with each type of DetectionStatus
    n_unknown = 0
    n_ambiguous = 0
    n_positive = 0
    n_negative = 0

    print('Preparing ground-truth annotations')
    for im in tqdm(indexed_db.db['images']):

        image_id = im['id']
        annotations = indexed_db.image_id_to_annotations[image_id]
        categories = [ann['category_id'] for ann in annotations]
        category_names = set(indexed_db.cat_id_to_name[cat] for cat in categories)

        # Check whether this image has:
        # - unknown / unassigned-type labels
        # - negative-type labels
        # - positive labels (i.e., labels that are neither unknown nor negative)
        has_unknown_labels = has_overlap(category_names, unknown_classes)
        has_negative_labels = has_overlap(category_names, negative_classes)
        has_positive_labels = 0 < len(category_names - (unknown_classes | negative_classes))
        # assert has_unknown_labels is False, '{} has unknown labels'.format(annotations)

        # If there are no image annotations...
        if len(categories) == 0:
                        
            if '#NO_LABELS#' in negative_classes:
                n_negative += 1
                im['_detection_status'] = DetectionStatus.DS_NEGATIVE
            else:
                n_unknown += 1
                im['_detection_status'] = DetectionStatus.DS_UNKNOWN

            # n_negative += 1
            # im['_detection_status'] = DetectionStatus.DS_NEGATIVE

        # If the image has more than one type of labels, it's ambiguous
        # note: bools are automatically converted to 0/1, so we can sum
        elif (has_unknown_labels + has_negative_labels + has_positive_labels) > 1:
            n_ambiguous += 1
            im['_detection_status'] = DetectionStatus.DS_AMBIGUOUS

        # After the check above, we can be sure it's only one of positive,
        # negative, or unknown.
        #
        # Important: do not merge the following 'unknown' branch with the first
        # 'unknown' branch above, where we tested 'if len(categories) == 0'
        #
        # If the image has only unknown labels
        elif has_unknown_labels:
            n_unknown += 1
            im['_detection_status'] = DetectionStatus.DS_UNKNOWN

        # If the image has only negative labels
        elif has_negative_labels:
            n_negative += 1
            im['_detection_status'] = DetectionStatus.DS_NEGATIVE

        # If the images has only positive labels
        elif has_positive_labels:
            n_positive += 1
            im['_detection_status'] = DetectionStatus.DS_POSITIVE

            # Annotate the category, if it is unambiguous
            if len(category_names) == 1:
                im['_unambiguous_category'] = list(category_names)[0]

        else:
            raise Exception('Invalid detection state')

    # ...for each image

    return n_negative, n_positive, n_unknown, n_ambiguous

# ...mark_detection_status()


def is_sas_url(s: str) -> bool:
    """
    Placeholder for a more robust way to verify that a link is a SAS URL.
    99.999% of the time this will suffice for what we're using it for right now.
    """
    
    return (s.startswith(('http://', 'https://')) and ('core.windows.net' in s)
            and ('?' in s))


def relative_sas_url(folder_url: str, relative_path: str) -> Optional[str]:
    """
    Given a container-level or folder-level SAS URL, create a SAS URL to the
    specified relative path.
    """
    
    relative_path = relative_path.replace('%','%25')
    relative_path = relative_path.replace('#','%23')
    relative_path = relative_path.replace(' ','%20')

    if not is_sas_url(folder_url):
        return None
    tokens = folder_url.split('?')
    assert len(tokens) == 2
    if not tokens[0].endswith('/'):
        tokens[0] = tokens[0] + '/'
    if relative_path.startswith('/'):
        relative_path = relative_path[1:]
    return tokens[0] + relative_path + '?' + tokens[1]


def render_bounding_boxes(
        image_base_dir,
        image_relative_path,
        display_name,
        detections,
        res,
        ground_truth_boxes=None,
        detection_categories=None,
        classification_categories=None,
        options=None):
    """
    Renders detection bounding boxes on a single image.

    The source image is:

        image_base_dir / image_relative_path

    The target image is, for example:

        [options.output_dir] / 
        ['detections' or 'non_detections'] / 
        [filename with slashes turned into tildes]

    "res" is a result type, e.g. "detections", "non-detections"; this determines the
    output folder for the rendered image.
    
    Only very preliminary support is provided for ground truth box rendering.
    
    Returns the html info struct for this image in the form that's used for
    write_html_image_list.
    """

    if options is None:
        options = PostProcessingOptions()

    # Leaving code in place for reading from blob storage, may support this
    # in the future.
    """
    stream = io.BytesIO()
    _ = blob_service.get_blob_to_stream(container_name, image_id, stream)
    # resize is to display them in this notebook or in the HTML more quickly
    image = Image.open(stream).resize(viz_size)
    """

    image_full_path = None
    
    if res in options.rendering_bypass_sets:

        sample_name = res + '_' + path_utils.flatten_path(image_relative_path)

    else:

        if is_sas_url(image_base_dir):
            image_full_path = relative_sas_url(image_base_dir, image_relative_path)
        else:
            image_full_path = os.path.join(image_base_dir, image_relative_path)

        # os.path.isfile() is slow when mounting remote directories; much faster
        # to just try/except on the image open.
        try:
            image = vis_utils.open_image(image_full_path)
        except:
            print('Warning: could not open image file {}'.format(image_full_path))
            image = None
            # return ''
        
        # Render images to a flat folder... we can use os.sep here because we've
        # already normalized paths
        sample_name = res + '_' + path_utils.flatten_path(image_relative_path)
        fullpath = os.path.join(options.output_dir, res, sample_name)

        if image is not None:
            
            original_size = image.size
            
            if options.viz_target_width is not None:
                image = vis_utils.resize_image(image, options.viz_target_width)
    
            if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
                
                # Create class labels like "gt_1" or "gt_27"
                gt_classes = [0] * len(ground_truth_boxes)
                label_map = {0:'ground truth'}
                # for i_box,box in enumerate(ground_truth_boxes):
                #    gt_classes.append('_' + str(box[-1]))
                vis_utils.render_db_bounding_boxes(ground_truth_boxes, gt_classes, image,
                                                   original_size=original_size,label_map=label_map,
                                                   thickness=4,expansion=4)
                
            vis_utils.render_detection_bounding_boxes(
                detections, image,
                label_map=detection_categories,
                classification_label_map=classification_categories,
                confidence_threshold=options.confidence_threshold,
                thickness=options.line_thickness,
                expansion=options.box_expansion)
    
            try:
                image.save(fullpath)
            except OSError as e:
                # errno.ENAMETOOLONG doesn't get thrown properly on Windows, so
                # we awkwardly check against a hard-coded limit
                if (e.errno == errno.ENAMETOOLONG) or (len(fullpath) >= 259):
                    extension = os.path.splitext(sample_name)[1]
                    sample_name = res + '_' + str(uuid.uuid4()) + extension
                    image.save(os.path.join(options.output_dir, res, sample_name))
                else:
                    raise

    # Use slashes regardless of os
    file_name = '{}/{}'.format(res,sample_name)

    info = {
        'filename': file_name,
        'title': display_name,
        'textStyle':\
         'font-family:verdana,arial,calibri;font-size:80%;text-align:left;margin-top:20;margin-bottom:5'
    }
    
    # Optionally add links back to the original images
    if options.link_images_to_originals and (image_full_path is not None):
        info['linkTarget'] = urllib.parse.quote(image_full_path)
        
    return info

# ...render_bounding_boxes


def prepare_html_subpages(images_html, output_dir, options=None):
    """
    Write out a series of html image lists, e.g. the fp/tp/fn/tn pages.

    image_html is a dictionary mapping an html page name (e.g. "fp") to a list
    of image structs friendly to write_html_image_list
    """
    
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

# ...prepare_html_subpages()

# Get unique categories above the threshold for this image
def get_positive_categories(detections,options):
    positive_categories = set()
    for d in detections:
        if d['conf'] >= options.confidence_threshold:
            positive_categories.add(d['category'])
    return sorted(positive_categories)

# Render an image (with no ground truth information)
def render_image_no_gt(file_info,detection_categories_to_results_name,
                       detection_categories,classification_categories,
                       options):

    image_relative_path = file_info[0]
    max_conf = file_info[1]
    detections = file_info[2]

    detection_status = DetectionStatus.DS_UNASSIGNED
    if max_conf >= options.confidence_threshold:
        detection_status = DetectionStatus.DS_POSITIVE
    else:
        if options.include_almost_detections:
            if max_conf >= options.almost_detection_confidence_threshold:
                detection_status = DetectionStatus.DS_ALMOST
            else:
                detection_status = DetectionStatus.DS_NEGATIVE
        else:
            detection_status = DetectionStatus.DS_NEGATIVE

    if detection_status == DetectionStatus.DS_POSITIVE:
        if options.separate_detections_by_category:
            positive_categories = tuple(get_positive_categories(detections,options))            
            if positive_categories not in detection_categories_to_results_name:
                raise ValueError('Error: {} not in category mapping (file {})'.format(
                    str(positive_categories),image_relative_path))
            res = detection_categories_to_results_name[positive_categories]
        else:
            res = 'detections'

    elif detection_status == DetectionStatus.DS_NEGATIVE:
        res = 'non_detections'
    else:
        assert detection_status == DetectionStatus.DS_ALMOST
        res = 'almost_detections'

    display_name = '<b>Result type</b>: {}, <b>Image</b>: {}, <b>Max conf</b>: {:0.3f}'.format(
        res, image_relative_path, max_conf)

    rendering_options = copy.copy(options)
    if detection_status == DetectionStatus.DS_ALMOST:
        rendering_options.confidence_threshold = \
            rendering_options.almost_detection_confidence_threshold
            
    rendered_image_html_info = render_bounding_boxes(
        image_base_dir=options.image_base_dir,
        image_relative_path=image_relative_path,
        display_name=display_name,
        detections=detections,
        res=res,
        ground_truth_boxes=None,
        detection_categories=detection_categories,
        classification_categories=classification_categories,
        options=rendering_options)

    image_result = None

    if len(rendered_image_html_info) > 0:

        image_result = [[res, rendered_image_html_info]]

        for det in detections:

            if ('classifications' in det):

                # This is a list of [class,confidence] pairs, sorted by confidence
                classifications = det['classifications']
                top1_class_id = classifications[0][0]
                top1_class_name = classification_categories[top1_class_id]
                top1_class_score = classifications[0][1]

                # If we either don't have a confidence threshold, or we've met our
                # confidence threshold
                if (options.classification_confidence_threshold < 0) or \
                    (top1_class_score >= options.classification_confidence_threshold):
                    image_result.append(['class_{}'.format(top1_class_name),
                                         rendered_image_html_info])
                else:
                    image_result.append(['class_unreliable',
                                         rendered_image_html_info])

            # ...if this detection has classification info

        # ...for each detection

    return image_result

# ...def render_image_no_gt()
    

def render_image_with_gt(file_info,ground_truth_indexed_db,
                         detection_categories,classification_categories,options):

    image_relative_path = file_info[0]
    max_conf = file_info[1]
    detections = file_info[2]

    # This should already have been normalized to either '/' or '\'

    image_id = ground_truth_indexed_db.filename_to_id.get(image_relative_path, None)
    if image_id is None:
        print('Warning: couldn''t find ground truth for image {}'.format(image_relative_path))
        return None

    image = ground_truth_indexed_db.image_id_to_image[image_id]
    annotations = ground_truth_indexed_db.image_id_to_annotations[image_id]

    ground_truth_boxes = []
    for ann in annotations:
        if 'bbox' in ann:
            ground_truth_box = [x for x in ann['bbox']]
            ground_truth_box.append(ann['category_id'])
            ground_truth_boxes.append(ground_truth_box)
    
    gt_status = image['_detection_status']

    gt_presence = bool(gt_status)

    gt_classes = CameraTrapJsonUtils.annotations_to_classnames(
        annotations, ground_truth_indexed_db.cat_id_to_name)
    gt_class_summary = ','.join(gt_classes)

    if gt_status > DetectionStatus.DS_MAX_DEFINITIVE_VALUE:
        print(f'Skipping image {image_id}, does not have a definitive '
              f'ground truth status (status: {gt_status}, classes: {gt_class_summary})')
        return None

    detected = max_conf > options.confidence_threshold

    if gt_presence and detected:
        if '_classification_accuracy' not in image.keys():
            res = 'tp'
        elif np.isclose(1, image['_classification_accuracy']):
            res = 'tpc'
        else:
            res = 'tpi'
    elif not gt_presence and detected:
        res = 'fp'
    elif gt_presence and not detected:
        res = 'fn'
    else:
        res = 'tn'

    display_name = '<b>Result type</b>: {}, <b>Presence</b>: {}, <b>Class</b>: {}, <b>Max conf</b>: {:0.3f}%, <b>Image</b>: {}'.format(
        res.upper(), str(gt_presence), gt_class_summary,
        max_conf * 100, image_relative_path)

    rendered_image_html_info = render_bounding_boxes(
        image_base_dir=options.image_base_dir,
        image_relative_path=image_relative_path,
        display_name=display_name,
        detections=detections,
        res=res,
        ground_truth_boxes=ground_truth_boxes,
        detection_categories=detection_categories,
        classification_categories=classification_categories,
        options=options)

    image_result = None
    if len(rendered_image_html_info) > 0:
        image_result = [[res, rendered_image_html_info]]
        for gt_class in gt_classes:
            image_result.append(['class_{}'.format(gt_class), rendered_image_html_info])

    return image_result

# ...def render_image_with_gt()

    
#%% Main function

def process_batch_results(options: PostProcessingOptions
                          ) -> PostProcessingResults:

    ppresults = PostProcessingResults()

    ##%% Expand some options for convenience

    output_dir = options.output_dir


    ##%% Prepare output dir

    os.makedirs(output_dir, exist_ok=True)


    ##%% Load ground truth if available

    ground_truth_indexed_db = None

    if (options.ground_truth_json_file is not None) and (len(options.ground_truth_json_file) > 0):

        if options.separate_detections_by_category:
            print("Warning: I don't know how to separate categories yet when doing " + \
                  "a P/R analysis, disabling category separation")
            options.separate_detections_by_category = False

        ground_truth_indexed_db = IndexedJsonDb(
            options.ground_truth_json_file, b_normalize_paths=True,
            filename_replacements=options.ground_truth_filename_replacements)

        # Mark images in the ground truth as positive or negative
        n_negative, n_positive, n_unknown, n_ambiguous = mark_detection_status(
            ground_truth_indexed_db, negative_classes=options.negative_classes,
            unknown_classes=options.unlabeled_classes)
        print(f'Finished loading and indexing ground truth: {n_negative} '
              f'negative, {n_positive} positive, {n_unknown} unknown, '
              f'{n_ambiguous} ambiguous')


    ##%% Load detection (and possibly classification) results

    # If the caller hasn't supplied results, load them
    if options.api_detection_results is None:
        detections_df, other_fields = load_api_results(
            options.api_output_file, normalize_paths=True,
            filename_replacements=options.api_output_filename_replacements)
        ppresults.api_detection_results = detections_df
        ppresults.api_other_fields = other_fields        

    else:
        print('Bypassing detection results loading...')
        assert options.api_other_fields is not None
        detections_df = options.api_detection_results
        other_fields = options.api_other_fields

    # Determine confidence thresholds if necessary
    
    if options.confidence_threshold is None:
        options.confidence_threshold = \
            get_typical_confidence_threshold_from_results(other_fields)
        print('Choosing default confidence threshold of {} based on MD version'.format(
            options.confidence_threshold))    
            
    if options.almost_detection_confidence_threshold is None:
        options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
        if options.almost_detection_confidence_threshold < 0:
            options.almost_detection_confidence_threshold = 0
            
    # Remove failed rows
    n_failures = 0
    if 'failure' in detections_df.columns:
        n_failures = detections_df['failure'].count()
        print('Warning: {} failed images'.format(n_failures))
        detections_df = detections_df[detections_df['failure'].isna()]
    
    assert other_fields is not None

    detection_categories = other_fields['detection_categories']
    
    # Convert keys and values to lowercase
    classification_categories = other_fields.get('classification_categories', {})
    if classification_categories is not None:
        classification_categories = {
            k.lower(): v.lower()
            for k, v in classification_categories.items()
        }

    # Add column 'pred_detection_label' to indicate predicted detection status,
    # not separating out the classes
    det_status = 'pred_detection_label'
    if options.include_almost_detections:
        detections_df[det_status] = DetectionStatus.DS_ALMOST
        confidences = detections_df['max_detection_conf']

        pos_mask = (confidences >= options.confidence_threshold)
        detections_df.loc[pos_mask, det_status] = DetectionStatus.DS_POSITIVE

        neg_mask = (confidences < options.almost_detection_confidence_threshold)
        detections_df.loc[neg_mask, det_status] = DetectionStatus.DS_NEGATIVE
    else:
        detections_df[det_status] = np.where(
            detections_df['max_detection_conf'] >= options.confidence_threshold,
            DetectionStatus.DS_POSITIVE, DetectionStatus.DS_NEGATIVE)

    n_positives = sum(detections_df[det_status] == DetectionStatus.DS_POSITIVE)
    print(f'Finished loading and preprocessing {len(detections_df)} rows '
          f'from detector output, predicted {n_positives} positives.')

    if options.include_almost_detections:
        n_almosts = sum(detections_df[det_status] == DetectionStatus.DS_ALMOST)
        print('...and {} almost-positives'.format(n_almosts))


    ##%% Pull out descriptive metadata

    if options.job_name_string is not None:
        job_name_string = options.job_name_string
    else:
        # This is rare; it only happens during debugging when the caller
        # is supplying already-loaded API results.
        if options.api_output_file is None:
            job_name_string = 'unknown'
        else:
            job_name_string = os.path.basename(options.api_output_file)
    
    if options.model_version_string is not None:
        model_version_string = options.model_version_string
    else:
        
        if 'info' not in other_fields or 'detector' not in other_fields['info']:
            print('No model metadata supplied, assuming MDv4')
            model_version_string = 'MDv4 (assumed)'
        else:            
            model_version_string = other_fields['info']['detector']
    
        
    ##%% If we have ground truth, remove images we can't match to ground truth

    if ground_truth_indexed_db is not None:

        b_match = detections_df['file'].isin(
            ground_truth_indexed_db.filename_to_id)
        print(f'Confirmed filename matches to ground truth for {sum(b_match)} '
              f'of {len(detections_df)} files')

        detections_df = detections_df[b_match]
        detector_files = detections_df['file'].tolist()

        assert len(detector_files) > 0, (
            'No detection files available, possible path issue?')

        print('Trimmed detection results to {} files'.format(len(detector_files)))


    ##%% Sample images for visualization

    images_to_visualize = detections_df

    if options.num_images_to_sample is not None and options.num_images_to_sample > 0:
        images_to_visualize = images_to_visualize.sample(
            n=min(options.num_images_to_sample, len(images_to_visualize)),
            random_state=options.sample_seed)

    output_html_file = ''

    style_header = """<head>
        <style type="text/css">
        a { text-decoration: none; }
        body { font-family: segoe ui, calibri, "trebuchet ms", verdana, arial, sans-serif; }
        div.contentdiv { margin-left: 20px; }
        </style>
        </head>"""


    ##%% Fork here depending on whether or not ground truth is available

    # If we have ground truth, we'll compute precision/recall and sample tp/fp/tn/fn.
    #
    # Otherwise we'll just visualize detections/non-detections.

    if ground_truth_indexed_db is not None:

        ##%% Detection evaluation: compute precision/recall

        # numpy array of detection probabilities
        p_detection = detections_df['max_detection_conf'].values
        n_detections = len(p_detection)

        # numpy array of bools (0.0/1.0), and -1 as null value
        gt_detections = np.zeros(n_detections, dtype=float)

        for i_detection, fn in enumerate(detector_files):
            image_id = ground_truth_indexed_db.filename_to_id[fn]
            image = ground_truth_indexed_db.image_id_to_image[image_id]
            detection_status = image['_detection_status']

            if detection_status == DetectionStatus.DS_NEGATIVE:
                gt_detections[i_detection] = 0.0
            elif detection_status == DetectionStatus.DS_POSITIVE:
                gt_detections[i_detection] = 1.0
            else:
                gt_detections[i_detection] = -1.0

        # Don't include ambiguous/unknown ground truth in precision/recall analysis
        b_valid_ground_truth = gt_detections >= 0.0

        p_detection_pr = p_detection[b_valid_ground_truth]
        gt_detections_pr = (gt_detections[b_valid_ground_truth] == 1.)

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
        print('Average precision: {:.1%}'.format(average_precision))

        # Thresholds go up throughout precisions/recalls/thresholds; find the last
        # value where recall is at or above target.  That's our precision @ target recall.
        
        i_above_target_recall = (np.where(recalls >= options.target_recall))
        
        # np.where returns a tuple of arrays, but in this syntax where we're 
        # comparing an array with a scalar, there will only be one element.
        assert len (i_above_target_recall) == 1
        
        # Convert back to a list
        i_above_target_recall = i_above_target_recall[0].tolist()
        
        if len(i_above_target_recall) == 0:
            precision_at_target_recall = 0.0
        else:
            precision_at_target_recall = precisions[i_above_target_recall[-1]]
        print('Precision at {:.1%} recall: {:.1%}'.format(options.target_recall,
                                                          precision_at_target_recall))

        cm_predictions = np.array(p_detection_pr) > options.confidence_threshold
        cm = confusion_matrix(gt_detections_pr, cm_predictions, labels=[False,True])

        # Flatten the confusion matrix
        tn, fp, fn, tp = cm.ravel()

        precision_at_confidence_threshold = tp / (tp + fp)
        recall_at_confidence_threshold = tp / (tp + fn)
        f1 = 2.0 * (precision_at_confidence_threshold * recall_at_confidence_threshold) / \
            (precision_at_confidence_threshold + recall_at_confidence_threshold)

        print('At a confidence threshold of {:.1%}, precision={:.1%}, recall={:.1%}, f1={:.1%}'.format(
                options.confidence_threshold, precision_at_confidence_threshold,
                recall_at_confidence_threshold, f1))

        ##%% Collect classification results, if they exist

        classifier_accuracies = []

        # Mapping of classnames to idx for the confusion matrix.
        #
        # The lambda is actually kind of a hack, because we use assume that
        # the following code does not reassign classname_to_idx
        classname_to_idx = collections.defaultdict(lambda: len(classname_to_idx))

        # Confusion matrix as defaultdict of defaultdict
        #
        # Rows / first index is ground truth, columns / second index is predicted category
        classifier_cm = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))

        # iDetection = 0; fn = detector_files[iDetection]; print(fn)
        assert len(detector_files) == len(detections_df)
        for iDetection, fn in enumerate(detector_files):

            image_id = ground_truth_indexed_db.filename_to_id[fn]
            image = ground_truth_indexed_db.image_id_to_image[image_id]
            detections = detections_df['detections'].iloc[iDetection]
            pred_class_ids = [det['classifications'][0][0] \
                for det in detections if 'classifications' in det.keys()]
            pred_classnames = [classification_categories[pd] for pd in pred_class_ids]

            # If this image has classification predictions, and an unambiguous class
            # annotated, and is a positive image...
            if len(pred_classnames) > 0 \
                    and '_unambiguous_category' in image.keys() \
                    and image['_detection_status'] == DetectionStatus.DS_POSITIVE:

                # The unambiguous category, we make this a set for easier handling afterward
                gt_categories = set([image['_unambiguous_category']])
                pred_categories = set(pred_classnames)

                # Compute the accuracy as intersection of union,
                # i.e. (# of categories in both prediciton and GT)
                #      divided by (# of categories in either prediction or GT
                #
                # In case of only one GT category, the result will be 1.0, if
                # prediction is one category and this category matches GT
                #
                # It is 1.0/(# of predicted top-1 categories), if the GT is
                # one of the predicted top-1 categories.
                #
                # It is 0.0, if none of the predicted categories is correct

                classifier_accuracies.append(
                    len(gt_categories & pred_categories)
                    / len(gt_categories | pred_categories)
                )
                image['_classification_accuracy'] = classifier_accuracies[-1]

                # Distribute this accuracy across all predicted categories in the
                # confusion matrix
                assert len(gt_categories) == 1
                gt_class_idx = classname_to_idx[list(gt_categories)[0]]
                for pred_category in pred_categories:
                    pred_class_idx = classname_to_idx[pred_category]
                    classifier_cm[gt_class_idx][pred_class_idx] += 1

        # ...for each file in the detection results

        # If we have classification results
        if len(classifier_accuracies) > 0:

            # Build confusion matrix as array from classifier_cm
            all_class_ids = sorted(classname_to_idx.values())
            classifier_cm_array = np.array(
                [[classifier_cm[r_idx][c_idx] for c_idx in all_class_ids] for \
                 r_idx in all_class_ids], dtype=float)
            classifier_cm_array /= (classifier_cm_array.sum(axis=1, keepdims=True) + 1e-7)

            # Print some statistics
            print('Finished computation of {} classification results'.format(
                len(classifier_accuracies)))
            print('Mean accuracy: {}'.format(np.mean(classifier_accuracies)))

            # Prepare confusion matrix output

            # Get confusion matrix as string
            sio = io.StringIO()
            np.savetxt(sio, classifier_cm_array * 100, fmt='%5.1f')
            cm_str = sio.getvalue()
            # Get fixed-size classname for each idx
            idx_to_classname = {v:k for k,v in classname_to_idx.items()}
            classname_list = [idx_to_classname[idx] for idx in sorted(classname_to_idx.values())]
            classname_headers = ['{:<5}'.format(cname[:5]) for cname in classname_list]

            # Prepend class name on each line and add to the top
            cm_str_lines = [' ' * 16 + ' '.join(classname_headers)]
            cm_str_lines += ['{:>15}'.format(cn[:15]) + ' ' + cm_line for cn, cm_line in \
                             zip(classname_list, cm_str.splitlines())]

            # Print formatted confusion matrix
            if False:
                # Actually don't, this gets really messy in all but the widest consoles
                print('Confusion matrix: ')
                print(*cm_str_lines, sep='\n')

            # Plot confusion matrix

            # To manually add more space at bottom: plt.rcParams['figure.subplot.bottom'] = 0.1
            #
            # Add 0.5 to figsize for every class. For two classes, this will result in
            # fig = plt.figure(figsize=[4,4])
            fig = plot_utils.plot_confusion_matrix(
                classifier_cm_array,
                classname_list,
                normalize=False,
                title='Confusion matrix',
                cmap=plt.cm.Blues,
                vmax=1.0,
                use_colorbar=True,
                y_label=True)
            cm_figure_relative_filename = 'confusion_matrix.png'
            cm_figure_filename = os.path.join(output_dir, cm_figure_relative_filename)
            plt.savefig(cm_figure_filename)
            plt.close(fig)

        # ...if we have classification results


        ##%% Render output

        # Write p/r table to .csv file in output directory
        pr_table_filename = os.path.join(output_dir, 'prec_recall.csv')
        precisions_recalls.to_csv(pr_table_filename, index=False)

        # Write precision/recall plot to .png file in output directory
        t = 'Precision-Recall curve: AP={:0.1%}, P@{:0.1%}={:0.1%}'.format(
            average_precision, options.target_recall, precision_at_target_recall)
        fig = plot_utils.plot_precision_recall_curve(precisions, recalls, t)
        
        pr_figure_relative_filename = 'prec_recall.png'
        pr_figure_filename = os.path.join(output_dir, pr_figure_relative_filename)
        fig.savefig(pr_figure_filename)
        plt.close(fig)


        ##%% Sampling

        # Sample true/false positives/negatives with correct/incorrect top-1
        # classification and render to html

        # Accumulate html image structs (in the format expected by write_html_image_lists)
        # for each category, e.g. 'tp', 'fp', ..., 'class_bird', ...
        images_html = collections.defaultdict(list)
        
        # Add default entries by accessing them for the first time
        [images_html[res] for res in ['tp', 'tpc', 'tpi', 'fp', 'tn', 'fn']]
        for res in images_html.keys():
            os.makedirs(os.path.join(output_dir, res), exist_ok=True)

        image_count = len(images_to_visualize)

        # Each element will be a list of 2-tuples, with elements [collection name,html info struct]
        rendering_results = []

        # Each element will be a three-tuple with elements file,max_conf,detections
        files_to_render = []

        # Assemble the information we need for rendering, so we can parallelize without
        # dealing with Pandas
        # i_row = 0; row = images_to_visualize.iloc[0]
        for _, row in images_to_visualize.iterrows():

            # Filenames should already have been normalized to either '/' or '\'
            files_to_render.append([row['file'], row['max_detection_conf'], row['detections']])

        start_time = time.time()
        if options.parallelize_rendering:
            if options.parallelize_rendering_n_cores is None:                
                if options.parallelize_rendering_with_threads:
                    pool = ThreadPool()
                else:
                    pool = Pool()
            else:
                if options.parallelize_rendering_with_threads:
                    pool = ThreadPool(options.parallelize_rendering_n_cores)
                    worker_string = 'threads'
                else:
                    pool = Pool(options.parallelize_rendering_n_cores)
                    worker_string = 'processes'
                print('Rendering images with {} {}'.format(options.parallelize_rendering_n_cores,
                                                           worker_string))
                
            rendering_results = list(tqdm(pool.imap(
                partial(render_image_with_gt,
                        ground_truth_indexed_db=ground_truth_indexed_db,
                        detection_categories=detection_categories,
                        classification_categories=classification_categories,
                        options=options), 
                files_to_render), total=len(files_to_render)))
        else:
            for file_info in tqdm(files_to_render):
                rendering_results.append(render_image_with_gt(
                    file_info,ground_truth_indexed_db,
                    detection_categories,classification_categories))
        elapsed = time.time() - start_time

        # Map all the rendering results in the list rendering_results into the
        # dictionary images_html
        image_rendered_count = 0
        for rendering_result in rendering_results:
            if rendering_result is None:
                continue
            image_rendered_count += 1
            for assignment in rendering_result:
                images_html[assignment[0]].append(assignment[1])

        # Prepare the individual html image files
        image_counts = prepare_html_subpages(images_html, output_dir)

        print('{} images rendered (of {})'.format(image_rendered_count,image_count))

        # Write index.html
        all_tp_count = image_counts['tp'] + image_counts['tpc'] + image_counts['tpi']
        total_count = all_tp_count + image_counts['tn'] + image_counts['fp'] + image_counts['fn']

        classification_detection_results = """&nbsp;&nbsp;&nbsp;&nbsp;<a href="tpc.html">with all correct top-1 predictions (TPC)</a> ({})<br/>
           &nbsp;&nbsp;&nbsp;&nbsp;<a href="tpi.html">with one or more incorrect top-1 prediction (TPI)</a> ({})<br/>
           &nbsp;&nbsp;&nbsp;&nbsp;<a href="tp.html">without classification evaluation</a><sup>*</sup> ({})<br/>""".format(
            image_counts['tpc'],
            image_counts['tpi'],
            image_counts['tp']
        )

        index_page = """<html>
        {}
        <body>
        <h2>Evaluation</h2>

        <h3>Job metadata</h3>
        
        <div class="contentdiv">
        <p>Job name: {}<br/>
        <p>Model version: {}</p>
        </div>
        
        <h3>Sample images</h3>
        <div class="contentdiv">
        <p>A sample of {} images, annotated with detections above {:.1%} confidence.</p>
        <a href="tp.html">True positives (TP)</a> ({}) ({:0.1%})<br/>
        CLASSIFICATION_PLACEHOLDER_1
        <a href="tn.html">True negatives (TN)</a> ({}) ({:0.1%})<br/>
        <a href="fp.html">False positives (FP)</a> ({}) ({:0.1%})<br/>
        <a href="fn.html">False negatives (FN)</a> ({}) ({:0.1%})<br/>
        CLASSIFICATION_PLACEHOLDER_2
        </div>
        """.format(
            style_header,job_name_string,model_version_string,
            image_count, options.confidence_threshold,
            all_tp_count, all_tp_count/total_count,
            image_counts['tn'], image_counts['tn']/total_count,
            image_counts['fp'], image_counts['fp']/total_count,
            image_counts['fn'], image_counts['fn']/total_count
        )

        index_page += """
            <h3>Detection results</h3>
            <div class="contentdiv">
            <p>At a confidence threshold of {:0.1%}, precision={:0.1%}, recall={:0.1%}</p>
            <p><strong>Precision/recall summary for all {} images</strong></p><img src="{}"><br/>
            </div>
            """.format(
                options.confidence_threshold, precision_at_confidence_threshold, recall_at_confidence_threshold,
                len(detections_df), pr_figure_relative_filename
           )

        if len(classifier_accuracies) > 0:
            index_page = index_page.replace('CLASSIFICATION_PLACEHOLDER_1',classification_detection_results)
            index_page = index_page.replace('CLASSIFICATION_PLACEHOLDER_2',"""<p><sup>*</sup>We do not evaluate the classification result of images
                if the classification information is missing, if the image contains
                categories like &lsquo;empty&rsquo; or &lsquo;human&rsquo;, or if the image has multiple
                classification labels.</p>""")
        else:
            index_page = index_page.replace('CLASSIFICATION_PLACEHOLDER_1','')
            index_page = index_page.replace('CLASSIFICATION_PLACEHOLDER_2','')

        if len(classifier_accuracies) > 0:
            index_page += """
                <h3>Classification results</h3>
                <div class="contentdiv">
                <p>Classification accuracy: {:.2%}<br>
                The accuracy is computed only for images with exactly one classification label.
                The accuracy of an image is computed as 1/(number of unique detected top-1 classes),
                i.e. if the model detects multiple boxes with different top-1 classes, then the accuracy
                decreases and the image is put into 'TPI'.</p>
                <p>Confusion matrix:</p>
                <p><img src="{}"></p>
                <div style='font-family:monospace;display:block;'>{}</div>
                </div>
                """.format(
                    np.mean(classifier_accuracies),
                    cm_figure_relative_filename,
                    "<br>".join(cm_str_lines).replace(' ', '&nbsp;')
                )

        # Show links to each GT class
        #
        # We could do this without classification results; currently we don't.
        if len(classname_to_idx) > 0:

            index_page += '<h3>Images of specific classes</h3><br/><div class="contentdiv">'
            # Add links to all available classes
            for cname in sorted(classname_to_idx.keys()):
                index_page += '<a href="class_{0}.html">{0}</a> ({1})<br>'.format(
                    cname,
                    len(images_html['class_{}'.format(cname)]))
            index_page += '</div>'

        # Close body and html tags
        index_page += '</body></html>'
        output_html_file = os.path.join(output_dir, 'index.html')
        with open(output_html_file, 'w') as f:
            f.write(index_page)

        print('Finished writing html to {}'.format(output_html_file))

    # ...for each image


    ##%% Otherwise, if we don't have ground truth...

    else:

        ##%% Sample detections/non-detections

        # Accumulate html image structs (in the format expected by write_html_image_list)
        # for each category
        images_html = collections.defaultdict(list)
        images_html['non_detections']

        # Add default entries by accessing them for the first time

        # Maps detection categories - e.g. "human" - to result set names, e.g.
        # "detections_human"
        detection_categories_to_results_name = {}

        if not options.separate_detections_by_category:
            images_html['detections']
        else:
            # Add a set of results for each category and combination of categories
            keys = detection_categories.keys()
            subsets = []
            for L in range(1, len(keys)+1):
                for subset in itertools.combinations(keys, L):
                    subsets.append(subset)
            for subset in subsets:
                sorted_subset = tuple(sorted(subset))
                results_name = 'detections'
                for category_id in sorted_subset:
                    results_name = results_name + '_' + detection_categories[category_id]
                images_html[results_name]
                detection_categories_to_results_name[sorted_subset] = results_name

        if options.include_almost_detections:
            images_html['almost_detections']

        # Create output directories
        for res in images_html.keys():
            os.makedirs(os.path.join(output_dir, res), exist_ok=True)

        image_count = len(images_to_visualize)
        has_classification_info = False

        # Each element will be a list of 2-tuples, with elements [collection name,html info struct]
        rendering_results = []

        # list of 3-tuples with elements (file, max_conf, detections)
        files_to_render = []

        # Assemble the information we need for rendering, so we can parallelize without
        # dealing with Pandas
        # i_row = 0; row = images_to_visualize.iloc[0]
        for _, row in images_to_visualize.iterrows():

            assert isinstance(row['detections'],list)
            
            # Filenames should already have been normalized to either '/' or '\'
            files_to_render.append([row['file'],
                                    row['max_detection_conf'],
                                    row['detections']])

        start_time = time.time()
        if options.parallelize_rendering:
            
            if options.parallelize_rendering_n_cores is None:                
                if options.parallelize_rendering_with_threads:
                    pool = ThreadPool()
                else:
                    pool = Pool()
            else:
                if options.parallelize_rendering_with_threads:
                    pool = ThreadPool(options.parallelize_rendering_n_cores)
                    worker_string = 'threads'
                else:
                    pool = Pool(options.parallelize_rendering_n_cores)
                    worker_string = 'processes'
                print('Rendering images with {} {}'.format(options.parallelize_rendering_n_cores,
                                                           worker_string))
                
            # render_image_no_gt(file_info,detection_categories_to_results_name,
            # detection_categories,classification_categories)

            rendering_results = list(tqdm(pool.imap(
                partial(render_image_no_gt, 
                        detection_categories_to_results_name=detection_categories_to_results_name,
                        detection_categories=detection_categories,
                        classification_categories=classification_categories,
                        options=options),
                        files_to_render), total=len(files_to_render)))
        else:
            for file_info in tqdm(files_to_render):
                rendering_results.append(render_image_no_gt(file_info,
                                                            detection_categories_to_results_name,
                                                            detection_categories,
                                                            classification_categories,
                                                            options=options))
                
        elapsed = time.time() - start_time

        # Map all the rendering results in the list rendering_results into the
        # dictionary images_html
        image_rendered_count = 0
        for rendering_result in rendering_results:
            if rendering_result is None:
                continue
            image_rendered_count += 1
            for assignment in rendering_result:
                if 'class' in assignment[0]:
                    has_classification_info = True
                images_html[assignment[0]].append(assignment[1])

        # Prepare the individual html image files
        image_counts = prepare_html_subpages(images_html, output_dir)

        if image_rendered_count == 0:
            seconds_per_image = 0.0
        else:
            seconds_per_image = elapsed/image_rendered_count

        print('Rendered {} images (of {}) in {} ({} per image)'.format(image_rendered_count,
              image_count,humanfriendly.format_timespan(elapsed),
              humanfriendly.format_timespan(seconds_per_image)))

        # Write index.html

        # We can't just sum these, because image_counts includes images in both their
        # detection and classification classes
        # total_images = sum(image_counts.values())
        total_images = 0
        for k in image_counts.keys():
            v = image_counts[k]
            if has_classification_info and k.startswith('class_'):
                continue
            total_images += v

        if total_images != image_count:
            print('Warning, missing images: image_count is {}, total_images is {}'.format(total_images,image_count))
        
        almost_detection_string = ''
        if options.include_almost_detections:
            almost_detection_string = ' (&ldquo;almost detection&rdquo; threshold at {:.1%})'.format(
                options.almost_detection_confidence_threshold)

        index_page = """<html>\n{}\n<body>\n
        <h2>Visualization of results for {}</h2>\n
        <p>A sample of {} images (of {} total)FAILURE_PLACEHOLDER, annotated with detections above {:.1%} confidence{}.</p>\n
        
        <div class="contentdiv">
        <p>Model version: {}</p>
        </div>
        
        <h3>Sample images</h3>\n
        <div class="contentdiv">\n""".format(
            style_header, job_name_string, image_count, len(detections_df), options.confidence_threshold,
            almost_detection_string, model_version_string)

        failure_string = ''
        if n_failures is not None:
            failure_string = ' ({} failures)'.format(n_failures)        
        index_page = index_page.replace('FAILURE_PLACEHOLDER',failure_string)
        
        def result_set_name_to_friendly_name(result_set_name):
            friendly_name = ''
            friendly_name = result_set_name.replace('_','-')
            if friendly_name.startswith('detections-'):
                friendly_name = friendly_name.replace('detections-', 'detections: ')
            friendly_name = friendly_name.capitalize()
            return friendly_name

        for result_set_name in images_html.keys():

            # Don't print classification classes here; we'll do that later with a slightly
            # different structure
            if has_classification_info and result_set_name.lower().startswith('class_'):
                continue

            filename = result_set_name + '.html'
            label = result_set_name_to_friendly_name(result_set_name)
            image_count = image_counts[result_set_name]
            if total_images == 0:
                image_fraction = -1
            else:
                image_fraction = image_count / total_images
            index_page += '<a href="{}">{}</a> ({}, {:.1%})<br/>\n'.format(
                filename,label,image_count,image_fraction)

        index_page += '</div>\n'

        if has_classification_info:
            index_page += '<h3>Images of detected classes</h3>'
            index_page += '<p>The same image might appear under multiple classes if multiple species were detected.</p>\n'
            index_page += '<p>Classifications with confidence less than {:.1%} confidence are considered "unreliable".</p>\n'.format(
                options.classification_confidence_threshold)
            index_page += '<div class="contentdiv">\n'

            # Add links to all available classes
            class_names = sorted(classification_categories.values())
            if 'class_unreliable' in images_html.keys():
                class_names.append('unreliable')

            if options.sort_classification_results_by_count:
                class_name_to_count = {}                
                for cname in class_names:
                    ccount = len(images_html['class_{}'.format(cname)])
                    class_name_to_count[cname] = ccount
                class_names = sorted(class_names,key=lambda x: class_name_to_count[x],reverse=True)                
                    
            for cname in class_names:
                ccount = len(images_html['class_{}'.format(cname)])
                if ccount > 0:
                    index_page += '<a href="class_{}.html">{}</a> ({})<br/>\n'.format(
                        cname, cname.lower(), ccount)
            index_page += '</div>\n'

        index_page += '</body></html>'
        output_html_file = os.path.join(output_dir, 'index.html')
        with open(output_html_file, 'w') as f:
            f.write(index_page)

        print('Finished writing html to {}'.format(output_html_file))

        # os.startfile(output_html_file)

    # ...if we do/don't have ground truth

    ppresults.output_html_file = output_html_file
    return ppresults

# ...process_batch_results


#%% Interactive driver(s)

if False:

    #%%

    base_dir = r'G:\temp\md'
    options = PostProcessingOptions()
    options.image_base_dir = base_dir
    options.output_dir = os.path.join(base_dir, 'postprocessing')
    options.api_output_filename_replacements = {} # {'20190430cameratraps\\':''}
    options.ground_truth_filename_replacements = {} # {'\\data\\blob\\':''}
    options.api_output_file = os.path.join(base_dir, 'results.json')
    options.ground_truth_json_file = os.path.join(base_dir, 'gt.json')
    # options.unlabeled_classes = ['human']

    ppresults = process_batch_results(options)
    # os.start(ppresults.output_html_file)


#%% Command-line driver

def main():
    
    options = PostProcessingOptions()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'api_output_file',
        help='path to .json file produced by the batch inference API')
    parser.add_argument(
        'output_dir',
        help='base directory for output')
    parser.add_argument(
        '--image_base_dir', default=options.image_base_dir,
        help='base directory for images (optional, can compute statistics '
             'without images)')
    parser.add_argument(
        '--ground_truth_json_file', default=options.ground_truth_json_file,
        help='ground truth labels (optional, can render detections without '
             'ground truth), in the COCO Camera Traps format')
    parser.add_argument(
        '--confidence_threshold', type=float,
        default=options.confidence_threshold,
        help='Confidence threshold for statistics and visualization')
    parser.add_argument(
        '--almost_detection_confidence_threshold', type=float,
        default=options.almost_detection_confidence_threshold,
        help='Almost-detection confidence threshold for statistics and visualization')
    parser.add_argument(
        '--target_recall', type=float, default=options.target_recall,
        help='Target recall (for statistics only)')
    parser.add_argument(
        '--num_images_to_sample', type=int,
        default=options.num_images_to_sample,
        help='number of images to visualize, -1 for all images (default: 500)')
    parser.add_argument(
        '--viz_target_width', type=int, default=options.viz_target_width,
        help='Output image width')
    parser.add_argument(
        '--include_almost_detections', action='store_true',
        help='Include a separate category for images just above a second confidence threshold')
    parser.add_argument(
        '--random_output_sort', action='store_true',
        help='Sort output randomly (defaults to sorting by filename)')
    parser.add_argument(
        '--n_cores', type=int, default=1,
        help='Number of threads to use for rendering (default: 1)')
    parser.add_argument(
        '--parallelize_rendering_with_processes', 
        action='store_true',
        help='Should we use processes (instead of threads) for parallelization?')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    args.sort_html_by_filename = (not args.random_output_sort)
    if args.n_cores != 1:
        assert (args.n_cores > 1), 'Illegal number of cores: {}'.format(args.n_cores)
        if args.parallelize_rendering_with_processes:
            args.parallelize_rendering_with_threads = False
        args.parallelize_rendering = True
        args.parallelize_rendering_n_cores = args.n_cores        

    args_to_object(args, options)
    
    process_batch_results(options)


if __name__ == '__main__':
    main()

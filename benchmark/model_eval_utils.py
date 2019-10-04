import json
from random import sample
import os

from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score

from api.batch_processing.postprocessing import load_api_results
from data_management.cct_json_utils import CameraTrapJsonUtils
from visualization import visualization_utils


#%% Empty and non-empty classification at image level

def empty_accuracy_image_level(gt_db_indexed, detection_res, threshold=0.5):
    gt = []
    pred = []

    for image_id, annotations in gt_db_indexed.image_id_to_annotations.items():
        max_det_score = detection_res[image_id]['max_detection_conf']
        pred_class = 0 if max_det_score < threshold else 1

        pred.append(pred_class)

        if len(annotations) > 0:
            gt_score = 0
            for a in annotations:
                if 'bbox' in a:
                    gt_score = 1  # not empty
                    break
            gt.append(gt_score)
        else:
            gt.append(0)  # empty
    accuracy = accuracy_score(gt, pred)
    return accuracy


def empty_precision_recall_image_level(gt_db_indexed, detection_res):
    """
    For empty/non-empty classification based on max_detection_conf in detection entries.
    Args:
        gt_db_indexed: IndexedJsonDb of the ground truth bbox json.
        detection_res: dict of image_id to image entry in the API output file's `images` field. The key needs to be
        the same image_id as those in the ground truth json db.

    Returns:
        precisions, recalls, thresholds (confidence levels)
    """
    gt = []
    pred = []

    for image_id, annotations in gt_db_indexed.image_id_to_annotations.items():
        det_image_obj = detection_res[image_id]

        max_det_score = det_image_obj['max_detection_conf']
        pred.append(max_det_score)

        if len(annotations) > 0:
            gt_score = 0
            for a in annotations:
                if 'bbox' in a:
                    gt_score = 1  # not empty
                    break
            gt.append(gt_score)
        else:
            gt.append(0)  # empty

    print('Length of gt and pred:', len(gt), len(pred))
    precisions, recalls, thresholds = precision_recall_curve(gt, pred)
    average_precision = average_precision_score(gt, pred)
    return precisions, recalls, thresholds, average_precision


#%% Empty and non-empty classification at sequence level

def is_gt_seq_non_empty(annotations, empty_category_id):
    """
    True if there are animals etc, False if empty.
    """
    category_on_images = set()
    for a in annotations:
        category_on_images.add(a['category_id'])
    if len(category_on_images) > 1:
        return True
    elif len(category_on_images) == 1:
        only_cat = list(category_on_images)[0]
        if only_cat == empty_category_id:
            return False
        else:
            return True
    else:
        raise Exception('No category information in annotation entry.')


def pred_seq_max_conf(detector_output_images_entries):
    """
    Surface the max_detection_conf field, include detections of all classes.
    """
    return max([entry['max_detection_conf'] for entry in detector_output_images_entries])


def get_number_empty_seq(gt_db_indexed):
    gt_seq_id_to_annotations = CameraTrapJsonUtils.annotations_groupby_image_field(gt_db_indexed, image_field='seq_id')
    empty_category_id_in_gt = gt_db_indexed.cat_name_to_id['empty']
    gt_seq_level = []
    for seq_id, seq_annotations in gt_seq_id_to_annotations.items():
        gt_seq_level.append(is_gt_seq_non_empty(seq_annotations, empty_category_id_in_gt))

    total = len(gt_seq_level)
    num_empty = total - sum(gt_seq_level)
    print('There are {} sequences, {} are empty, which is {}%'.format(total, num_empty, 100 * num_empty / total))


def empty_accuracy_seq_level(gt_db_indexed, detector_output_path, file_to_image_id,
                             threshold=0.5, visualize_wrongly_classified=False, images_dir=''):
    """ Ground truth label is empty if the fine-category label on all images in this sequence are "empty"

    Args:
        gt_db_indexed: an instance of IndexedJsonDb containing the ground truth
        detector_output_path: path to a file containing the detection results in the API output format
        file_to_image_id: see load_api_results.py - a function to convert image_id
        threshold: threshold between 0 and 1 below which an image is considered empty
        visualize_wrongly_classified: True if want to visualize 5 sequences where the predicted
            classes don't agree with gt
        images_dir: directory where the 'file' field in the detector output is rooted at. Relevant only if
           visualize_wrongly_classified is true
    Returns:

    """
    # TODO move detector_output_path specific code out so that this function evaluates only on classification results (confidences)
    gt_seq_id_to_annotations = CameraTrapJsonUtils.annotations_groupby_image_field(gt_db_indexed, image_field='seq_id')
    pred_seq_id_to_res = load_api_results.api_results_groupby(detector_output_path, gt_db_indexed,
                                                              file_to_image_id)

    gt_seq_level = []
    pred_seq_level = []

    empty_category_id_in_gt = gt_db_indexed.cat_name_to_id['empty']

    # evaluate on sequences that are present in both gt and the detector output file
    gt_sequences = set(gt_seq_id_to_annotations.keys())
    pred_sequences = set(pred_seq_id_to_res.keys())

    diff = gt_sequences.symmetric_difference(pred_sequences)
    print('Number of sequences not in both gt and pred: {}'.format(len(diff)))

    intersection_sequences = list(gt_sequences.intersection(pred_sequences))

    for seq_id in intersection_sequences:
        gt_seq_level.append(is_gt_seq_non_empty(gt_seq_id_to_annotations[seq_id], empty_category_id_in_gt))
        pred_seq_level.append(pred_seq_max_conf(pred_seq_id_to_res[seq_id]))

    pred_class = [0 if max_conf < threshold else 1 for max_conf in pred_seq_level]
    accuracy = accuracy_score(gt_seq_level, pred_class)

    if visualize_wrongly_classified:
        show_wrongly_classified_seq(pred_seq_id_to_res, intersection_sequences, gt_seq_level, pred_class, images_dir)

    return accuracy, gt_seq_level, pred_seq_level, intersection_sequences


def show_wrongly_classified_seq(pred_seq_id_to_res, seq_ids, gt_seq_level, pred_binary_seq_level, images_dir):
    wrongly_classified_seqs = []
    for seq_id, gt, pred in zip(seq_ids, gt_seq_level, pred_binary_seq_level):
        if gt != pred:
            wrongly_classified_seqs.append((seq_id, gt, pred))

    num_to_sample = 5
    sampled = sample(wrongly_classified_seqs, num_to_sample)

    for seq_id, gt, pred in sampled:
        print('Ground truth is {}, predicted class is {}, seq_id {}.'.format(gt, pred, seq_id))
        predicted_res = pred_seq_id_to_res[seq_id]
        predicted_res_files = [os.path.join(images_dir, item['file']) for item in predicted_res]

        fig = visualization_utils.show_images_in_a_row(predicted_res_files)


#%% Utilities

def find_precision_at_recall(precision, recall, thresholds, recall_level=0.9):
    """ Returns the precision at a specified level of recall. The thresholds should go from 0 to 1.0

    Args:
        precision: List of precisions for each confidence
        recall: List of recalls for each confidence, paired with items in the `precision` list
        recall_level: A float between 0 and 1.0, the level of recall to retrieve precision at.

    Returns:
        precision at the specified recall_level
    """

    if precision is None or recall is None:
        print('precision or recall is None')
        return 0.0, 0.0

    for p, r, t in zip(precision, recall, thresholds):
        if r < recall_level:
            return p, t


def make_detection_res(results_path, file_prefix=''):
    """ Make the detection result into a dictionary of file : result entry

    Args:
        results_path: path to output of API containing the detection results
        file_prefix: this prefix will be taken out of the result entry 'file' field to be consistent
            with the 'file_name' field in a CCT formatted json DB.
    Returns:
        A dictionary of file : API result entry
    """
    with open(results_path) as f:
        res = json.load(f)

    detection_res = {}

    for i in res['images']:
        file_name = i['file'].split(file_prefix)[1].split('.jpg')[0].split('.JPG')[0]
        detection_res[file_name] = i  # all detections on that image is in this dict
    return detection_res


def get_gt_db(gt_db_path):
    """ Load the CCT formatted DB and index it.

    Args:
        gt_db_path: path to the json DB.

    Returns:
       An IndexedJsonDb object
    """
    with open(gt_db_path) as f:
        gt_db = json.load(f)
    gt_indexed = cct_json_utils.IndexedJsonDb(gt_db)
    return gt_indexed




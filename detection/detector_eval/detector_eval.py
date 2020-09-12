"""
Functions for evaluating detectors on detection tasks.

See benchmark/model_eval_utils.py for functions to evaluate on more generic
tasks.

TF Object Detection API needs to be installed.
"""
from collections import defaultdict
import math
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
from object_detection.utils import per_image_evaluation, metrics  # TFODAPI
from tqdm import tqdm

from ct_utils import convert_xywh_to_tf


def compute_precision_recall_bbox(
        per_image_detections: Mapping[str, Mapping[str, Any]],
        per_image_gts: Mapping[str, Mapping[str, Any]],
        num_gt_classes: int,
        matching_iou_threshold: float = 0.5
        ) -> Dict[Union[str, int], Dict[str, Any]]:
    """
    Compute the precision and recall at each confidence level for detection
    results of various classes.

    Args:
        per_image_detections: dict, image_id (str) => dict with fields
            'boxes': array-like, shape [N, 4], type float, each row is
                [ymin, xmin, ymax, xmax] in normalized coordinates
            'scores': array-like, shape [N], float
            'labels': array-like, shape [N], integers in [1, num_gt_classes]
        per_image_gts: dic, image_id (str) => dict with fields
            'gt_boxes': array-like, shape [M, 4], type float, each row is
                [ymin, xmin, ymax, xmax] in normalized coordinates
            'gt_labels': array-like, shape [M], integers in [1, num_gt_classes]
        num_gt_classes: int, number of classes in the ground truth labels
        matching_iou_threshold: float, IoU above which a detected and a ground
            truth box are considered overlapping

    Returns: dict, per-class metrics, keys are integers in [1, num_gt_classes]
        and 'one_class' which considers all classes. Each value is a dict with
        fields ['precision', 'recall', 'average_precision', ...]
    """
    per_image_eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes=num_gt_classes,
        matching_iou_threshold=matching_iou_threshold,
        nms_iou_threshold=1.0,
        nms_max_output_boxes=10000)

    print('Running per-object analysis...', flush=True)

    # keys are categories (int)
    detection_tp_fp = defaultdict(list)  # in each list, 1 is tp, 0 is fp
    detection_scores = defaultdict(list)
    num_total_gt: Dict[int, int] = defaultdict(int)

    for image_id, dets in tqdm(per_image_detections.items()):
        # we force *_boxes to have shape [N, 4], even in case that N = 0
        detected_boxes = np.asarray(
            dets['boxes'], dtype=np.float32).reshape(-1, 4)
        detected_scores = np.asarray(dets['scores'])
        # labels input to compute_object_detection_metrics() needs to start at 0, not 1
        detected_labels = np.asarray(dets['labels'], dtype=np.int) - 1  # start at 0
        # num_detections = len(dets['boxes'])

        gts = per_image_gts[image_id]
        gt_boxes = np.asarray(gts['gt_boxes'], dtype=np.float32).reshape(-1, 4)
        gt_labels = np.asarray(gts['gt_labels'], dtype=np.int) - 1  # start at 0
        num_gts = len(gts['gt_boxes'])

        # place holders - we don't have these
        groundtruth_is_difficult_list = np.zeros(num_gts, dtype=bool)
        groundtruth_is_group_of_list = np.zeros(num_gts, dtype=bool)

        results = per_image_eval.compute_object_detection_metrics(
            detected_boxes=detected_boxes,
            detected_scores=detected_scores,
            detected_class_labels=detected_labels,
            groundtruth_boxes=gt_boxes,
            groundtruth_class_labels=gt_labels,
            groundtruth_is_difficult_list=groundtruth_is_difficult_list,
            groundtruth_is_group_of_list=groundtruth_is_group_of_list)
        scores, tp_fp_labels, is_class_correctly_detected_in_image = results

        for i, tp_fp_labels_cat in enumerate(tp_fp_labels):
            # true positives < gt of that category
            assert sum(tp_fp_labels_cat) <= sum(gt_labels == i)

            cat = i + 1  # categories start at 1
            detection_tp_fp[cat].append(tp_fp_labels_cat)
            detection_scores[cat].append(scores[i])
            num_total_gt[cat] += sum(gt_labels == i)  # gt_labels start at 0

    all_scores = []
    all_tp_fp = []

    print('Computing precision recall for each category...')
    per_cat_metrics: Dict[Union[int, str], Dict[str, Any]] = {}
    for i in range(num_gt_classes):
        cat = i + 1
        scores_cat = np.concatenate(detection_scores[cat])
        tp_fp_cat = np.concatenate(detection_tp_fp[cat]).astype(np.bool)
        all_scores.append(scores_cat)
        all_tp_fp.append(tp_fp_cat)

        precision, recall = metrics.compute_precision_recall(
            scores_cat, tp_fp_cat, num_total_gt[cat])
        average_precision = metrics.compute_average_precision(precision, recall)

        per_cat_metrics[cat] = {
            'category': cat,
            'precision': precision,
            'recall': recall,
            'average_precision': average_precision,
            'scores': scores_cat,
            'tp_fp': tp_fp_cat,
            'num_gt': num_total_gt[cat]
        }
        print(f'Number of ground truth in category {cat}: {num_total_gt[cat]}')

    # compute one-class precision/recall/average precision (if every box is just
    # of an object class)
    all_scores = np.concatenate(all_scores)
    all_tp_fp = np.concatenate(all_tp_fp)
    overall_gt_count = sum(num_total_gt.values())

    one_class_prec, one_class_recall = metrics.compute_precision_recall(
        all_scores, all_tp_fp, overall_gt_count)
    one_class_average_precision = metrics.compute_average_precision(
        one_class_prec, one_class_recall)

    per_cat_metrics['one_class'] = {
        'category': 'one_class',
        'precision': one_class_prec,
        'recall': one_class_recall,
        'average_precision': one_class_average_precision,
        'scores': all_scores,
        'tp_fp': all_tp_fp,
        'num_gt': overall_gt_count
    }

    return per_cat_metrics


def get_per_image_gts_and_detections(
        gt_db_dict: Mapping[str, Mapping[str, Any]],
        detection_res: Mapping[str, Mapping[str, Any]],
        label_map_name_to_id: Mapping[str, int]
        ) -> Tuple[
            Dict[str, Dict[str, Any]],
            Dict[str, Dict[str, Any]]
        ]:
    """Group the detected and ground truth bounding boxes by image_id.
    For use when the gt_db_dict is from MegaDB.

    Args:
        gt_db_dict: dict, ground truth bbox JSON, usually query result from
            MegaDB, maps image_id => dict with key 'bbox'
        detection_res: dict, image_id => detections dict, each key must also
            appear in gt_db_dict. Each detection dict is an entry in the Batch
            API output file's `images` field.
        label_map_name_to_id: dict, e.g. 'animal': 1

    Returns:
        per_image_gts: dict, image_id (str) => dict with keys
            'gt_boxes': list of list, inner lists are [ymin, xmin, ymax, xmax]
                in normalized coordinates
            'gt_labels':  list of int
        per_image_detections: dict, image_id (str) => dict with keys
            'boxes': list of list, inner lists are [ymin, xmin, ymax, xmax]
                in normalized coordinates
            'scores': list of float, detection confidences
            'labels': list of int
    """
    per_image_gts = {}
    per_image_detections = {}

    for image_id, det_image_obj in detection_res.items():
        gt_boxes = []
        gt_labels = []

        gt_entry = gt_db_dict[image_id]
        for b in gt_entry['bbox']:
            if b['category'] not in label_map_name_to_id:
                continue
            gt_boxes.append(convert_xywh_to_tf(b['bbox']))
            gt_labels.append(label_map_name_to_id[b['category']])

        per_image_gts[image_id] = {
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels
        }

        detection_boxes = []
        detection_scores = []
        detection_labels = []

        for det in det_image_obj['detections']:
            detection_boxes.append(convert_xywh_to_tf(det['bbox']))
            detection_scores.append(det['conf'])
            detection_labels.append(int(det['category']))

        per_image_detections[image_id] = {
            'boxes': detection_boxes,
            'scores': detection_scores,
            'labels': detection_labels
        }
    return per_image_gts, per_image_detections


def get_per_image_gts_and_detections_deprecated(gt_db_indexed, detection_res):
    """
    Deprecated - used when ground truth labels are in a CCT JSON DB.
    """
    per_image_gts = {}
    per_image_detections = {}

    # iterate through each image in the gt file, not the detection file

    for image_id, annotations in gt_db_indexed.image_id_to_annotations.items():
        # ground truth
        image_obj = gt_db_indexed.image_id_to_image[image_id]
        im_h, im_w = image_obj['height'], image_obj['width']

        gt_boxes = []
        gt_labels = []

        for gt_anno in annotations:
            # convert gt box coordinates to TFODAPI format
            gt_box_x, gt_box_y, gt_box_w, gt_box_h = gt_anno['bbox']
            gt_y_min, gt_x_min = gt_box_y / im_h, gt_box_x / im_w
            gt_y_max = (gt_box_y + gt_box_h) / im_h
            gt_x_max = (gt_box_x + gt_box_w) / im_w
            gt_boxes.append([gt_y_min, gt_x_min, gt_y_max, gt_x_max])

            gt_labels.append(gt_anno['category_id'])

        per_image_gts[image_id] = {
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels
        }

        # detections
        det_image_obj = detection_res[image_id]

        detection_boxes = []
        detection_scores = []
        detection_labels = []

        for det in det_image_obj['detections']:
            x_min, y_min, width_of_box, height_of_box = det['bbox']
            y_max = y_min + height_of_box
            x_max = x_min + width_of_box
            detection_boxes.append([y_min, x_min, y_max, x_max])

            detection_scores.append(det['conf'])
            detection_labels.append(int(det['category']))

        # only include a detection entry if that image had detections
        if len(detection_boxes) > 0:
            per_image_detections[image_id] = {
                'boxes': detection_boxes,
                'scores': detection_scores,
                'labels': detection_labels
            }

    return per_image_gts, per_image_detections


def find_mAP(per_cat_metrics: Mapping[Union[str, int], Mapping[str, Any]]
             ) -> float:
    """
    Mean average precision, the mean of the average precision for each category

    Args:
        per_cat_metrics: dict, result of compute_precision_recall()

    Returns: float, mAP for this set of detection results
    """
    # minus the 'one_class' set of metrics
    num_gt_classes = len(per_cat_metrics) - 1

    mAP_from_cats = sum(
        v['average_precision']
        if k != 'one_class' and not math.isnan(v['average_precision']) else 0
        for k, v in per_cat_metrics.items()
    ) / num_gt_classes

    return mAP_from_cats

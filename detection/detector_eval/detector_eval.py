from collections import defaultdict
import math

import numpy as np
from object_detection.utils import per_image_evaluation, metrics
from tqdm import tqdm


def compute_precision_recall(per_image_detections, per_image_gts, num_gt_classes,
                             matching_iou_threshold=0.5):
    per_image_eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes=num_gt_classes,
        matching_iou_threshold=matching_iou_threshold,
        nms_iou_threshold=1.0,
        nms_max_output_boxes=10000
    )

    print('Running per-object analysis...')

    detection_tp_fp = defaultdict(list)  # key is the category; in each list, 1 is tp, 0 is fp
    detection_scores = defaultdict(list)
    num_total_gt = defaultdict(int)

    for image_id, dets in tqdm(per_image_detections.items()):

        detected_boxes = np.array(dets['boxes'], dtype=np.float32)
        detected_scores = np.array(dets['scores'], dtype=np.float32)
        # labels input to compute_object_detection_metrics() needs to start at 0, not 1
        detected_labels = np.array(dets['labels'], dtype=np.int) - 1  # start at 0
        # num_detections = len(dets['boxes'])

        gts = per_image_gts[image_id]
        gt_boxes = np.array(gts['gt_boxes'], dtype=np.float32)
        gt_labels = np.array(gts['gt_labels'], dtype=np.int) - 1  # start at 0
        num_gts = len(gts['gt_boxes'])

        groundtruth_is_difficult_list = np.zeros(num_gts, dtype=bool)  # place holders - we don't have these
        groundtruth_is_group_of_list = np.zeros(num_gts, dtype=bool)

        # to prevent 'Invalid dimensions for box data.' error
        if num_gts == 0:
            # this box will not match any detections
            gt_boxes = np.array([[0, 0, 0, 0]], dtype=np.float32)

        scores, tp_fp_labels, is_class_correctly_detected_in_image = (
            per_image_eval.compute_object_detection_metrics(
                detected_boxes=detected_boxes,
                detected_scores=detected_scores,
                detected_class_labels=detected_labels,
                groundtruth_boxes=gt_boxes,
                groundtruth_class_labels=gt_labels,
                groundtruth_is_difficult_list=groundtruth_is_difficult_list,
                groundtruth_is_group_of_list=groundtruth_is_group_of_list
            )
        )

        for i, tp_fp_labels_cat in enumerate(tp_fp_labels):
            assert sum(tp_fp_labels_cat) <= sum(gt_labels == i)  # true positives < gt of that category
            cat = i + 1  # categories start at 1
            detection_tp_fp[cat].append(tp_fp_labels_cat)
            detection_scores[cat].append(scores[i])
            num_total_gt[cat] += sum(gt_labels == i)  # gt_labels start at 0

    all_scores = []
    all_tp_fp = []

    print('Computing precision recall for each category...')
    per_cat_metrics = {}
    for i in range(num_gt_classes):
        cat = i + 1
        scores_cat = np.concatenate(detection_scores[cat])
        tp_fp_cat = np.concatenate(detection_tp_fp[cat]).astype(np.bool)
        all_scores.append(scores_cat)
        all_tp_fp.append(tp_fp_cat)

        precision, recall = metrics.compute_precision_recall(
            scores_cat, tp_fp_cat, num_total_gt[cat]
        )
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
        print('Number of ground truth in category {} is {}'.format(cat, num_total_gt[cat]))

    # compute one-class precision/recall/average precision (if every box is just of an object class)
    all_scores = np.concatenate(all_scores)
    all_tp_fp = np.concatenate(all_tp_fp)
    overall_gt_count = sum(num_total_gt.values())

    one_class_prec, one_class_recall = metrics.compute_precision_recall(
        all_scores, all_tp_fp, overall_gt_count
    )
    one_class_average_precision = metrics.compute_average_precision(one_class_prec, one_class_recall)

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


def find_mAP(per_cat_metrics):
    """
    Mean average precision, the mean of the average precision for each category
    Args:
        per_cat_metrics: result of compute_precision_recall()

    Returns:
        The mAP for this set of detection results
    """
    num_gt_classes = len(per_cat_metrics) - 1  # minus the 'one_class' set of metrics

    mAP_from_cats = sum([v['average_precision'] if k != 'one_class' and not math.isnan(v['average_precision']) else 0
                         for k, v in per_cat_metrics.items()]) / num_gt_classes

    return mAP_from_cats


def find_precision_at_recall(precision, recall, recall_level=0.9):
    """ Returns the precision at a specified level of recall.

    Args:
        precision: List of precisions for each confidence
        recall: List of recalls for each confidence, paired with items in the `precision` list
        recall_level: A float between 0 and 1.0, the level of recall to retrieve precision at.

    Returns:
        precision at the specified recall_level
    """

    if precision is None or recall is None:
        print('Returning 0')
        return 0.0

    for p, r in zip(precision, recall):
        if r is None or r < recall_level:
            continue
        return p

    return 0.0  # recall level never reaches recall_level specified



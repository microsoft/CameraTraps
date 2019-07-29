#
# evaluate_detections.py
#
# Adapted from analyze_detection.py which is now archived.
#

#%% Imports and constants

import argparse
import math
import pickle
import sys
from collections import defaultdict
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from object_detection.utils import per_image_evaluation, metrics

import detection_eval_utils as utils

PLOT_WIDTH = 5  # in inches


#%% Functions

def _compute_precision_recall(per_image_detections, per_image_gts, num_gt_classes):
    
    per_image_eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes=num_gt_classes,
        matching_iou_threshold=0.5,
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
    

def compute_precision_recall(results, exclude_person_images):
    num_images = len(results['gt_labels'])

    gt_labels_flat = []
    for gt_labels_im in results['gt_labels']:
        gt_labels_flat.extend(gt_labels_im)

    num_gt_classes = len(set(gt_labels_flat))

    print('Number of images to be evaluated: {}'.format(num_images))
    print('Number of gt bounding boxes: {}'.format(len(gt_labels_flat)))
    print('Number of gt classes: {}'.format(num_gt_classes))

    use_im = utils.get_non_person_images(results) if exclude_person_images else None

    per_image_detections, per_image_gts = utils.group_detections_by_image(results, use_im)
    print('Length of per_image_detections:', len(per_image_detections))

    per_cat_metrics = _compute_precision_recall(per_image_detections, per_image_gts, num_gt_classes)

    mAP_from_cats = sum([v['average_precision'] if k != 'one_class' and not math.isnan(v['average_precision']) else 0
                         for k, v in per_cat_metrics.items()]) / num_gt_classes
    print('mAP as the average of AP across the {} categories is {:.4f}'.format(num_gt_classes, mAP_from_cats))

    fig = plt.figure(figsize=(PLOT_WIDTH + 0.2, PLOT_WIDTH * len(per_cat_metrics)))  # (width, height) in inches

    per_cat_metrics = sorted(per_cat_metrics.items(), key=lambda x: abs(hash(x[0])))  #  cast 'one_class' as int

    for i, (cat, cat_metrics) in enumerate(per_cat_metrics):
        ax = fig.add_subplot(num_gt_classes + 1, 1, i + 1)  # nrows, ncols, and index
        ax.set_aspect('equal')
        ax.set_title('Category {}, AP {:.4f}'.format(cat, cat_metrics['average_precision']))
        utils.plot_precision_recall(ax, cat_metrics['precision'], cat_metrics['recall'])

    fig.tight_layout()

    return per_cat_metrics, fig


#%% Command-line driver
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('detection_file', action='store', type=str,
                        help='.p file containing detection results')
    parser.add_argument('out_dir', action='store', type=str,
                        help='path to directory where outputs will be stored (an image and a pickle file)')
    parser.add_argument('--exclude_person_images', action='store_true', default=False,
                        help='This flag causes evaluation to not look at any image with persons')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    args = parser.parse_args()

    print('Flag args.exclude_person_images:', args.exclude_person_images)

    os.makedirs(args.out_dir, exist_ok=True)
    p = pickle.load(open(args.detection_file, 'rb'))

    per_cat_metrics, fig_precision_recall = compute_precision_recall(p, args.exclude_person_images)

    fig_precision_recall.savefig(os.path.join(args.out_dir, 'precicision_recall.png'))
    pickle.dump(per_cat_metrics, open(os.path.join(args.out_dir, 'per_category_metrics.p'), 'wb'))


if __name__ == '__main__':
    main()

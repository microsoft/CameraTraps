#
# detection_eval_utils.py
#
# Adapted from detection/eval/utils.py which is now archived.
#

def group_detections_by_image(detection_results, use_im=None):

    if use_im == None:
        use_im = {im :True for im in detection_results['images']}

    per_image_detections = { detection_results['images'][idx] :{'boxes': detection_results['detection_boxes'][idx],
                                                               'scores': detection_results['detection_scores'][idx],
                                                               'labels': detection_results['detection_labels'][idx]} for
                            idx in range(len(detection_results['images'])) if use_im[detection_results['images'][idx]]}

    # group the ground truth annotations by image id
    per_image_gts = {detection_results['images'][idx]: {'gt_boxes': detection_results['gt_boxes'][idx],
                                                        'gt_labels': detection_results['gt_labels'][idx]} for idx in
                     range(len(detection_results['images'])) if use_im[detection_results['images'][idx]]}

    return per_image_detections, per_image_gts


def plot_precision_recall(ax, cat, precision, recall, average_precision):
    ax.set_title('Category {}, AP {:.4f}'.format(cat, average_precision))
    ax.plot(recall, precision)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')


def find_precision_at_recall(precision, recall, recall_level=0.9):
    """ Returns the precision at a specified level of recall.

    Args:
        precision: Fraction of positive instances over detected ones.
        recall: Fraction of detected positive instance over all positive instances.
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


#
# detection_eval_utils.py
#
# Utility functions used in evaluate_detections.py
#

def get_non_person_images(results):
    
    num_boxes_excluded = 0
    use_im = {im: True for im in results['images']}
    for image_id, gt_labels in zip(results['images'], results['gt_labels']):
        for l in gt_labels:
            if int(l) == 2:
                use_im[image_id] = False
                num_boxes_excluded += 1

    num_images_excluded = sum([0 if include else 1 for im, include in use_im.items()])
    print('Number of bbox of person excluded: {}, images excluded: {}'.format(num_boxes_excluded, num_images_excluded))
    return use_im


def group_detections_by_image(results, use_im=None):
    
    if use_im == None:
        use_im = {im: True for im in results['images']}

    per_image_detections = {results['images'][idx]: {'boxes': results['detection_boxes'][idx],
                                                     'scores': results['detection_scores'][idx],
                                                     'labels': results['detection_labels'][idx]} for
                            idx in range(len(results['images'])) if use_im[results['images'][idx]]}

    # group the ground truth annotations by image id
    per_image_gts = {results['images'][idx]: {'gt_boxes': results['gt_boxes'][idx],
                                              'gt_labels': results['gt_labels'][idx]} for idx in
                     range(len(results['images'])) if use_im[results['images'][idx]]}

    return per_image_detections, per_image_gts


def plot_precision_recall(ax, precision, recall):
    if precision is None or recall is None:
        return

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

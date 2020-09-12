import numpy as np
import json


def get_im_to_seq_map(db_file):
    with open(db_file,'r') as f:
        data = json.load(f)
    im_to_seq = {}
    for im in data['images']:
        im_to_seq[im['id']] = im['seq_id']
    return im_to_seq

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def cluster_detections_by_image(detection_results, use_im=None):
    
    if use_im == None:
        use_im = {im:True for im in detection_results['images']}

    per_image_detections = {detection_results['images'][idx] :{'bboxes': detection_results['detections'][idx], 'scores': detection_results['detection_scores'][idx], 'labels':detection_results['detection_labels'][idx]} for idx in range(len(detection_results['images'])) if use_im[detection_results['images'][idx]]}

    # group the ground truth annotations by image id:
    per_image_gts ={detection_results['images'][idx] : {'bboxes':detection_results['gts'][idx],'labels':detection_results['gt_labels'][idx]} for idx in range(len(detection_results['images'])) if use_im[detection_results['images'][idx]]}

    return per_image_detections, per_image_gts

def get_results_per_image(dets, gts, per_image_eval):
    num_detections = len(dets['bboxes'])

    # [ymin, xmin, ymax, xmax] in absolute image coordinates.
    detected_boxes = np.zeros([num_detections, 4], dtype=np.float32)
    # detection scores for the boxes
    detected_scores = np.zeros([num_detections], dtype=np.float32)
    # 0-indexed detection classes for the boxes
    detected_class_labels = np.zeros([num_detections], dtype=np.int32)
    detected_masks = None

    for i in range(num_detections):
        x1, y1, x2, y2 = dets['bboxes'][i]
        detected_boxes[i] = np.array([y1, x1, y2, x2])
        detected_scores[i] = dets['scores'][i]
        detected_class_labels[i] = dets['labels'][i] - 1


       
    num_gts = len(gts['bboxes'])
    if num_gts > 0:

        # [ymin, xmin, ymax, xmax] in absolute image coordinates
        groundtruth_boxes = np.zeros([num_gts, 4], dtype=np.float32)
        # 0-indexed groundtruth classes for the boxes
        groundtruth_class_labels = np.zeros(num_gts, dtype=np.int32)
        groundtruth_masks = None
        groundtruth_is_difficult_list = np.zeros(num_gts, dtype=bool)
        groundtruth_is_group_of_list = np.zeros(num_gts, dtype=bool)


        for i in range(num_gts):
            x1, y1, x2, y2 = gts['bboxes'][i]
            groundtruth_boxes[i] = np.array([y1, x1, y2, x2])
            groundtruth_class_labels[i] = gts['labels'][i] - 1


        scores, tp_fp_labels, is_class_correctly_detected_in_image = (
            per_image_eval.compute_object_detection_metrics(
                    detected_boxes=detected_boxes,
                    detected_scores=detected_scores,
                    detected_class_labels=detected_class_labels,
                    groundtruth_boxes=groundtruth_boxes,
                    groundtruth_class_labels=groundtruth_class_labels,
                    groundtruth_is_difficult_list=groundtruth_is_difficult_list,
                    groundtruth_is_group_of_list=groundtruth_is_group_of_list,
                    detected_masks=detected_masks,
                    groundtruth_masks=groundtruth_masks
            )
        )

        scores = scores[0]
        tp_fp_labels = tp_fp_labels[0]

    else:
        tp_fp_labels = np.zeros(num_detections, dtype=np.int32)
        scores = detected_scores

    return scores, tp_fp_labels

def get_images_to_consider(detection_results, images_to_consider, get_night_day):
    if images_to_consider == 'all':
        use_im = {im:True for im in detection_results['images']}
    else:
        print('Getting '+images_to_consider+' images')
        use_im = {im:False for im in detection_results['images']}
        is_night_json = json.load(open(get_night_day,'r'))
        is_night = {im.split('.')[0]:is_night_json[im] for im in is_night_json} #convert to image id keys instead of filename
        if images_to_consider == 'night':
            for im in use_im:
                if im in is_night:
                    if is_night[im]:
                        use_im[im] = True
                else:
                    print(im)
        elif images_to_consider == 'day':
            for im in use_im:
                if im in is_night:
                    if not is_night[im]:
                        use_im[im] = True
                else:
                    print(im)
    
    return use_im

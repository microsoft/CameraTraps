import cPickle as  pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import json

#export PYTHONPATH=$PYTHONPATH:tfmodels/research
from object_detection.utils import metrics, np_box_ops
from object_detection.utils import per_image_evaluation

det_folder = '/ai4efs/models/object_detection/inception_resnet_v2_atrous/train_on_eccv_18_only/predictions/'

def compute_precision_recall_with_images(detection_file):
    
    print('Loading detection file...')
    
    with open(detection_file) as f:
        detection_results = pickle.load(f)
    
    print('Clustering detections by image...')

    # group the detections by image id:
    per_image_detections = {detection_results['images'][idx] :{'bboxes': detection_results['detections'][idx], 'scores': detection_results['detection_scores'][idx], 'labels':detection_results['detection_labels'][idx]} for idx in range(len(detection_results['images']))}

    # group the ground truth annotations by image id:
    per_image_gts ={detection_results['images'][idx] : {'bboxes':detection_results['gts'][idx],'labels':detection_results['gt_labels'][idx]} for idx in range(len(detection_results['images']))}

    per_image_eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes=1,
        matching_iou_threshold=0.5,
        nms_iou_threshold=1.0,
        nms_max_output_boxes=10000
    )
    
    print('Running per-image analysis...')

    detection_labels = []
    detection_scores = []
    num_total_gts = 0
    count = 0
    for image_id, dets in per_image_detections.iteritems():
        im_detection_labels = []
        im_detection_scores = []
        im_num_gts  = []
        max_im_scores = []
        
        
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

        max_im_scores.append(np.max(detected_scores))
        box_id = np.argmax(detected_scores)
            
        gts = per_image_gts[image_id]
        num_gts = len(gts['bboxes'])
        im_num_gts = num_gts
            
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

            ious = np_box_ops.iou(detected_boxes,groundtruth_boxes)
            if np.max(ious[box_id, :]) < 0.5:
                max_im_scores[-1] = 0
                
                #print('detected animal box')

            #print(groundtruth_boxes, groundtruth_class_labels,detected_scores[0],detected_boxes[0], detected_class_labels[0])

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
            #print(scores, tp_fp_labels)
            im_detection_labels = tp_fp_labels[0]
            im_detection_scores = scores[0]
            #num_total_gts += num_gts
            
            count +=1
            if count % 1000 == 0:
                print(str(count) + ' images complete') 

            #if (tp_fp_labels[0].shape[0] != num_detections):
            #    print('Incorrect label length')
            #if scores[0].shape[0] != num_detections:
            #    print('Incorrect score length')
            #if tp_fp_labels[0].sum() > num_gts:
            #    print('Too many correct detections')

        else:
            im_detection_labels = np.zeros(num_detections, dtype=np.int32)
            im_detection_scores = detected_scores
            max_im_scores[-1] = 0

        best_score = np.max(max_im_scores)
        if best_score > 0:
            #print('valid box')
            best_im = np.argmax(max_im_scores)
            #print(best_im, best_score)
            
                
            temp_labels = np.zeros(len(im_detection_labels),  dtype=np.int32)
            temp_scores = np.zeros(len(im_detection_scores), dtype=np.float32)
            for j in range(min(im_num_gts, len(im_detection_labels))):
                temp_labels[j] = True #TODO: this currently only works for oneclass?
                temp_scores[j] = best_score
            im_detection_labels = temp_labels
            im_detection_scores = temp_scores

        num_total_gts+=im_num_gts
        
        detection_labels.append(im_detection_labels)
        detection_scores.append(im_detection_scores)

    print(len(detection_scores), len(detection_scores[0]), len(detection_scores[1]))
    scores = np.concatenate(detection_scores)
    labels = np.concatenate(detection_labels).astype(np.bool)

    precision, recall = metrics.compute_precision_recall(
        scores, labels, num_total_gts
    )

    average_precision = metrics.compute_average_precision(precision, recall)
    
    
    return precision, recall, average_precision

if __name__ == '__main__':

    inter_prec, inter_recall, inter_ap = compute_precision_recall_with_images(det_folder + 'cis_test.p')
    loc_prec, loc_recall, loc_ap = compute_precision_recall_with_images(det_folder + 'trans_test.p')
    print('Cis mAP: ', inter_ap,', Trans mAP: ', loc_ap)
    recall_thresh = 0.9
    recall_idx = np.argmin([np.abs(x-recall_thresh) for x in inter_recall])
    print('Cis prec. at ',inter_recall[recall_idx],' recall: ', inter_prec[recall_idx])
    recall_idx = np.argmin([np.abs(x-recall_thresh) for x in loc_recall])
    print('Trans prec. at ',loc_recall[recall_idx],' recall: ', loc_prec[recall_idx])


    plt.figure("Precision Recall Curve - Images")
    plt.plot(inter_recall, inter_prec, 'C0-', label='cis-locations')
    plt.plot(loc_recall, loc_prec, 'C1-', label='trans-locations')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend()
    plt.title("%0.2f mAP Cis vs %0.2f mAP Trans" % (inter_ap, loc_ap))
    plt.savefig(det_folder + 'im_PR_cis_v_trans.jpg')

    np.savez(det_folder + 'im_cis_v_trans_prec_recall_data.npz', cis_prec=inter_prec, cis_recall=inter_recall, cis_ap=inter_ap, trans_prec=loc_prec, trans_recall=loc_recall, trans_ap=loc_ap)

import cPickle as  pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import json

#export PYTHONPATH=$PYTHONPATH:tfmodels/research
from object_detection.utils import metrics, np_box_ops
from object_detection.utils import per_image_evaluation
from utils import *

det_folder = '/ai4efs/models/object_detection/inception_resnet_v2_atrous/train_on_eccv_18_and_imerit_2/predictions/'
exp_name = 'ss_test'
detection_file = det_folder + exp_name + '.p'
db_file = '/ai4efs/annotations/modified_annotations/imerit_ss_annotations_1.json'

def compute_precision_recall_per_loc(detection_file, db_file):

    print('Loading detection file...')

    with open(detection_file) as f:
        detection_results = pickle.load(f)

    with open(db_file,'r') as f:
        data = json.load(f)
    print('Images: ', len(data['images']))
    print('Detection result Images: ', len(detection_results['images']))
    
    loc_to_ims = {}
    for im in data['images']:
        if im['location'] not in loc_to_ims:
            loc_to_ims[im['location']] = []
        loc_to_ims[im['location']].append(im['id'])
   
    print('Clustering detections by image...')
    #print(detection_results.keys())
    # group the detections and gts by image id:
    per_image_detections, per_image_gts = cluster_detections_by_image(detection_results) 

    per_image_eval = per_image_evaluation.PerImageEvaluation(
        num_groundtruth_classes=1,
        matching_iou_threshold=0.5,
        nms_iou_threshold=1.0,
        nms_max_output_boxes=10000
    )

    detection_labels = {loc:[] for loc in loc_to_ims}
    detection_scores = {loc:[] for loc in loc_to_ims}
    num_total_gts = {loc:0 for loc in loc_to_ims}
    count = {loc:0 for loc in loc_to_ims}
    
    precision = {}
    recall = {}
    average_precision = {}

    for cat, images in loc_to_ims.iteritems():
        
        for image_id in images:
            if image_id not in per_image_detections:
                #print(image_id)
                count[cat] += 1
                continue
            scores, tp_fp_labels = get_results_per_image(per_image_detections[image_id], 
                                                         per_image_gts[image_id], per_image_eval)
          
            detection_labels[cat].append(tp_fp_labels)
            detection_scores[cat].append(scores)
            num_gts = len(per_image_gts[image_id]['bboxes'])
            num_total_gts[cat] += num_gts

        if len(detection_scores[cat]) > 0:

            scores = np.concatenate(detection_scores[cat])
            labels = np.concatenate(detection_labels[cat]).astype(np.bool)
            #print(len(scores))
            #print(len(labels))
            precision[cat], recall[cat] = metrics.compute_precision_recall(
                scores, labels, num_total_gts[cat]
            )
        
            average_precision[cat] = metrics.compute_average_precision(precision[cat], recall[cat])
        else:
            print("no detections for " + cat)
        print(cat, count[cat], len(images))

    return precision, recall, average_precision

if __name__ == '__main__':
    prec, recall, ap= compute_precision_recall_per_loc(detection_file, db_file)
    for loc in ap:
        print(loc, ap[loc])
    #recall_thresh = 0.9
    #recall_idx = np.argmin([np.abs(x-recall_thresh) for x in inter_recall])
    #print('Cis prec. at ',inter_recall[recall_idx],' recall: ', inter_prec[recall_idx])
    #recall_idx = np.argmin([np.abs(x-recall_thresh) for x in loc_recall])
    #print('Trans prec. at ',loc_recall[recall_idx],' recall: ', loc_prec[recall_idx])
    plt.figure("Precision Recall Curves per Location")
    
    colors = cm.rainbow(np.linspace(0, 1, len(prec.keys())))
    for i in range(len(prec.keys())):
        cat = prec.keys()[i]
        if recall[cat] is not None:
            plt.plot(recall[cat], prec[cat], color = colors[i], label=cat)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend()
    plt.title("Per-location precision-recall")
    plt.savefig(det_folder + exp_name + '_PR_per_loc.jpg')

    plt.figure("AP per Location")
    sorted_ap = sorted(ap.items(), key = lambda x: x[1])
    #print(sorted_ap)
    plt.bar(range(len(sorted_ap)), [x[1] for x in sorted_ap])
    #plt.bar(sorted_ap)
    plt.ylabel("Location")
    plt.xlabel("Average Precision")
    plt.title("Per-location average precision")

    np.savez(det_folder + exp_name + '_per_loc_prec_recall_data.npz', prec = prec, recall = recall, ap = ap)



        

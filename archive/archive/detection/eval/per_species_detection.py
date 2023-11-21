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


det_folder = '/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_ss_no_deer_and_inat/predictions/'

exp_name = 'eccv_train'
detection_file = det_folder + exp_name + '.p'
db_file = '/ai4efs/databases/caltechcameratraps/eccv_train_and_imerit_2.json'

def compute_precision_recall_per_cat(detection_file, db_file):

    print('Loading detection file...')

    with open(detection_file) as f:
        detection_results = pickle.load(f)

    with open(db_file,'r') as f:
        data = json.load(f)

    im_to_seq = {}
    for im in data['images']:
        im_to_seq[im['id']] = im['seq_id']

    im_to_cat = {}
    for ann in data['annotations']:
        im_to_cat[ann['image_id']] = ann['category_id']
    #add empty category
    empty_id = max([cat['id'] for cat in data['categories']]) + 1
    data['categories'].append({'name': 'empty', 'id': empty_id})
    #add all images that don't have annotations, with cat empty
    for im in data['images']:
        if im['id'] not in im_to_cat:
            im_to_cat[im['id']] = empty_id    

    cat_id_to_cat = {}
    for cat in data['categories']:
        cat_id_to_cat[cat['id']] = cat['name']

    cat_to_ims = {cat_id:[] for cat_id in cat_id_to_cat}
    for im in data['images']:
        cat_to_ims[im_to_cat[im['id']]].append(im['id'])

    seqs = {}
    for im in detection_results['images']:
        if im in im_to_seq:
            if im_to_seq[im] not in seqs:
                seqs[im_to_seq[im]] = []
            seqs[im_to_seq[im]].append(im)

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

    detection_labels = {cat:[] for cat in cat_to_ims}
    detection_scores = {cat:[] for cat in cat_to_ims}
    num_total_gts = {cat:0 for cat in cat_to_ims}
    count = {cat:0 for cat in cat_to_ims}
    
    precision = {}
    recall = {}
    average_precision = {}

    for cat, images in cat_to_ims.iteritems():
        
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
            print("no detections for " + cat_id_to_cat[cat])
        print(cat_id_to_cat[cat], count[cat], len(images))

    return precision, recall, average_precision, cat_id_to_cat

if __name__ == '__main__':
    prec, recall, ap, cat_id_to_cat = compute_precision_recall_per_cat(detection_file, db_file)
    for cat in ap:
        print(cat_id_to_cat[cat], ap[cat])
    #recall_thresh = 0.9
    #recall_idx = np.argmin([np.abs(x-recall_thresh) for x in inter_recall])
    #print('Cis prec. at ',inter_recall[recall_idx],' recall: ', inter_prec[recall_idx])
    #recall_idx = np.argmin([np.abs(x-recall_thresh) for x in loc_recall])
    #print('Trans prec. at ',loc_recall[recall_idx],' recall: ', loc_prec[recall_idx])
    plt.figure("Precision Recall Curves per Class")
    colors = cm.rainbow(np.linspace(0, 1, len(prec.keys())))
    for i in range(len(prec.keys())):
        cat = prec.keys()[i]
        if recall[cat] is not None:
            plt.plot(recall[cat], prec[cat], color = colors[i], label=cat_id_to_cat[cat])
        
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend()
    plt.title("Per-category precision-recall")
    plt.savefig(det_folder + exp_name + '_PR_per_cat.jpg')

    np.savez(det_folder + exp_name + '_per_cat_prec_recall_data.npz', prec = prec, recall = recall, ap = ap, cat_id_to_cat = cat_id_to_cat)



        

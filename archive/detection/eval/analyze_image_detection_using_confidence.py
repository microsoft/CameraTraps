import cPickle as  pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import json

#export PYTHONPATH=$PYTHONPATH:tfmodels/research
from object_detection.utils import metrics
from utils import *

det_folder = '/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_eccv_18_only/predictions/'

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def compute_precision_recall_with_images(detection_file, detection_results=None, images_to_consider='all', get_night_day = None):
    
    if detection_results == None:
        print('Loading detection file...')
    
        with open(detection_file) as f:
            detection_results = pickle.load(f)
    
    print('Clustering detections by image...')
    use_im = get_images_to_consider(detection_results, images_to_consider, get_night_day)

    per_image_detections, per_image_gts = cluster_detections_by_image(detection_results, use_im)
    
    im_id_to_box_scores = {im:per_image_detections[im]['scores'] for im in per_image_detections}
    print('Running per-image analysis...')
    #need to loop over confidence values
    #for each value, check if any detections on the image are > conf
    #If so, that image gets class "animal"
    # then run prec, rec over the images to get the values for that confidence threshold, where gt is "animal" if num gt boxes > 0
    prec = []
    rec = []
    scores = []
    for conf in frange(0.001,1.0,0.001):
        scores.append(conf)
        tp = 0
        fp = 0
        fn = 0
        num_total_gts = 0
        count = 0
        for image_id, dets in per_image_detections.iteritems():
            im_detection_label = False
            im_gt_label = False
            im_num_gts  = []
            count +=1
            '''
            if count % 1000 == 0:
                print(str(count) + ' images complete')
            '''
            if max(im_id_to_box_scores[image_id]) > conf:
                im_detection_label = True
        
            gts = per_image_gts[image_id]
            num_gts = len(gts['bboxes'])
            
            if num_gts > 0:
                im_gt_label = True
                num_total_gts += 1
                if im_detection_label == True:
                    tp += 1
                else:
                    fn += 1
            elif im_detection_label == True:
                fp += 1


        #calc prec, rec for this confidence thresh
        im_prec = tp / float(tp + fp)
        im_rec = tp / float(tp + fn)
        prec.append(im_prec)
        rec.append(im_rec)
    
    prec.reverse()
    rec.reverse()
    scores.reverse()
    average_precision = metrics.compute_average_precision(np.asarray(prec), np.asarray(rec))
    
    
    return prec, rec, average_precision, scores

if __name__ == '__main__':

    inter_prec, inter_recall, inter_ap, _ = compute_precision_recall_with_images(det_folder + 'cis_test.p')
    loc_prec, loc_recall, loc_ap, _ = compute_precision_recall_with_images(det_folder + 'trans_test.p')
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
    plt.savefig(det_folder + 'im_PR_cis_v_trans_with_conf_thresh.jpg')

    np.savez(det_folder + 'im_cis_v_trans_prec_recall_data_with_conf_thresh.npz', cis_prec=inter_prec, cis_recall=inter_recall, cis_ap=inter_ap, trans_prec=loc_prec, trans_recall=loc_recall, trans_ap=loc_ap)

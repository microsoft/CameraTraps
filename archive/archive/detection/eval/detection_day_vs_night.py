import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import json
import pickle

from analyze_detection import compute_precision_recall
from analyze_image_detection_one_guess_per_image import compute_precision_recall_with_images
from analyze_sequence_detection_one_guess_per_sequence import compute_precision_recall_with_sequences


det_folder = '/ai4efs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/train_on_ss_inat_and_inat_night_balanced/predictions/'
exp_name = 'eccv_train'
#exp_name = 'small_balanced_cct'
detection_file = det_folder + exp_name + '.p'

db_file = '/ai4efs/databases/caltechcameratraps/eccv_train_and_imerit_2.json'
day_night_json = '/ai4efs/databases/caltechcameratraps/eccv_18_train_nightday.json'

if __name__ == '__main__':
    detection_results = pickle.load(open(detection_file,'r'))
    day_prec, day_recall, day_ap = compute_precision_recall(detection_file, detection_results=detection_results, images_to_consider='day',get_night_day=day_night_json)
    night_prec, night_recall, night_ap = compute_precision_recall(detection_file, detection_results=detection_results, images_to_consider='night',get_night_day=day_night_json)
    print('day mAP: ', day_ap,', night mAP: ',night_ap)
    recall_thresh = 0.95
    recall_idx = np.argmin([np.abs(x-recall_thresh) for x in day_recall])
    print('Day prec. at ',day_recall[recall_idx],' recall: ', day_prec[recall_idx])
    recall_idx = np.argmin([np.abs(x-recall_thresh) for x in night_recall])
    print('Night prec. at ',night_recall[recall_idx],' recall: ', night_prec[recall_idx])
    plt.figure("Precision Recall Curve Day v. Night")
    plt.plot(day_recall, day_prec, 'C0-', label='day')
    plt.plot(night_recall, night_prec, 'C1--', label = 'night')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.legend()
    plt.title("%0.2f mAP day, %0.2f mAP night" % (day_ap, night_ap))
    plt.savefig(det_folder + exp_name +'_PR_day_v_night.jpg')

    np.savez(det_folder + exp_name + '_day_v_night_prec_recall_data.npz', day_prec=day_prec, day_recall=day_recall, day_ap=day_ap, night_prec=night_prec, night_recall=night_recall, night_ap=night_ap)


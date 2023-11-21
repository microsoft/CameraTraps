#
# make_detection_db_for_viewing.py
#
# Given a .json file with ground truth bounding boxes, and a .p file containing detections for the same images,
# creates a new .json file with separate classes for ground truth and detection, suitable for viewing in the Visipedia
# annotation tool.
#
 
#%% Imports and constants

import json
import pickle
import uuid

detection_file = '/ai4efs/models/object_detection/inception_resnet_v2_atrous/train_on_eccv_18_and_imerit_2/predictions/ss_test.p'
gt_db = '/ai4efs/annotations/modified_annotations/imerit_ss_annotations_1.json'
output_file = '/ai4efs/models/object_detection/inception_resnet_v2_atrous/train_on_eccv_18_and_imerit_2/predictions/ss_test_detection_db.json'


#%% Main function

def make_detection_db(detection_file, gt_db, det_thresh=0.9):
    
    with open(detection_file,'r') as f:
        detection_results = pickle.load(f)

    with open(gt_db,'r') as f:
        data = json.load(f)
    
    images = data['images']
    # im_id_to_im = {im['id']:im for im in images}
    for im in images:
        im['id'] = im['id'].split('/')[-1]
    print(images[0])
    annotations = data['annotations']
    
    for ann in annotations:
        ann['image_id'] = ann['image_id'].split('/')[-1]    
    # make new categories to distinguish between ground truth and detections
    categories = [{'name': 'gt', 'id': 0},{'name':'det','id':1}]

    # update all gt annotations to be class "gt"
    for ann in annotations:
        ann['category_id'] = 0

    # collect all detections by image
    per_image_detections = {detection_results['images'][idx] :{'bboxes': detection_results['detections'][idx], 'scores': detection_results['detection_scores'][idx], 'labels':detection_results['detection_labels'][idx]} for idx in range(len(detection_results['images']))}
    
    # keep any detection with score above det_thresh
    for im, dets in per_image_detections.iteritems():
        for idx in range(len(dets['bboxes'])):
            if dets['scores'][idx] >= det_thresh:
                new_ann = {}
                new_ann['image_id'] = im.split('/')[-1]
                new_ann['category_id'] = 1 #category "det" for detection
                #need to convert bbox from [x1,y1,x2,y2] to [x,y,w,h]
                bbox = dets['bboxes'][idx]
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                new_ann['bbox'] = bbox
                new_ann['score'] = float(dets['scores'][idx])
                new_ann['id'] = str(uuid.uuid1())
                annotations.append(new_ann)

    # add "info" and "licenses" for annotation tools to function
    info = data['info']
    info['description'] = 'detections above %0.2f'.format(det_thresh)
    licenses = []
    
    # create new db
    new_data = {}
    new_data['images'] = images
    new_data['categories'] = categories
    new_data['annotations'] = annotations
    new_data['licenses'] = licenses
    new_data['info'] = info            

    return new_data


#%% Command-line handling
    
if __name__ == '__main__':
    new_data = make_detection_db(detection_file, gt_db)
    with open(output_file,'w') as f:
        json.dump(new_data,f)




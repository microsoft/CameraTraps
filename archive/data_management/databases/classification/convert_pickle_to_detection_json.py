# This file converts the CSV output of the batch processing API to a pickle file, which can be used by the 
# script ./make_classification_dataset.py

import argparse
import os
import json
import pickle 
import pandas
import numpy as np
import tqdm
from pycocotools.coco import COCO

# Assumes that the root of the CameraTrap repo is in the PYTHONPATH
import ct_utils

# Minimum threshold to put a detection into the output JSON file
DETECTION_CONF_THRESHOLD = 0.1

parser = argparse.ArgumentParser('Converts pickle files with detection created by ./make_classification_dataset.py ' + \
    ' to a JSON file that follows the detection API output format.')
parser.add_argument("input_pkl", type=str, help='Path to the desired input pickle file')
parser.add_argument("dataset_json", type=str, help='Path to the dataset description in COCO json format.')
parser.add_argument("output_json", type=str, help='Path to the output file that will contain the detection in API JSON format.')
args = parser.parse_args()

# Parameter check
assert os.path.isfile(args.input_pkl), 'ERROR: The input file could not be found!'
assert os.path.isfile(args.dataset_json), 'ERROR: The dataset json file could not be found!'
assert not os.path.isfile(args.output_json), 'ERROR: The output file exists already!'

# Load detections from input
print('Loading detections from ' + args.input_pkl)
with open(args.input_pkl, 'rb') as f:
    detections = pickle.load(f)

print('Loading dataset annotations from ' + args.dataset_json)
# Load COCO style annotations
coco = COCO(args.dataset_json)

# Build output JSON in format version 1.0
js = dict()
# Adding the only known metadata info
js['info'] = dict(format_version = '1.0')
# The pickle file does not contain category information, so we assume the default
js['detection_categories'] = {"1": "animal", "2": "person", "4": "vehicle"}

js['images'] = list()
# For each image with detections
for im_key in tqdm.tqdm(list(detections.keys())):

    im_dict = dict()
    im_dict['file'] = coco.imgs[im_key]['file_name']
    # for each detection 
    det_list = list()
    det_boxes_old_format = detections[im_key]['detection_boxes']
    det_classes = detections[im_key]['detection_classes']
    det_conf = detections[im_key]['detection_scores']
    # Convert boxes from [ymin, xmin, ymax, xmax] format to 
    # [x_min, y_min, width_of_box, height_of_box]
    tmp = det_boxes_old_format.T
    det_boxes = np.array([tmp[1], tmp[0], tmp[3]-tmp[1], tmp[2]-tmp[0]]).T
    del tmp

    for det_id in range(len(det_boxes)):

        if det_conf[det_id] > DETECTION_CONF_THRESHOLD:
            det_list.append(dict(category=str(det_classes[det_id]),
                                conf=ct_utils.truncate_float(det_conf[det_id].item()),
                                bbox=ct_utils.truncate_float_array(det_boxes[det_id].tolist())))
    im_dict['detections'] = det_list
    if len(im_dict['detections']) > 0:
        im_dict['max_detection_conf'] = ct_utils.truncate_float(max(det_conf).item())
    else:
        im_dict['max_detection_conf'] = 0.

    js['images'].append(im_dict)
    
# Write output json
with open(args.output_json, 'wt') as fi:
    json.dump(js, fi, indent=1)

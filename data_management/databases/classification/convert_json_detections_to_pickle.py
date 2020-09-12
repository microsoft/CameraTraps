# This file converts the JSON output of the batch processing API to a pickle file, which can be used by the 
# script ./make_classification_dataset.py

import argparse
import os
import json
import pickle 
import pandas
import numpy as np
import tqdm


parser = argparse.ArgumentParser('This file converts the JSON output of the batch processing API to a pickle file, ' + \
                   'which can be used by the script ./make_classification_dataset.py')
parser.add_argument("input_json", type=str, help='Path to the JSON file that contains the API output')
parser.add_argument("output_pkl", type=str, help='Path to the desired output pickle file')
parser.add_argument("--detection_category_whitelist", nargs='+', default=['1'], metavar='CAT_ID',
                    help='List of detection categories to use. Default: ["1"]')
args = parser.parse_args()

assert os.path.isfile(args.input_json), 'ERROR: The input CSV file could not be found!'
assert not os.path.isfile(args.output_pkl), 'ERROR: The output file exists already!'
assert isinstance(args.detection_category_whitelist, list)
assert len(args.detection_category_whitelist) > 0

with open(args.input_json, 'rt') as fi:
    j = json.load(fi)

detection_dict = dict()

for row in tqdm.tqdm(list(range(len(j['images'])))):

    cur_image = j['images'][row]
    
    key = cur_image['file']

    max_conf = 0
    conf = []
    boxes = []

    for det in cur_image['detections']:
        if det['category'] in args.detection_category_whitelist:
            max_conf = max(max_conf, float(det['conf']))
            conf.append(float(det['conf']))
            # Convert boxes from JSON   [x_min, y_min, width_of_box, height_of_box]
            #                 to PICKLE [ymin,  xmin,  ymax,         xmax] 
            box = det['bbox']
            boxes.append([box[1], box[0], box[1] + box[3], box[0]+ box[2]])

    detection_dict[key] = dict(detection_scores=conf, detection_boxes=boxes)


# Write detections to file with pickle
with open(args.output_pkl, 'wb') as f:
    pickle.dump(detection_dict, f, pickle.HIGHEST_PROTOCOL)

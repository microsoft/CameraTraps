# This file converts the CSV output of the batch processing API to a pickle file, which can be used by the 
# script ./make_classification_dataset.py

import argparse
import os
import json
import pickle 
import pandas
import numpy as np
import tqdm

parser = argparse.ArgumentParser('This file converts the CSV output of the batch processing API to a pickle file, ' + \
                   'which can be used by the script ./make_classification_dataset.py')
parser.add_argument("input_csv", type=str, help='Path to the CSV file that contains the API output')
parser.add_argument("output_pkl", type=str, help='Path to the desired output pickle file')
args = parser.parse_args()

assert os.path.isfile(args.input_csv), 'ERROR: The input CSV file could not be found!'
assert not os.path.isfile(args.output_pkl), 'ERROR: The output file exists already!'

csv = pandas.read_csv(args.input_csv, header=0, names=['filename',  'maxscore', 'boxes'])
detection_dict = dict()
for row in tqdm.tqdm(list(csv.itertuples())):
    boxes = np.array(json.loads(row.boxes))
    if boxes.size > 0:
        box_selection = np.isclose(boxes[:,5], 1)
        conf = boxes[box_selection, 4].ravel()
        boxes = boxes[box_selection, :4]
    else:
        conf = np.array([[]])
        boxes = np.array([[]])
    key = row.filename
    detection_dict[key] = dict(detection_scores=conf, detection_boxes=boxes)


# Write detections to file with pickle
with open(args.output_pkl, 'wb') as f:
    pickle.dump(detection_dict, f, pickle.HIGHEST_PROTOCOL)

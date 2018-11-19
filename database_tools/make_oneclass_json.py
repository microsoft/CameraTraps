#
# make_oneclass_json.py
#
# Takes a coco-camera-traps .json database and collapses species classes to binary, optionally removing
# labels from empty images (to be detector-friendly) (depending on "experiment_type").
#
# Assumes that empty images are labeled as "empty".
#

import json
import argparse

def make_binary_json(data, experiment_type='detection',ignore_humans = False):
    #converts a multiclass file to oneclass animal/no animal, for either detection or classification
    cat_id_to_name = {cat['id']:cat['name'] for cat in data['categories']}
    new_cats = [{'name': 'animal', 'id':1},{'name':'empty', 'id':0}]
    new_anns = []
    for ann in data['annotations']:
        if experiment_type == 'classification':
            if cat_id_to_name[ann['category_id']] in ['empty']:
                ann['category_id'] = 0
                new_anns.append(ann)
            else:
                ann['category_id'] = 1
                new_anns.append(ann)
        else:
            if 'bbox' in ann and cat_id_to_name[ann['category_id']] not in ['empty']:
                ann['category_id'] = 1
                new_anns.append(ann)
    print(len(data['annotations']),len(new_anns))

    data['categories'] = new_cats
    data['annotations'] = new_anns

    return data


def parse_args():

    parser = argparse.ArgumentParser(description = 'Convert a multiclass .json to a oneclass .json')
    
    parser.add_argument('--input_file', dest='input_file', 
                         help='Path to multiclass .json to be converted',
                         type=str, required=True)
    parser.add_argument('--output_file', dest='output_file',
                         help='Path to store oneclass .json',
                         type=str, required=True)
    parser.add_argument('--experiment_type', dest='experiment_type',
                         help='Type of experiment: classification or detection',
                         type=str, default='detection')
    parser.add_argument('--ignore_humans', dest='ignore_humans',
                         help='Should human boxes be ignored?',
                         required=False, action='store_true',default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('Reading input file')
    with open(args.input_file,'r') as f:
        data = json.load(f)
    print('Converting to oneclass')
    oneclass_data = make_binary_json(data, args.experiment_type, args.ignore_humans)

    json.dump(oneclass_data,open(args.output_file,'w'))

if __name__ == '__main__':
    main()

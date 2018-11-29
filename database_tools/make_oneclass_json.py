#
# make_oneclass_json.py
#
# Takes a coco-camera-traps .json database and collapses species classes to binary, 
# optionally removing labels from empty images (to be detector-friendly) (depending on 
# "experiment_type").
#
# Assumes that empty images are labeled as "empty".
#

#%% Imports and environment

import json
import argparse
import copy


#%% Core conversion function

def make_binary_json(data,experiment_type='detection',ignore_humans = False):
    '''
    converts a multiclass .json object to one-class animal/no animal, for either detection or 
    classification.  Modifies "data" in-place.
    '''
    cat_id_to_name = {cat['id']:cat['name'] for cat in data['categories']}
    print('Mapping {} categories to binary'.format(len(cat_id_to_name)))
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
        elif experiment_type == 'detection':
            # We're removing empty images from the annotation list, but not from
            # the "images" list; they'll still get used in detector training.
            if ('bbox' in ann) and (cat_id_to_name[ann['category_id']] not in ['empty']):
                ann['category_id'] = 1
                new_anns.append(ann)
            else:
                pass
                # print('Ignoring empty annotation')
        else:
            raise ValueError('Unknown experiment type: {}'.format(experiment_type))
    
    data['categories'] = new_cats
    data['annotations'] = new_anns

    return data


#%% Interactive driver
    
if False:

    #%%
    
    import os
    base_dir = r'D:\temp\snapshot_serengeti_tfrecord_generation'
    input_file = os.path.join(base_dir,'imerit_batch7_renamed_uncorrupted_filtered.json')
    output_file = os.path.join(base_dir,'imerit_batch7_renamed_uncorrupted_filtered_oneclass.json')
    ignore_humans = True
    experiment_type = 'detection'
    
    assert(os.path.isfile(input_file))
    
    # Load annotations
    with open(input_file,'r') as f:
            data_multiclass = json.load(f)    
            
    # Convert from multi-class to one-class
    data_oneclass = copy.deepcopy(data_multiclass)
    data_oneclass = make_binary_json(data_oneclass,experiment_type,ignore_humans)
    
    # Write out the one-class data
    json.dump(data_oneclass, open(output_file,'w'))

    print('Wrote {} annotations (of {} original annotations) to {}'.format(
            len(data_oneclass['annotations']),
            len(data_multiclass['annotations']),
            output_file))


#%% Command-line driver
    
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

#
# create_tfrecords_from_coco.py
#
# This script creates a tfrecords file from a classification dataset in COCO format.

#%% Imports and environment

import json
import os
from tqdm import tqdm
import numpy as np
import argparse
import sys
if sys.version_info.major >= 3:
    from create_tfrecords_py3 import create
else:
    from create_tfrecords import create


#%% Main tfrecord generation function

def create_tfrecords_format(dataset_root, annotation_filename, output_tfrecords_folder, dataset_name,
                                                   num_threads=5, ims_per_record=200):
    with open(os.path.join(dataset_root, annotation_filename), 'rt') as fi:
        js = json.load(fi)

    cat_dict = {cat['id']: cat for cat in js['categories']}
    # We remap all category IDs such that they are consecutive starting from zero
    # If this is already the case for the input dataset, then the remapping will not 
    # have any effect, i.e. the order of the classes will remain unchanged
    cat_ids_sorted = sorted(list(cat_dict.keys()))
    cat_id_remap = {old_id: new_id for new_id, old_id in enumerate(cat_ids_sorted)}
    img_to_new_cat_id = {ann['image_id']: cat_id_remap[ann['category_id']] for ann in js['annotations']}

    cat_names = [cat_dict[cat_id]['supercategory'] + ' - ' + cat_dict[cat_id]['name'] for cat_id in cat_ids_sorted]

    vis_data = []
    for cur_img in tqdm(js['images']):
        img_file = os.path.join(dataset_root, cur_img['file_name'])
        assert os.path.exists(img_file), 'Could not find image ' + img_file

        image_data = {}
        image_data['filename'] = img_file
        image_data['id'] = cur_img['id']

        image_data['class'] = {}
        image_data['class']['label'] = img_to_new_cat_id[cur_img['id']]
        image_data['class']['text'] = cat_names[image_data['class']['label']]

        # Propagate optional metadata to tfrecords
        image_data['height'] = cur_img['height']
        image_data['width'] = cur_img['width']

        # endfor each annotation for the current image
        vis_data.append(image_data)
    # endfor each image

    print('Creating tfrecords for {} from {} images'.format(dataset_name,len(vis_data)))

    # Calculate number of shards to get the desired number of images per record,
    # ensure it is evenly divisible by the number of threads
    num_shards = int(np.ceil(float(len(vis_data))/ims_per_record))
    while num_shards % num_threads:
        num_shards += 1
    print('Number of shards: ' + str(num_shards))

    failed_images = create(
      dataset=vis_data,
      dataset_name=dataset_name,
      output_directory=output_tfrecords_folder,
      num_shards=num_shards,
      num_threads=num_threads,
      store_images=True
    )

    return failed_images, cat_names

#%% Command-line driver

#%% Driver
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Make tfrecords from a COCO-style classification dataset.')
    parser.add_argument('--data_root', required=True,
                         type=str, metavar='DATASET_ROOT', help='Path to the root directory of the dataset.')
    parser.add_argument('--train_file', default='trainval2017.json',
                        type=str, metavar='TRAIN_FILE', help='Name of the json file containing the training annotation ' + \
                        '(default: trainval2017.json). Should be located within the data directory.')
    parser.add_argument('--val_file', default='minival2017.json',
                        type=str, metavar='VAL_FILE', help='Name of the json file containing the validation annotation ' + \
                        '(default: minival2017.json). Should be located within the data directory.')
    parser.add_argument('--output_tfrecords_folder', dest='output_tfrecords_folder',
                         help='Path to folder to save tfrecords in',
                         type=str, required=True)
    parser.add_argument('--num_threads', dest='num_threads',
                         help='Number of threads to use while creating tfrecords',
                         type=int, default=5)
    parser.add_argument('--ims_per_record', dest='ims_per_record',
                         help='Number of images to store in each tfrecord file',
                         type=int, default=200)

    args = parser.parse_args()

    try:
        os.makedirs(args.output_tfrecords_folder)
    except Exception as e:
        if os.path.isdir(args.output_tfrecords_folder):
            raise Exception('Directory {} already exists, '.format(args.output_tfrecords_folder) + \
                            'please remove any existing content')
        else:
            raise e
    failed_train, train_classnames = create_tfrecords_format(args.data_root,
                                                       args.train_file,
                                                       args.output_tfrecords_folder,
                                                       'train',
                                                       args.num_threads,
                                                       args.ims_per_record)

    failed_test, val_classnames = create_tfrecords_format(args.data_root,
                                                       args.val_file,
                                                       args.output_tfrecords_folder,
                                                       'test',
                                                       args.num_threads,
                                                       args.ims_per_record)
    assert train_classnames == val_classnames, 'Error: classes in training and testing json differ.'

    label_map = []
    for idx, cname in list(enumerate(train_classnames)):
        label_map += ['item {{name: "{}" id: {}}}\n'.format(cname, idx)]
    with open(os.path.join(args.output_tfrecords_folder, 'label_map.pbtxt'), 'w') as f:
        f.write(''.join(label_map))

    print('Finished with {} failed images'.format(len(failed_train) + len(failed_test)))

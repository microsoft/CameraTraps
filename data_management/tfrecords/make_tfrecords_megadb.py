"""
Create TF records with images available locally and a JSON containing entries in the MegaDB.

Need to be run using the `cameratraps-detector` environment specified by `environment-detector.yml`
"""

import json
import math
import os

from tqdm import tqdm

from data_management.tfrecords.utils.create_tfrecords import create

#%% Parameters

images_per_record = 5000  # image size ranges from 200KB to 900KB
num_threads = 100

dataset_name = 'mdv5boxes'  # 'inat2017t', 'coco2017t'
assert len(dataset_name) == 9

split_name = 'train'  # 'train', 'val__' or 'test_'
assert len(split_name) == 5

image_dir = '/ilipika_disk_0/camtraps/all_images_splitted/train'
assert split_name.replace('_', '') in image_dir

annotations_path = '/ilipika_disk_0/camtraps/20210802220004_all_id2.json'

tfrecord_dir = '/ilipika_disk_0/camtraps/mdv5_tfrecords'

label_map = {
    'animal': 1,
    'person': 2,
    'vehicle': 3
}


#%% Load the annotations queried from megadb

with open(annotations_path) as f:
    all_annotations = json.load(f)

all_annotations = {i['download_id']: i for i in all_annotations}


#%% Make the "dataset" required by create_tfrecords.py
print('Compiling the dataset required by create_tfrecords...')

dataset = []
num_im_w_group = 0

for image_file_name in tqdm(os.listdir(image_dir)):
    download_id, ext = os.path.splitext(image_file_name)

    if download_id not in all_annotations:
        print('{} not in the annotation file, skipping.'.format(download_id))
        continue

    anno_entry = all_annotations[download_id]

    xmin_li, xmax_li, ymin_li, ymax_li, label_li = [], [], [], [], []

    skip_im_because_of_group = False  # skip all images with "group" boxes
    for bbox_entry in anno_entry['bbox']:
        cat = bbox_entry['category']
        if cat == 'group':
            skip_im_because_of_group = True
            num_im_w_group += 1
            break

        box = bbox_entry['bbox']
        xmin, ymin, width, height = box[0], box[1], box[2], box[3]
        xmax = xmin + width
        ymax = ymin + height
        xmin_li.append(xmin)
        xmax_li.append(xmax)
        ymin_li.append(ymin)
        ymax_li.append(ymax)
        label_li.append(label_map[cat])

    if skip_im_because_of_group:
        continue  # continue to the next image; do not append examples

    example = {
        'filename': os.path.join(image_dir, image_file_name),
        'id': download_id,
        'object': {
            'bbox': {
                'xmin': xmin_li,
                'xmax': xmax_li,
                'ymin': ymin_li,
                'ymax': ymax_li,
                'label': label_li
            }
        }
    }
    dataset.append(example)

print('Number of examples: {}. Number of images skipped because of group: {}'.format(len(dataset),
                                                                                     num_im_w_group))


#%% Create tfrecords

name_for_tfrecords_set = dataset_name + '_' + split_name

num_shards = int(math.ceil(float(len(dataset)) / images_per_record))
while num_shards % num_threads:
    num_shards += 1  # increase num_shards until it works with num_threads

print('Using {} shards for {}'.format(num_shards, name_for_tfrecords_set))

failed_images = create(
    dataset=dataset,
    dataset_name=name_for_tfrecords_set,
    output_directory=tfrecord_dir,
    num_shards=num_shards,
    num_threads=num_threads,
    shuffle=True,
    store_images=True
)

print('Number of images that failed: {}.'.format(len(failed_images)))
with open(os.path.join(tfrecord_dir, 'failed_{}_{}.json'.format(dataset_name, split_name)), 'w') as f:
    json.dump(failed_images, f, indent=1)
print('Done!')

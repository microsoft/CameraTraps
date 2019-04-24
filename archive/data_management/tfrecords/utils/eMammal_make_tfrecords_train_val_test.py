import json
import os
import sys
from datetime import datetime

import numpy as np
import random

sys.path.insert(0, "./..")
from utils.create_tfrecords_py3 import create

# eMammal_make_tfrecords_train_val_test.py
#
# From the tfrecords_format json version of the database, creates three splits
# of tf_records according to a previously decided split of full image IDs.

# configurations and paths
ims_per_record = 200.0
num_threads = 5
save_empty_prob = 0.25  # given an empty image assigned to val/test set, include it in the set with this probability to create a balanced empty/non-empty set

splits_path = '/home/yasiyu/yasiyu_temp/splits_eMammal/eMammal_loc_splits_20180929.json'  # output of eMammal_make_splits.py
split_by = 'location'  # 'location' or 'image'

# a tfrecord_format json
database_file = '/home/yasiyu/yasiyu_temp/eMammal_db/eMammal_20180929_nohuman_tfrecord_format.json'

output_dir = '/datadrive/emammal_tfrecords/eMammal_loc_splits_20180929_nohuman'
os.makedirs(output_dir, exist_ok=True)

start_time = datetime.now()

splits = json.load(open(splits_path))  # in terms of full image IDs or locations
train = splits['train']
val = splits['val']
test = splits['test']

data = json.load(open(database_file,'r'))
print('Number of entries in data: ', len(data))
print(data[0])

# these are number of images
num_train, num_val, num_test = 0, 0, 0
num_empty_train, num_empty_val, num_empty_test = 0, 0, 0

if split_by == 'location':  # make train, val and test in terms of image IDs
    print('Retrieving images from each location...')
    train_loc, val_loc, test_loc = set(train), set(val), set(test)
    train, val, test = [], [], []
    for im in data:
        is_empty = True if im['text'] == 'empty' else False
        rand = random.uniform(0, 1)

        if im['location'] in train_loc:
            num_train += 1
            # do not include empty images in the train set; note that some images from non-empty sequences
            # end up being empty (no bbox can be labeled), so these will be included in train set anyways
            if is_empty:
                num_empty_train += 1
            else:
                train.append(im['id'])
        elif im['location'] in val_loc:
            num_val += 1
            if is_empty:
                num_empty_val += 1
                if rand < save_empty_prob:
                    val.append(im['id'])
            else:
                val.append(im['id'])

        elif im['location'] in test_loc:
            num_test += 1
            if is_empty:
                num_empty_test += 1
                if rand < save_empty_prob:
                    test.append(im['id'])
            else:
                test.append(im['id'])

    print('Including empties, there are {} train imgs, {} val imgs, {} test imgs.'.format(num_train, num_val, num_test))

    print('Number of empty images: {} in train, {} in val, {} in test.'.format(num_empty_train, num_empty_val, num_empty_test))

    print('Actual number of images included in tfrecords: {} in train, {} in val, {} in test'.format(len(train), len(val), len(test)))


im_id_to_im = {im['id']:im for im in data}
failed_images = []

def create_tfrecords(dataset, dataset_name):
    num_shards = int(np.ceil(float(len(dataset)) / ims_per_record))
    while num_shards % num_threads:
        num_shards += 1
    print('Using {} shards for {}'.format(num_shards, dataset_name))
    failed_images = create(
        dataset=dataset,
        dataset_name=dataset_name,
        output_directory=output_dir,
        num_shards=num_shards,
        num_threads=num_threads,
        store_images=True
    )
    return failed_images


print('Creating train tfrecords...')
dataset = [im_id_to_im[id] for id in train]
failed_images.append(create_tfrecords(dataset, 'train'))

print('Creating val tfrecords...')
dataset = [im_id_to_im[id] for id in val]
failed_images.append(create_tfrecords(dataset, 'val'))

print('Creating test tfrecords...')
dataset = [im_id_to_im[id] for id in test]
failed_images.append(create_tfrecords(dataset, 'test'))


print('Saving failed images...')
with open(os.path.join(output_dir, 'failed_images.json'), 'w') as f:
    json.dump(failed_images, f)

print('Running the script took {}.'.format(datetime.now() - start_time))

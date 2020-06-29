import json
import math
import os
from random import shuffle
import sys

from utils.create_tfrecords_format import create_tfrecords_format
from utils.create_splits import create_splits
from utils.create_tfrecords_py3 import create

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#%% Configurations
dataset_name = 'ss_sea1'  # this needs to be exactly 7 letters long

# if the following two files are provided, only the tfrecord generation section will be executed
tfrecord_format_json_path = None  # '/datadrive/megadetectorv3_tfrecords/intermediate/rspb_bboxes_20190409_tfrecord_format.json'
split_file_path = '/beaver_disk/camtrap/ss_season1/benchmark/SnapshotSerengetiSplits_v0_benchmark.json' # None if no split files made yet

# proceed to generate tfrecords if True
generate_tfrecords = True

database_path = '/beaver_disk/camtrap/ss_season1/benchmark/SnapshotSerengetiBboxesS01_20190903_train.json'
image_file_root = '/home/marmot/camtrap/mnt/snapshot-serengeti-v2/SER'

tfrecords_out_dir = '/beaver_disk/camtrap/benchmark_tfrecords'
others_out_dir = '/beaver_disk/camtrap/benchmark_tfrecords/intermediate'  # where the tfrecord_format json and the splits will be written

exclude_images_without_bbox = False

# approximate fraction to split the new entries by
split_frac = {
    'train': 1,
    'val': 0.0,
    'test': 0.0
}

append_to_previous_split = False  # True if adding to splits from a previous splits file
previous_split_path = None  # path to the splits json or None
split_by = 'location'  # 'location' or 'id', a field in the image entries in the database

# categories in the database to include
cats_to_include = ['animal', 'person']
# in addition, any images with a 'group' label will not be included to avoid confusion
# see 'image_contains_group' in create_tfrecords_format.py

images_per_record = 700
num_threads = 5


#%% Input validation
assert len(dataset_name) == 7 and '~' not in dataset_name, \
    'dataset_name needs to be exactly 7 letters long. Do not use ~ for padding; use _'

assert math.isclose(split_frac['train'] + split_frac['val'] + split_frac['test'], 1.0), \
    'Fractions to split dataset by do not add up to ~1.0.'

assert os.path.exists(database_path), 'Database file does not exist at {}'.format(database_path)
assert os.path.exists(image_file_root), 'No directory exists at image_file_root {}'.format(image_file_root)

if append_to_previous_split:
    assert os.path.exists(previous_split_path), \
        'append_to_previous_split is True but previous_split_path does not exist.'

assert split_by in ['location', 'id', 'seq_id'], "split_by attribute '{}' not allowed.".format(split_by)
assert len(cats_to_include) > 0, 'cats_to_include is empty'

os.makedirs(tfrecords_out_dir, exist_ok=True)
os.makedirs(others_out_dir, exist_ok=True)


#%% Convert the COCO Camera Trap format data to another json
# that aligns with the fields in the resulting tfrecords
if tfrecord_format_json_path is None:
    print('Creating tfrecords format json...')
    tfrecord_format_json = create_tfrecords_format(dataset_name, database_path, image_file_root,
                                                   cats_to_include=cats_to_include,
                                                   exclude_images_without_bbox=exclude_images_without_bbox)

    print('Saving the tfrecord_format json...')
    name = os.path.basename(database_path).split('.json')[0] + '_tfrecord_format.json'
    out_path = os.path.join(others_out_dir, name)
    with open(out_path, 'w') as f:
        json.dump(tfrecord_format_json, f, ensure_ascii=False, indent=1, sort_keys=True)

    print('Json in tfrecord_format saved at {}.\n'.format(out_path))
else:
    tfrecord_format_json = json.load(open(tfrecord_format_json_path))


#%% Make train/val/test splits
if split_file_path is None:
    previous_split = {}
    if append_to_previous_split:
        previous_split = json.load(open(previous_split_path, 'r'))

    updated_split = create_splits(tfrecord_format_json, split_by, split_frac,
                                  previous_split=previous_split)

    name = os.path.basename(database_path).split('.json')[0] + '_split.json'
    out_path = os.path.join(others_out_dir, name)
    with open(out_path, 'w') as f:
        json.dump(updated_split, f, indent=1)

    print('Updated splits saved at {}.\n'.format(out_path))
else:
    updated_split = json.load(open(split_file_path))


#%% Write the tfrecords
if not generate_tfrecords:
    print('generate_tfrecords is False, exiting.')
    sys.exit(0)

train_locs, val_locs, test_locs = set(updated_split['train']), set(updated_split['val']), set(updated_split['test'])

train, val, test = [], [], []
for im in tfrecord_format_json:
    if im['location'] in train_locs:
        train.append(im)
    elif im['location'] in val_locs:
        val.append(im)
    elif im['location'] in test_locs:
        test.append(im)

shuffle(train)
shuffle(val)
shuffle(test)

print('Number of images included in tfrecords: {} in train, {} in val, {} in test'.format(len(train), len(val), len(test)))


def create_tfrecords(dataset, dataset_name):
    num_shards = int(math.ceil(float(len(dataset)) / images_per_record))
    while num_shards % num_threads:
        num_shards += 1  # increase num_shards until it works with num_threads
    print('Using {} shards for {}'.format(num_shards, dataset_name))
    failed_images = create(
        dataset=dataset,
        dataset_name=dataset_name,
        output_directory=tfrecords_out_dir,
        num_shards=num_shards,
        num_threads=num_threads,
        store_images=True
    )
    print('Number of images that failed: {}.'.format(len(failed_images)))
    return failed_images


failed_images = []

# want the file names of all tfrecords to be of the same format and length in each part
print('Creating train tfrecords...')
failed_images.append(create_tfrecords(train, '{}~train'.format(dataset_name)))

print('Creating val tfrecords...')
failed_images.append(create_tfrecords(val, '{}~val__'.format(dataset_name)))

print('Creating test tfrecords...')
failed_images.append(create_tfrecords(test, '{}~test_'.format(dataset_name)))

print('Saving failed images...')
with open(os.path.join(others_out_dir, '{}_failed_images.json'.format(dataset_name)), 'w') as f:
    json.dump(failed_images, f)

print('Tfrecords generated.')

#
# make_tfrecords_cis_trans.py
#
# Given a .json file that contains a three-element list (train/val/test) of image IDs and a .json database that contains
# those image IDs, generates tfrecords whose filenames include "train"/"val"/"test"
# 

import json
import numpy as np
from utils.create_tfrecords import create
from utils.create_tfrecords_format import create_tfrecords_from_json
from archive.data_management.tfrecords.create_classification_tfrecords_format import create_classification_tfrecords_from_json
import tensorflow as tf

datafolder = '/ai4efs/'
database_file = datafolder + 'databases/snapshotserengeti/oneclass/SnapshotSerengeti_Seasons_1_to_4_tfrecord_format_valid_files.json'
output_dir = '/ss_data/tfrecords'
image_file_root = '/ss_data/dataD/snapshot/'
experiment_type = 'classification'
ims_per_record = 500.0
num_threads = 5


if 'tfrecord_format' in database_file:
    with open(database_file,'r') as f:
        data = json.load(f)
else:
    print('Creating tfrecords format database')
    if experiment_type == 'classification':
        data = create_classification_tfrecords_from_json(database_file, image_file_root)
        json.dump(data,open('/ai4efs/databases/snapshotserengeti/oneclass/SnapshotSerengeti_Seasons_1_to_4_tfrecord_format.json','w'))
    else:
        data = create_tfrecords_from_json(database_file, image_file_root)

print('Images: ',len(data))
print(data[0])
data_split = json.load(open(datafolder+'databases/snapshotserengeti/oneclass/SnapshotSerengeti_Seasons_1_to_4_classification_train_test_split.json'))
im_id_to_im = {im['id']:im for im in data}

train = [i for i in data_split['train_ims'] if i in im_id_to_im]
trans_val = [i for i in data_split['val_ims'] if i in im_id_to_im]
trans_test = [i for i in data_split['test_ims'] if i in im_id_to_im]

print('train: ', len(train), ', val: ', len(trans_val), ' test: ', len(trans_test) )

print('Creating train tfrecords')
#dataset = json.load(open('/ai4efs/databases/snapshotserengeti/oneclass/SnapshotSerengeti_Seasons_1_to_4_tfrecord_format_valid_ims.json','r'))

dataset = [im_id_to_im[idx] for idx in trans_val]

invalid_jpeg = []
valid_dataset = []
for im in dataset:
    fn = im['filename']
    try:
        with tf.Graph().as_default():
            image_contents = tf.read_file(fn)
            image = tf.image.decode_jpeg(image_contents, channels=3)
            init_op = tf.initialize_all_tables()
            with tf.Session() as sess:
                sess.run(init_op)
                tmp = sess.run(image)
    except:
        invalid_jpeg.append(im['id'])
        continue
    valid_dataset.append(im)
print(len(valid_dataset))

json.dump(valid_dataset, open('/ai4efs/databases/snapshotserengeti/oneclass/SnapshotSerengeti_Seasons_1_to_4_tfrecord_format_valid_ims_val.json','w'))

'''
print(dataset[0])
num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
while num_shards % num_threads:
    num_shards += 1
print(num_shards)
failed_images = create(
  dataset=dataset,
  dataset_name="train",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=num_threads,
  store_images=True
)
'''
'''
print('Creating cis_val tfrecords')
dataset = [im_id_to_im[idx] for idx in cis_val]
num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
while num_shards % num_threads:
    num_shards += 1
failed_images = create(
  dataset=dataset,
  dataset_name="cis_val",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5,
  store_images=True
)
'''

#print('Creating trans_val tfrecords')
#dataset = [im_id_to_im[idx] for idx in trans_val]
dataset = valid_dataset
num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
while num_shards % num_threads:
    num_shards += 1
failed_images = create(
  dataset=dataset,
  dataset_name="trans_val",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5,
  store_images=True
)

'''
print('Creating cis_test tfrecords')
dataset = [im_id_to_im[idx] for idx in cis_test]
num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
while num_shards % num_threads:
    num_shards += 1
failed_images = create(
  dataset=dataset,
  dataset_name="cis_test",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5,
  store_images=True
)
'''
'''
print('Creating trans_test tfrecords')
dataset = [im_id_to_im[idx] for idx in trans_test]
num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
while num_shards % num_threads:
    num_shards += 1
failed_images = create(
  dataset=dataset,
  dataset_name="trans_test",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5,
  store_images=True
)
'''


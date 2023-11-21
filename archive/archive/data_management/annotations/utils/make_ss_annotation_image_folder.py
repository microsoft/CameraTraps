# 
# make_ss_annotation_image_folder.py
#
# Take a directory full of images with the very long filenames we give annotators:
#
# dataset[dataset_id].seq[sequence_id].frame[frame_number].img[img_id].extension
#
# ...along with a COCO-camera-traps database referring to those files, and:
#
# 1) Creates a new COCO-camera-traps database with the original filenames in them 
#    (copying the annotations)
#
# 2) Optionally creates a new directory with those images named according to the 
#    Snapshot Serengeti naming convention, including complete relative paths.
#
# See convert_imerit_json_to_coco_json to see how we get from the original annotation
# .json to a COCO-camera-traps database.
#

#%% Constants and imports

import json
import os
import re
import time
import humanfriendly
from shutil import copyfile

COPY_FILES = False
CHECK_IMAGE_EXISTENCE = False


#%% Configure files/paths

BASE_DIR = r'd:\temp\snapshot_serengeti_tfrecord_generation'
ANNOTATION_SET = 'imerit_batch7'
annotation_file = os.path.join(BASE_DIR,ANNOTATION_SET + '.json')
output_file = os.path.join(BASE_DIR,ANNOTATION_SET + '_renamed.json')
image_input_folder = os.path.join(BASE_DIR,'imerit_batch7_snapshotserengeti_2018.10.26','to_label')
image_output_folder = os.path.join(BASE_DIR,ANNOTATION_SET + '_images_renamed')

if not os.path.exists(image_output_folder):
    os.makedirs(image_output_folder)

assert(os.path.isdir(image_input_folder))
assert(os.path.isfile(annotation_file))


#%% Read the annotations (referring to the old filenames)

with open(annotation_file,'r') as f:
    data = json.load(f)

print('Finished reading {} annotations and {} images from input file {}'.format(
        len(data['annotations']),len(data['images']),annotation_file))


#%% Update filenames, optionally copying files

startTime = time.time()

# im = data['images'][0]

# For each image...
for im in data['images']:
    
    # E.g. datasetsnapshotserengeti.seqASG000001a.frame0.imgS1_B06_R1_PICT0008.JPG
    old_filename = im['file_name']
    old_path = os.path.join(image_input_folder,old_filename)
    
    if (CHECK_IMAGE_EXISTENCE):
        assert(os.path.isfile(old_path))

    # Find the image name, e.g. S1_B06_R1_PICT0008
    pat = r'img(S.*\.jpg)$'
    m = re.findall(pat, old_filename, re.M|re.I)
    assert(len(m) == 1)
    
    # Convert:
    #
    # S1_B06_R1_PICT0008.JPG
    #
    # ...to:
    # 
    # S1/B06/B06_R1/S1_B06_R1_PICT0008.JPG
    tokens = m[0].split('_')
    assert(len(tokens)==4)
    new_filename = tokens[0] + '/' + tokens[1] + '/' + tokens[1] + '_' + tokens[2] + '/' + m[0]
    im['file_name'] = new_filename
    
    new_path = os.path.join(image_output_folder,new_filename)
    
    if COPY_FILES:
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        copyfile(old_path,new_path)

# ...for each image        
elapsed = time.time() - startTime
print("Done processing {} files in {}".format(len(data['images']),
      humanfriendly.format_timespan(elapsed)))


#%% Write the revised database

json.dump(data, open(output_file,'w'))


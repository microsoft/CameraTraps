# 
# make_ss_annotation_image_folder.py
#
# Take a directory full of images with the very long filenames we give annotators:
#
# dataset[dataset_id].seq[sequence_id].frame[frame_number].img[img_id].extension
#
# ...along with the annotation file we get back from the annotators, and does two things:
#
# 1) Creates a new directory with those images named according to the Snapshot Serengeti naming convention,
#    though not including full paths (only the terminal part of the filename)
#
# 2) Creates a new COCO-camera-traps database with the original filenames in them (copying the annotations)
#

#%% Constants and imports

import json
import os
from shutil import copyfile

COPY_FILES = False


#%% Configure files/paths

BASE_DIR = r'd:\temp\snapshot_serengeti_tfrecord_generation'
ANNOTATION_SET = 'microsoft_batch7_12Nov2018'
annotation_file = os.path.join(BASE_DIR,ANNOTATION_SET + '.json')
output_file = os.path.join(BASE_DIR,ANNOTATION_SET + '_renamed.json')
image_input_folder = os.path.join(BASE_DIR,'imerit_batch7_snapshotserengeti_2018.10.26','to_label')
image_output_folder = os.path.join(BASE_DIR,ANNOTATION_SET + '_images_renamed')

# Previous configurations
if False:
    output_file = 'C:/Users/t-sabeer/Documents/databases/imerit_annotation_images_ss_1_new_filenames.json'
    image_input_folder = 'D:/snapshot_serengeti/'
    image_output_folder = 'D:/imerit_annotation_images_ss_1/images/'
    annotation_file = 'C:/Users/t-sabeer/Documents/databases/imerit_annotation_images_ss_1.json'

if not os.path.exists(image_output_folder):
    os.makedirs(image_output_folder)

assert(os.path.isdir(image_input_folder))
assert(os.path.isfile(annotation_file))


#%% Read the annotations (bounding boxes)

with open(annotation_file,'r') as f:
    data = json.load(f)


print(len(data['images']))
not_found = []
for im in data['images']:
    old_filename = im['file_name'].split('/')
    old_image_name = old_filename[-1].split('_')[-1]
    filename = ''
    for chunk in old_filename[:-1]:
        filename += chunk + '/'
    filename += old_image_name
    if os.path.isfile(image_input_folder+filename):
        new_filename = im['seq_id']+'_'+str(im['frame_num'])+'_'+im['file_name'].split('/')[-1]
        if COPY_FILES:
            copyfile(image_input_folder+filename,image_output_folder+new_filename)
        im['file_name'] = new_filename
    else: 
        not_found.append(im['id'])
print(len(not_found))

data['images'] = [im for im in data['images'] if im['id'] not in not_found]
data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] not in not_found]

json.dump(data, open(output_file,'w'))

print(data['images'][0])

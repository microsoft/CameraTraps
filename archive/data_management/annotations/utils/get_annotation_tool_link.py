#
# get_annotation_tool_link.py
#
# Takes a COCO-camera-traps-style .json file with URLs already embedded, and prepares
# a link to the visipedia annotation tool that reviews a subset of those images.
#

import json
import pickle

datafile = '/datadrive/iwildcam/imerit/imerit_iwildcam_annotations_2.json'

with open(datafile,'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']

im_to_anns = {im['id']:[] for im in images}
im_id_to_seq_id = {im['id']:im['seq_id'] for im in images}
seq_id_to_seq = {seqId:[] for seqId in im_id_to_seq_id.values()}
for im in images:
    seqID = im['seq_id']
    if len(seq_id_to_seq[seqID]) == 0:
        seq_id_to_seq[seqID] = ['' for i in range(im['seq_num_frames'])]
        if im['frame_num'] <= im['seq_num_frames']:
            seq_id_to_seq[seqID][im['frame_num']-1] = im['id']
        else:
            print(im['frame_num'], im['seq_num_frames'])

for ann in annotations:
    im_to_anns[ann['image_id']].append(ann)

labeled_1000_images = []

for ann in annotations:
    if ann['category_id'] == 1000:
        labeled_1000_images.append(ann['image_id'])
all_seq_images_1000 = []
for im in labeled_1000_images:
    all_seq_images_1000 += [i for i in seq_id_to_seq[im_id_to_seq_id[im]] if len(i) > 0]

images_to_check = []
for i in all_seq_images_1000:
    if i not in images_to_check:
        images_to_check.append(i)

link = 'http://capybaravm.westus2.cloudapp.azure.com:8008/edit_task/?image_ids='

for im in images_to_check[:100]:
    link += im+','
print(link)
















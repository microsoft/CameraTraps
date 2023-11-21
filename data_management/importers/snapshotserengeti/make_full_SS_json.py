#
# make_full_SS_json.py
#
# Create a COCO-camera-traps .json file for Snapshot Serengeti data from
# the original .csv files provided on Dryad.
#
# This was used for "version 1.0" of the public Snapshot Serengeti archive; it's no 
# longer used as of v2.0 (early 2020).  See snapshot_serengeti_lila.py for the updated
# Snapshot Safari preparation process.
#

#%% Imports and constants

import csv
import json
import uuid
import datetime

output_file = '/datadrive/snapshotserengeti/databases/SnapshotSerengeti_multiple_classes.json'
csv_file_name = '/datadrive/snapshotserengeti/databases/consensus_data.csv'
all_image_file = '/datadrive/snapshotserengeti/databases/all_images.csv'


#%% Read annotation .csv file, format into a dictionary mapping field names to data arrays

data = []
with open(csv_file_name,'r') as f:
    reader = csv.reader(f, dialect = 'excel')
    for row in reader:
        data.append(row)

data_fields = data[0]

data_dicts = {}
for event in data[1:]:
    if event[0] not in data_dicts:
        data_dicts[event[0]] = []
    data_dicts[event[0]].append({data_fields[i]:event[i] for i in range(len(data_fields))})

# Count the number of images with multiple species
mult_species = 0
for event in data_dicts:
    if len(data_dicts[event]) > 1:
        mult_species += 1


#%% Read image .csv file, format into a dictionary mapping images to capture events

with open(all_image_file,'r') as f:
    reader = csv.reader(f,dialect = 'excel')
    next(reader)
    im_name_to_cap_id = {row[1]:row[0] for row in reader}
    
total_ims = len(im_name_to_cap_id)
total_seqs = len(data_dicts)
print('Percent seqs with mult species: ',float(mult_species)/float(total_seqs))


#%% Create CCT-style .json 

images = []
annotations = []
categories = []

capture_ims = {i:[] for i in im_name_to_cap_id.values()}
for im_id in im_name_to_cap_id:
    capture_ims[im_name_to_cap_id[im_id]].append(im_id)

im_to_seq_num = {im:None for im in im_name_to_cap_id}
for event in capture_ims:
    capture_ims[event] = sorted(capture_ims[event])
    seq_count = 0
    for im in capture_ims[event]:
        im_to_seq_num[im] = seq_count
        seq_count += 1

cat_to_id = {}
cat_to_id['empty'] = 0
cat_count = 1
seasons = []

for im_id in im_name_to_cap_id:
    
    im = {}
    im['id'] = im_id.split('.')[0]
    im['file_name'] = im_id
    
    im['location'] = im_id.split('/')[1]
    im['season'] = im_id.split('/')[0]
    if im['season'] not in seasons:
        seasons.append(im['season'])
    im['seq_id'] = im_name_to_cap_id[im_id]
    im['frame_num'] = im_to_seq_num[im_id]
    im['seq_num_frames'] = len(capture_ims[im['seq_id']])

    ann = {}
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']

    if im_name_to_cap_id[im_id] in data_dicts:
        im_data_per_ann = data_dicts[im_name_to_cap_id[im_id]]
        for im_data in im_data_per_ann:
            im['datetime'] = im_data['DateTime']
            if im_data['Species'] not in cat_to_id:
                cat_to_id[im_data['Species']] = cat_count
                cat_count += 1
            ann = {}
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = im['id']
            ann['category_id'] = cat_to_id[im_data['Species']]
            annotations.append(ann)
    else:
        ann = {}
        ann['id'] = str(uuid.uuid1())
        ann['image_id'] = im['id']
        ann['category_id'] = 0
        annotations.append(ann)
            
    # still need image width and height
    images.append(im)
    
# ...for each image
    
print(seasons)

for cat in cat_to_id:
    new_cat = {}
    new_cat['id'] = cat_to_id[cat]
    new_cat['name'] = cat
    categories.append(new_cat) 


#%% Write output files
    
json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
info = {}
info['year'] = 2018
info['version'] = 1
info['description'] = 'COCO style Snapshot Serengeti database'
info['contributor'] = 'SMB'
info['date_created'] = str(datetime.date.today())
json_data['info'] = info
json.dump(json_data,open(output_file,'w'))

print(images[0])
print(annotations[0]) 


#
# make_per_season_SS_json.py
#
# Create a COCO-camera-traps .json file for each Snapshot Serengeti season from 
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

output_file_folder = 'C:/Users/t-sabeer/Documents/databases/'
csv_file_name = 'D:/consensus_data.csv'


#%% Read annotation .csv file, format into a dictionary mapping field names to data arrays

data = []
with open(csv_file_name,'r') as f:
    reader = csv.reader(f, dialect = 'excel')
    for row in reader:
        data.append(row)

data_fields = data[0]

data_dicts = {}
for event in data[1:]:
    data_dicts[event[0]] = {data_fields[i]:event[i] for i in range(len(data_fields))}


#%% Read image .csv file, format into a dictionary mapping images to capture events
    
all_image_file = 'D:/all_images.csv'
with open(all_image_file,'r') as f:
    reader = csv.reader(f,dialect = 'excel')
    next(reader)
    im_name_to_cap_id = {row[1]:row[0] for row in reader}


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
    im['seq_id'] = im_name_to_cap_id[im_id]
    im['frame_num'] = im_to_seq_num[im_id]
    im['seq_num_frames'] = len(capture_ims[im['seq_id']])
    if im['season'] not in seasons:
        seasons.append(im['season'])

    ann = {}
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']

    if im_name_to_cap_id[im_id] in data_dicts:
        im_data = data_dicts[im_name_to_cap_id[im_id]]
        im['datetime'] = im_data['DateTime']
        if im_data['Species'] not in cat_to_id:
            cat_to_id[im_data['Species']] = cat_count
            cat_count += 1
        ann['category_id'] = cat_to_id[im_data['Species']]
    else:
        ann['category_id'] = 0
            
    #still need image width and height
    images.append(im)
    annotations.append(ann)

# ...for each image ID
    
for cat in cat_to_id:
    new_cat = {}
    new_cat['id'] = cat_to_id[cat]
    new_cat['name'] = cat
    categories.append(new_cat) 


#%% Write output files
    
output_file = output_file_folder + 'SnapshotSerengeti.json'
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

for season in seasons:
    
    output_file = output_file_folder + season + '.json'
    inSeason = {im['id']:False for im in images}
    for im in images:
        if im['season'] == season:
            inSeason[im['id']] = True
    new_ims = [im for im in images if inSeason[im['id']]]
    new_anns = [ann for ann in annotations if inSeason[ann['image_id']]]

    json_data = {}
    json_data['images'] = new_ims
    json_data['annotations'] = new_anns
    json_data['categories'] = categories
    info = {}
    info['year'] = 2018
    info['version'] = 1
    info['description'] = 'COCO style Snapshot Serengeti database. season ' + season
    info['contributor'] = 'SMB'
    info['date_created'] = str(datetime.date.today())
    json_data['info'] = info
    json.dump(json_data,open(output_file,'w'))
    
    print('Season ' + season)
    print(str(len(new_ims)) + ' images')
    print(str(len(new_anns)) + ' annotations') 
    
# ...for each season    
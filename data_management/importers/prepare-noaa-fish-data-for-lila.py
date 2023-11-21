#%% Constants and imports

import os
import json
import uuid
import pandas as pd

from path_utils import open_file

base_folder = r'G:\temp\noaa'
output_json_fn = os.path.join(base_folder,'noaa_estuary_fish.json')
edited_image_folders = ['edited_clip_2017','edited_clip_2018']
jpeg_image_folder = 'JPEGImages'
metadata_file = 'MasterDataForMicrosoft.xlsx'


#%% Enumerate files

edited_image_files = []

# edited_image_folder = edited_image_folders[0]
for edited_image_folder in edited_image_folders:
    folder_path = os.path.join(base_folder,edited_image_folder)
    image_files = os.listdir(folder_path)
    assert all([fn.endswith('.jpg') for fn in image_files])
    edited_image_files.extend([os.path.join(folder_path,fn) for fn in image_files])
    
jpeg_image_folder_files = os.listdir(os.path.join(base_folder,jpeg_image_folder))
assert all([fn.endswith('.jpg') for fn in jpeg_image_folder_files])

relative_edited_image_files_set = set()

# fn = edited_image_files[0]
for fn in edited_image_files:
    bn = os.path.basename(fn)
    assert bn not in relative_edited_image_files_set
    relative_edited_image_files_set.add(bn)
    
jpeg_image_folder_files_set = set(jpeg_image_folder_files)

assert len(jpeg_image_folder_files_set) == len(relative_edited_image_files_set)

assert jpeg_image_folder_files_set == relative_edited_image_files_set


#%% Read metadata and capture location information

df = pd.read_excel(os.path.join(base_folder,metadata_file))

print('Read {} rows from metadata file'.format(len(df)))

id_string_to_site = {}

# i_row = 0; row = df.iloc[i_row]
for i_row,row in df.iterrows():
    
    assert row['sd'].lower().startswith('sd')
    assert isinstance(row['id'],int) and row['id'] > 0 and row['id'] < 10000
    date_string = row['date']
    date_tokens = date_string.split('_')
    
    # Sometimes '2017' was just '17' in the date column
    if len(date_tokens[2]) != 4:
        assert len(date_tokens[2]) == 2
        date_tokens[2] = '20' + date_tokens[2]
        date_string = '_'.join(date_tokens)
    else:
        assert date_tokens[2].startswith('201')
        
    id_string = row['sd'].upper() + '_' + str(row['id']) + '_' + date_string
    id_string_to_site[id_string] = row['site']
    
print('Found {} unique locations'.format(len(pd.unique(df['site']))))


#%% Read the .json files and build output dictionaries

json_files = [fn for fn in os.listdir(base_folder) if (fn.endswith('.json') and (fn != os.path.basename(output_json_fn)))]
json_files = [os.path.join(base_folder,fn) for fn in json_files]

fn_to_image = {}
annotations = []

CATEGORY_ID_EMPTY = 0
CATEGORY_ID_FISH = 1

categories = [{'id':CATEGORY_ID_EMPTY,'name':'empty'},{'id':CATEGORY_ID_FISH,'name':'animal'}]

empty_images = set()
non_empty_images = set()

n_matched_locations = 0
images_with_unmatched_locations = []

import random
random.seed(1)

site_to_location_id = {}

# json_fn = json_files[0]
for json_fn in json_files:
    
    # if 'partial' in json_fn:
    #    continue
    
    with open(json_fn,'r') as f:
        
        lines = f.readlines()
        
        # line = lines[0]
        for line in lines:
            
            d = json.loads(line)
            image_fn = d['image']
        
            # if image_fn == 'SD1_238_6_26_17_16_76.73.jpg':
            #    asdfad
                
            # SD29_079_5_14_2018_17_52.85.jpg
            
            tokens = image_fn.split('_')
            assert len(tokens) == 7
            assert tokens[0].startswith('SD')
            
            # Re-write two-digit years as four-digit years
            if len(tokens[4]) != 4:
                assert len(tokens[4]) == 2
                tokens[4] = '20' + tokens[4]
            else:
                assert tokens[4].startswith('201')                
            
            # Sometimes the year was written with two digits instead of 4
            # assert len(tokens[4]) == 4 and tokens[4].startswith('20')
            
            while tokens[1].startswith('0'):
                tokens[1] = tokens[1][1:]
            assert not tokens[1].startswith('0')
            assert len(tokens[1]) > 0

            id_string = '_'.join(tokens[0:5])
            
            location_id = 'unknown'
                        
            if id_string in id_string_to_site:
                
                site_id = id_string_to_site[id_string]
                
                # Have we seen this location already?
                if site_id in site_to_location_id:
                    location_id = site_to_location_id[site_id]
                else:
                    location_id = 'loc_' + str(uuid.uuid1())
                    site_to_location_id[site_id] = location_id
                    print('Adding new location ID {} for site {}'.format(
                        location_id,site_id))                    
                n_matched_locations += 1
                
            else:
                raise ValueError('Could not match location ID')
                images_with_unmatched_locations.append(image_fn)
            
            assert image_fn in jpeg_image_folder_files_set
            assert d['type'] == 'image/jpg'
            input_ann = d['annotations']
            assert len(input_ann) == 1 and len(input_ann.keys()) == 1 and 'object' in input_ann
            input_ann = input_ann['object']
            assert input_ann['metainfo']['image']['height'] == 1080
            assert input_ann['metainfo']['image']['width'] == 1920
        
            im = {}
                        
            img_h = input_ann['metainfo']['image']['height']
            img_w = input_ann['metainfo']['image']['width']
            
            im['width'] = img_w
            im['height'] = img_h
            im['file_name'] = image_fn
            
            if image_fn in fn_to_image:
                assert fn_to_image[image_fn]['file_name'] == image_fn
                assert fn_to_image[image_fn]['width'] == img_w
                assert fn_to_image[image_fn]['height'] == img_h                
                im = fn_to_image[image_fn]
            else:
                fn_to_image[image_fn] = im
                im['location'] = location_id
                im['id'] = image_fn # str(uuid.uuid1())                
            
            # Not a typo, it's actually "formateddata"
            formatted_data = input_ann['formateddata']
            if len(formatted_data) == 0:
                
                # An image shouldn't be annotated as both empty and non-empty
                assert image_fn not in non_empty_images
                empty_images.add(image_fn)                
                ann = {}
                ann['id'] = str(uuid.uuid1())
                ann['image_id'] = im['id']
                ann['category_id'] = CATEGORY_ID_EMPTY
                ann['sequence_level_annotation'] = False                
                annotations.append(ann)
                
            else:
                
                # An image shouldn't be annotated as both empty and non-empty
                assert image_fn not in empty_images
                non_empty_images.add(image_fn)                
                
                n_boxes = len(formatted_data)
                
                # box = formatteddata[0]
                for box in formatted_data:
                                        
                    attributes = box['attribute']
                    assert len(attributes) == 2 and 'occluded' in attributes and 'truncated' in attributes                    
                    coordinates = box['coordinates']
                    assert box['object_type'] == 'bbox'
                    assert box['class']['type'] == 'Fish'
                    assert len(coordinates) == 4
                    for coord in coordinates:
                        assert len(coord) == 2 and 'x' in coord and 'y' in coord
                    assert coordinates[0]['y'] == coordinates[1]['y']
                    assert coordinates[2]['y'] == coordinates[3]['y']
                    assert coordinates[0]['x'] == coordinates[3]['x']
                    assert coordinates[1]['x'] == coordinates[2]['x']
                          
                    assert coordinates[0]['x'] < coordinates[1]['x']
                    assert coordinates[0]['y'] < coordinates[3]['y']
                    
                    if False:
                        x = coordinates[0]['x'] / img_w
                        y = coordinates[0]['y'] / img_h
                        box_w = (coordinates[1]['x'] - coordinates[0]['x']) / img_w
                        box_h = (coordinates[3]['y'] - coordinates[0]['y']) / img_h
                    else:
                        x = coordinates[0]['x']
                        y = coordinates[0]['y']
                        box_w = (coordinates[1]['x'] - coordinates[0]['x'])
                        box_h = (coordinates[3]['y'] - coordinates[0]['y'])
                        
                    bbox = [x,y,box_w,box_h]
                    
                    ann = {}
                    ann['id'] = str(uuid.uuid1())
                    ann['image_id'] = im['id']
                    ann['category_id'] = CATEGORY_ID_FISH
                    ann['sequence_level_annotation'] = False
                    ann['bbox'] = bbox
                    
                    annotations.append(ann)
                                        
                    # open_file(os.path.join(base_folder,jpeg_image_folder,image_fn))
                    
                # ...for each box
                
            # ...if there are boxes on this image
        
        # ...for each line
        
    # ...with open()
    
# ...for each json file
        
print('Found annotations for {} images (of {})'.format(len(fn_to_image),
                                                       len(jpeg_image_folder_files_set)))
                

print('Matched locations for {} images (failed to match {})'.format(
    n_matched_locations,len(images_with_unmatched_locations)))

images = list(fn_to_image.values())


#%% Prepare the output .json

info = {}
info['version'] = '2022.07.31.00'
info['description'] = 'NOAA Estuary Fish 2022'
info['year'] = 2022
info['contributor'] = 'NOAA Fisheries'
    
d = {}
d['info'] = info
d['annotations'] = annotations
d['images'] = images
d['categories'] = categories

with open(output_json_fn,'w') as f:
    json.dump(d,f,indent=1)


#%% Check DB integrity

from data_management.databases import integrity_check_json_db

options = integrity_check_json_db.IntegrityCheckOptions()
options.baseDir = os.path.join(base_folder,jpeg_image_folder)
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True

_, _, _ = integrity_check_json_db.integrity_check_json_db(output_json_fn, options)


#%% Print unique locations

from collections import defaultdict
location_to_count = defaultdict(int)
for im in d['images']:
    location_to_count[im['location']] += 1
for loc in location_to_count.keys():
    print(loc + ': ' + str(location_to_count[loc]))

print('{} unique locations'.format(len(location_to_count)))
assert 'unknown' not in location_to_count.keys()

# SD12_202_6_23_2017_1_31.85.jpg


#%% Preview some images

from visualization import visualize_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 10000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.include_filename_links = True

html_output_file, _ = visualize_db.process_images(db_path=output_json_fn,
                                                         output_dir=os.path.join(base_folder,'preview'),
                                                         image_base_dir=os.path.join(base_folder,jpeg_image_folder),
                                                         options=viz_options)
open_file(html_output_file)


#%% Statistics

print('Empty: {}'.format(len(empty_images)))
print('Non-empty: {}'.format(len(non_empty_images)))

images_with_no_boxes = 0
n_boxes = 0
for ann in annotations:
    if 'bbox' not in ann:
        images_with_no_boxes += 1
    else:
        assert len(bbox) == 4
        n_boxes += 1

print('N boxes: {}'.format(n_boxes))

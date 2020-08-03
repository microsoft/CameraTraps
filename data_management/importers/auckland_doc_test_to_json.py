#
# auckland_doc_test_to_json.py
#
# Convert Auckland DOC data set to COCO camera traps format.  This was
# for a testing data set where a .csv file was provided with class
# information.
#

#%% Constants and imports

import json
import os
import uuid
import pandas as pd
import datetime
import ntpath
import re
import numpy as np
from tqdm import tqdm

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
from path_utils import find_images, split_path, insert_before_extension

input_base_dir = r'e:\auckland-test\2_Testing'

input_metadata_file = r'G:\auckland-doc\Maukahuka - Auckland Island - Cat camera data Master April 2019 - DOC-5924483.xlsx'

# Filenames will be stored in the output .json relative to this base dir
output_base_dir = r'g:\auckland-doc'
output_json_filename = os.path.join(output_base_dir, 'auckland-doc-test.json')

assert os.path.isdir(input_base_dir)
os.makedirs(output_base_dir,exist_ok=True)

output_encoding = 'utf-8'
read_image_sizes = True

info = {}
info['year'] = 2020
info['version'] = '1.0'
info['description'] = 'Auckaland DOC Camera Traps (test)'
info['contributor'] = 'Auckland DOC'
info['date_created'] = str(datetime.date.today())


#%% Enumerate files

print('Enumerating files from {}'.format(input_base_dir))
absolute_image_paths = find_images(input_base_dir, recursive=True)
print('Enumerated {} images'.format(len(absolute_image_paths)))

relative_image_paths = []
for fn in absolute_image_paths:
    relative_image_paths.append(os.path.relpath(fn,input_base_dir).replace('\\','/'))

relative_image_paths_set = set(relative_image_paths)

assert len(relative_image_paths_set) == len(relative_image_paths)

# The excel file uses only filenames, not full paths; store just the filename.  
# 
# We store relative paths as cameraname_filename
camera_filename_to_relative_path = {}
camera_names = set()

# relative_path = relative_image_paths[0]
for relative_path in relative_image_paths:
    
    # Example relative path:
    #
    # relative_path = 'Summer_Trial_2019/A1_1_42_SD114_20190210/AucklandIsland_A1_1_42_SD114_20190210_01300001.jpg'
    fn = ntpath.basename(relative_path)    
    
    # Find the camera name
    tokens = relative_path.split('/')
    camera_token = tokens[1]
    camera_name = None
    m = re.search('^(.+)_SD',camera_token)
    if m:
        camera_name = m.group(1)
    else:
        # For camera tokens like C1_5_D_190207
        m = re.search('^(.+_.+_.+)_',camera_token)
        camera_name = m.group(1)
    
    assert camera_name
    camera_filename = camera_name + '_' + fn
    assert camera_filename not in camera_filename_to_relative_path
    camera_filename_to_relative_path[camera_filename] = relative_path
    

#%% Load input data

input_metadata = pd.read_excel(input_metadata_file)

print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns),
      len(input_metadata)))


#%% Assemble dictionaries

images = []
image_id_to_image = {}
annotations = []
categories = []

category_name_to_category = {}

# Force the empty category to be ID 0
empty_category = {}
empty_category['name'] = 'empty'
empty_category['id'] = 0
category_name_to_category['empty'] = empty_category
categories.append(empty_category)

filenames_not_in_folder = []
image_id_to_rows = {}

next_id = 1

category_names = ['cat','mouse','unknown','human','pig','sealion','penguin','dog']

# array([nan, 'Blackbird', 'Bellbird', 'Tomtit', 'Song thrush', 'Pippit',
#       'Pippet', '?', 'Dunnock', 'Song thursh', 'Kakariki', 'Tui', ' ',
#       'Silvereye', 'NZ Pipit', 'Blackbird and Dunnock', 'Unknown',
#       'Pipit', 'Songthrush'], dtype=object)

bird_names = input_metadata.Bird_ID.unique()
for bird_name in bird_names:
    if isinstance(bird_name,float):
        continue
    bird_name = bird_name.lower().strip().replace(' ','_').replace('song_thursh','song_thrush')
    if bird_name == '?' or bird_name == '' or bird_name == 'unknown':
        category_name = 'unknown_bird'
    else:
        category_name = bird_name
    if category_name not in category_names:
        category_names.append(category_name)
        
for category_name in category_names:
    cat = {}
    cat['name'] = category_name
    cat['id'] = next_id
    next_id = next_id +1
    category_name_to_category[category_name] = cat
    
def create_annotation(image_id,category_name,count):
    ann = {}    
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = image_id
    category = category_name_to_category[category_name]
    category_id = category['id']
    ann['category_id'] = category_id
    ann['count'] = count
    return ann
    
# i_row = 0; row = input_metadata.iloc[i_row]
for i_row,row in input_metadata.iterrows():

    # E.g.: AucklandIsland_A1_1_42_SD114_20190210_01300009.jpg
    filename = row['File']
    if not filename in filename_to_relative_path:
        filenames_not_in_folder.append(filename)
        continue        

    assert filename.endswith('.jpg') or filename.endswith('.JPG')
    image_id = filename.lower().replace('.jpg','')

    if image_id in image_id_to_rows:
        image_id_to_rows[image_id].append(i_row)
        # print('Warning: ignoring duplicate entry for {}'.format(image_id))
        continue
        
    image_id_to_rows[image_id] = [i_row]
        
    relative_path = filename_to_relative_path[filename]
    
    im = {}
    im['id'] = image_id 
    im['file_name'] = relative_path
    im['datetime'] = str(row['Date and time'])
    # Not a typo; the spreadsheet has a space after "Camera"
    im['camera'] = row['Camera ']
    im['sd_card'] = row['SD_Card']
    im['sd_change'] = row['SD_Change']
    im['comments'] = row['Comments']
    
    image_id_to_image[im['id']] = im
    
    images.append(im)
    
    if (not np.isnan(row['Cat'])):
       assert np.isnan(row['Collared_cat'] )
       ann = create_annotation(im['id'],'cat',row['Cat'])
    
    
#%%    

    if category_name not in category_name_to_category:        

        category_id = next_id
        next_id += 1
        category = {}
        category['id'] = category_id
        category['name'] = category_name
        category['count'] = 0
        categories.append(category)
        category_name_to_category[category_name] = category
        category_id_to_category[category_id] = category

    else:
        
        category = category_name_to_category[category_name]
        
    category_id = category['id']
    
    category['count'] += 1
    behavior = None
    if (category_name) != 'test':
        behavior = fn.split('\\')[-2]
        behaviors.add(behavior)            

    ann = {}
    
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']
    ann['category_id'] = category_id
    if behavior is not None:
        ann['behavior'] = behavior
    annotations.append(ann)

# ...for each image
    

#%% Write output .json

data = {}
data['info'] = info
data['images'] = images
data['annotations'] = annotations
data['categories'] = categories

json.dump(data, open(output_json_filename, 'w'), indent=2)
print('Finished writing json to {}'.format(output_json_filename))


#%% Write train/test .jsons

train_images = []; test_images = []
train_annotations = []; test_annotations = []

for ann in tqdm(annotations):
    category_id = ann['category_id']
    image_id = ann['image_id']
    category_name = category_id_to_category[category_id]['name']
    im = image_id_to_image[image_id]
    if category_name == 'test':
        test_images.append(im)
        test_annotations.append(ann)
    else:
        train_images.append(im)
        train_annotations.append(ann)

train_fn = insert_before_extension(output_json_filename,'train')
test_fn = insert_before_extension(output_json_filename,'test')

data['images'] = train_images
data['annotations'] = train_annotations
json.dump(data, open(train_fn, 'w'), indent=2)

data['images'] = test_images
data['annotations'] = test_annotations
json.dump(data, open(test_fn, 'w'), indent=2)

    
#%% Validate .json files

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = input_base_dir
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True

sortedCategories, data = sanity_check_json_db.sanity_check_json_db(output_json_filename, options)
sortedCategories, data = sanity_check_json_db.sanity_check_json_db(train_fn, options)
sortedCategories, data = sanity_check_json_db.sanity_check_json_db(test_fn, options)


#%% Preview labels

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 2000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.classes_to_exclude = ['test']
html_output_file, image_db = visualize_db.process_images(db_path=output_json_filename,
                                                         output_dir=os.path.join(
                                                         output_base_dir, 'preview'),
                                                         image_base_dir=input_base_dir,
                                                         options=viz_options)
os.startfile(html_output_file)

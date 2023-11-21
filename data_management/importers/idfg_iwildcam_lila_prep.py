#
# idfg_iwildcam_lila_prep.py
#
# Adding class labels (from the private test .csv) to the iWildCam 2019 IDFG 
# test set, in preparation for release on LILA.
#
# This version works with the public iWildCam release images.
#

#%% ############ Take one, from iWildCam .json files ############ 

#%% Imports and constants

import uuid
import json
import os
from tqdm import tqdm

base_folder = r'h:\iWildCam_2019_IDFG'
input_json = os.path.join(base_folder,'iWildCam_2019_IDFG_info.json')
input_csv = os.path.join(base_folder,'IDFG_eval_public_v_private.csv')
output_json = os.path.join(base_folder,'idaho_camera_traps.json')

assert os.path.isfile(input_json)
assert os.path.isfile(input_csv)


#%% Read input files

with open(input_json,'r') as f:
    input_data = json.load(f)

with open(input_csv,'r') as f:
    private_csv_lines = f.readlines()

private_csv_lines = [s.strip() for s in private_csv_lines]
    
# Remove the header line    
assert private_csv_lines[0] == 'Id,Category,Usage'
private_csv_lines = private_csv_lines[1:]

print('Read {} annotations for {} images'.format(len(private_csv_lines),len(input_data['images'])))

assert len(private_csv_lines) == len(input_data['images'])
n_images = len(input_data['images'])


#%% Parse annotations

image_id_to_category_ids = {}
for line in tqdm(private_csv_lines):
    
    # Lines look like:
    #
    # b005e5b2-2c0b-11e9-bcad-06f1011196c4,1,Private
    
    tokens = line.split(',')
    assert len(tokens) == 3
    assert tokens[2] in ['Private','Public']
    image_id_to_category_ids[tokens[0]] = int(tokens[1])
    
assert len(image_id_to_category_ids) == n_images


#%% Minor cleanup re: images

for im in tqdm(input_data['images']):
    image_id = im['id']
    im['file_name'] = im['file_name'].replace('iWildCam_IDFG_images/','')
    assert isinstance(im['location'],int)
    im['location'] = str(im['location'])
    

#%% Create annotations

annotations = []

for image_id in tqdm(image_id_to_category_ids):
    category_id = image_id_to_category_ids[image_id]
    ann = {}
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = image_id
    ann['category_id'] = category_id
    annotations.append(ann)
    

#%% Prepare info

info = input_data['info']
info['contributor'] = 'Images acquired by the Idaho Department of Fish and Game, dataset curated by Sara Beery'
info['description'] = 'Idaho Camera traps'
info['version'] = '2021.07.19'


#%% Minor adjustments to categories

input_categories = input_data['categories']

category_id_to_name = {cat['id']:cat['name'] for cat in input_categories}
category_name_to_id = {cat['name']:cat['id'] for cat in input_categories}
assert category_id_to_name[0] == 'empty'

category_names_to_counts = {}
for category in input_categories:
    category_names_to_counts[category['name']] = 0
    
for ann in annotations:
    category_id = ann['category_id']
    category_name = category_id_to_name[category_id]
    category_names_to_counts[category_name] = category_names_to_counts[category_name] + 1
    
categories = []

for category_name in category_names_to_counts:
    count = category_names_to_counts[category_name]    
    
    # Remove unused categories
    if count == 0:
        continue
    
    category_id = category_name_to_id[category_name]
    
    # Name adjustments
    if category_name == 'prongs':
        category_name = 'pronghorn'
    
    categories.append({'id':category_id,'name':category_name})
    
    
#%% Create output

output_data = {}
output_data['images'] = input_data['images']
output_data['annotations'] = annotations
output_data['categories'] = categories
output_data['info'] = info


#%% Write output

with open(output_json,'w') as f:
    json.dump(output_data,f,indent=2)
    

#%% Validate .json file

from data_management.databases import sanity_check_json_db

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = os.path.join(base_folder,'images'); assert os.path.isdir(options.baseDir)
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False

_, _, _ = sanity_check_json_db.sanity_check_json_db(output_json, options)


#%% Preview labels

from visualization import visualize_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 100
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.include_filename_links = True

# viz_options.classes_to_exclude = ['test']
html_output_file, _ = visualize_db.process_images(db_path=output_json,
                                                         output_dir=os.path.join(
                                                         base_folder,'preview'),
                                                         image_base_dir=os.path.join(base_folder,'images'),
                                                         options=viz_options)
os.startfile(html_output_file)


#%% ############ Take two, from pre-iWildCam .json files created from IDFG .csv files ############ 

#%% Imports and constants

import json
import os

base_folder = r'h:\idaho-camera-traps'
input_json_sl = os.path.join(base_folder,'iWildCam_IDFG.json')
input_json = os.path.join(base_folder,'iWildCam_IDFG_ml.json')
output_json = os.path.join(base_folder,'idaho_camera_traps.json')
remote_image_base_dir = r'z:\idfg'

assert os.path.isfile(input_json)


#%% One-time line break addition

if not os.path.isfile(input_json):
    
    sl_json = input_json_sl
    ml_json = input_json
    
    with open(sl_json,'r') as f:
        d = json.load(f)
    with open(ml_json,'w') as f:
        json.dump(d,f,indent=2)
        
        
#%% Read input files

with open(input_json,'r') as f:
    input_data = json.load(f)

print('Read {} annotations for {} images'.format(len(input_data['annotations']),len(input_data['images'])))

n_images = len(input_data['images'])


#%% Prepare info

info = {}
info['contributor'] = 'Images acquired by the Idaho Department of Fish and Game, dataset curated by Sara Beery'
info['description'] = 'Idaho Camera traps'
info['version'] = '2021.07.19'


#%% Minor adjustments to categories

input_categories = input_data['categories']
output_categories = []

for c in input_categories:
    category_name = c['name']
    category_id = c['id']
    if category_name == 'prong':
        category_name = 'pronghorn'
    category_name = category_name.lower()
    output_categories.append({'name':category_name,'id':category_id})


#%% Minor adjustments to annotations

for ann in input_data['annotations']:
    ann['id'] = str(ann['id'])


#%% Create output

output_data = {}
output_data['images'] = input_data['images']
output_data['annotations'] = input_data['annotations']
output_data['categories'] = output_categories
output_data['info'] = info


#%% Write output

with open(output_json,'w') as f:
    json.dump(output_data,f,indent=2)
    

#%% Validate .json file

from data_management.databases import sanity_check_json_db

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = remote_image_base_dir
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False

_, _, _ = sanity_check_json_db.sanity_check_json_db(output_json, options)


#%% Preview labels

from visualization import visualize_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 100
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.include_filename_links = True

# viz_options.classes_to_exclude = ['test']
html_output_file, _ = visualize_db.process_images(db_path=output_json,
                                                         output_dir=os.path.join(
                                                         base_folder,'preview'),
                                                         image_base_dir=remote_image_base_dir,
                                                         options=viz_options)
os.startfile(html_output_file)

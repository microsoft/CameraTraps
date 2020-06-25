#
# save_the_elephants_survey_B.py
#
# Convert the .csv file provided for the Save the Elephants Survey B data set to a 
# COCO-camera-traps .json file
#

#%% Constants and environment

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
import pandas as pd
import os
import glob
import json
import uuid
import time
import humanfriendly
from PIL import Image
import numpy as np
import logging
from tqdm import tqdm

from path_utils import find_images

input_base = r'z:/ste_2019_08_drop'
# input_base = r'/mnt/blobfuse/wildlifeblobssc/ste_2019_08_drop'
input_metadata_file = os.path.join(input_base,'SURVEY B.xlsx')

output_base = r'f:/save_the_elephants/survey_b'
# output_base = r'/home/gramener/survey_b'
output_json_file = os.path.join(output_base,'ste_survey_b.json')
image_directory = os.path.join(input_base,'SURVEY B with False Triggers')
                        
os.makedirs(output_base,exist_ok=True)
assert(os.path.isdir(image_directory))
assert(os.path.isfile(input_metadata_file))

# Handle all unstructured fields in the source data as extra fields in the annotations
mapped_fields = {'No. of Animals in Photo':'num_animals',
                 'No. of new indiviauls (first sighting of new individual)':'num_new_individuals',
                 'Number Adult Males (first sighting of new individual)':'num_adult_males',
                 'Number Adult Females (first sighting of new individual)':'num_adult_females',
                 'Number Adult Unknown (first sighting of new individual)':'num_adult_unknown',
                 'Number Sub-adult Males (first sighting of new individual)':'num_subadult_males',
                 'Number Sub-adult Females (first sighting of new individual)':'num_subadult_females',
                 'Number Sub-adult Unknown (first sighting of new individual)':'num_subadult_unknown',
                 'Number Juvenile (first sighting of new individual)':'num_juvenile',
                 'Number Newborn (first sighting of new individual)':'num_newborn',
                 'Activity':'activity',
                 'Animal ID':'animal_id',
                 'Specific Notes':'notes'}

# photo_type really should be an image property, but there are a few conflicts
# that forced me to handle it as an annotation proprerty
mapped_fields['Photo Type '] = 'photo_type'

#%% Read source data

input_metadata = pd.read_excel(input_metadata_file, sheet_name='9. CT Image')
input_metadata = input_metadata.iloc[2:]

print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns),
      len(input_metadata)))


#%% Map filenames to rows, verify image existence

#%% Map filenames to rows, verify image existence

start_time = time.time()

# Maps relative paths to row indices in input_metadata
filenames_to_rows = {}
filenames_with_multiple_annotations = []
missing_images = []

# Build up a map from filenames to a list of rows, checking image existence as we go
for i_row, fn in tqdm(enumerate(input_metadata['Image Name']), total=len(input_metadata)):
    try:
        # Ignore directories
        if not fn.endswith('.JPG'):
            continue

        if fn in filenames_to_rows:
            filenames_with_multiple_annotations.append(fn)
            filenames_to_rows[fn].append(i_row)
        else:
            filenames_to_rows[fn] = [i_row]
            image_path = os.path.join(image_directory, fn)
            if not os.path.isfile(image_path):
                missing_images.append(image_path)
    except:
        continue

elapsed = time.time() - start_time

print('Finished verifying image existence for {} files in {}, found {} filenames with multiple labels, {} missing images'.format(
      len(filenames_to_rows), humanfriendly.format_timespan(elapsed),
      len(filenames_with_multiple_annotations), len(missing_images)))

#%% Make sure the multiple-annotation cases make sense

if False:

    #%%

    fn = filenames_with_multiple_annotations[1000]
    rows = filenames_to_rows[fn]
    assert(len(rows) > 1)
    for i_row in rows:
        print(input_metadata.iloc[i_row]['Species'])

#%% Check for images that aren't included in the metadata file

# Enumerate all images
image_full_paths = find_images(image_directory, bRecursive=True)

unannotated_images = []

for iImage, image_path in tqdm(enumerate(image_full_paths),total=len(image_full_paths)):
    relative_path = os.path.relpath(image_path,image_directory)
    if relative_path not in filenames_to_rows:
        unannotated_images.append(relative_path)

print('Finished checking {} images to make sure they\'re in the metadata, found {} unannotated images'.format(
        len(image_full_paths),len(unannotated_images)))


#%% Create CCT dictionaries

images = []
annotations = []
categories = []

image_ids_to_images = {}

category_name_to_category = {}

# Force the empty category to be ID 0
empty_category = {}
empty_category['name'] = 'empty'
empty_category['id'] = 0
category_name_to_category['empty'] = empty_category
categories.append(empty_category)
next_category_id = 1

start_time = time.time()
# i_image = 0; image_name = list(filenames_to_rows.keys())[i_image]
for image_name in tqdm(list(filenames_to_rows.keys())):

    # Example filename:
    #        
    # 'Site 1_Oloisukut_1\Oloisukut_A11_UP\Service_2\100EK113\EK001382.JPG'
    # 'Site 1_Oloisukut_1\Oloisukut_A11_UP\Service_2.1\100EK113\EK001382.JPG'
    img_id = image_name.replace('\\','/').replace('\n','').replace('/','_').replace(' ','_')
    
    row_indices = filenames_to_rows[image_name]
    
    # i_row = row_indices[0]
    for i_row in row_indices:
        
        row = input_metadata.iloc[i_row]
        assert(row['Image Name'] == image_name)
        try:
            timestamp = row['Date'].strftime("%d/%m/%Y")
        except:
            timestamp = ""
        # timestamp = row['Date']
        station_label = row['Camera Trap Station Label']
        photo_type = row['Photo Type ']
        if isinstance(photo_type,float):
            photo_type = ''
        photo_type = photo_type.strip().lower()
            
        if img_id in image_ids_to_images:
            
            im = image_ids_to_images[img_id]
            assert im['file_name'] == image_name
            assert im['station_label'] == station_label
            
            # There are a small handful of datetime mismatches across annotations
            # for the same image
            # assert im['datetime'] == timestamp
            if im['datetime'] != timestamp:
                print('Warning: timestamp conflict for image {}: {},{}'.format(
                    image_name,im['datetime'],timestamp))
                
        else:
            
            im = {}
            im['id'] = img_id
            im['file_name'] = image_name
            im['datetime'] = timestamp
            im['station_label'] = station_label
            im['photo_type'] = photo_type
            
            image_ids_to_images[img_id] = im
            images.append(im)
    
        species = row['Species']
        
        if (isinstance(species,float) or \
            (isinstance(species,str) and (len(species) == 0))):
            category_name = 'empty'
        elif species.startswith('?')
            category_name = 'unknown'
        else:
            category_name = species
        
        # Special cases based on the 'photo type' field
        if 'vehicle' in photo_type:
            category_name = 'vehicle'
        # Various spellings of 'community'
        elif 'comm' in photo_type:
            category_name = 'human'
        elif 'camera' in photo_type or 'researcher' in photo_type:
            category_name = 'human'
        elif 'livestock' in photo_type:
            category_name = 'livestock'
        elif 'blank' in photo_type:
            category_name = 'empty'
        elif 'plant movement' in photo_type:
            category_name = 'empty'
            
        category_name = category_name.strip().lower()
            
        # Have we seen this category before?
        if category_name in category_name_to_category:
            category_id = category_name_to_category[category_name]['id'] 
        else:
            category_id = next_category_id
            category = {}
            category['id'] = category_id
            category['name'] = category_name
            category_name_to_category[category_name] = category
            categories.append(category)
            next_category_id += 1
        
        # Create an annotation
        ann = {}        
        ann['id'] = str(uuid.uuid1())
        ann['image_id'] = im['id']    
        ann['category_id'] = category_id
        
        # fieldname = list(mapped_fields.keys())[0]
        for fieldname in mapped_fields:
            target_field = mapped_fields[fieldname]
            val = row[fieldname]
            if isinstance(val,float) and np.isnan(val):
                val = ''
            else:
                val = str(val).strip()
            ann[target_field] = val
            
        annotations.append(ann)
        
    # ...for each row
                
# ...for each image
    
print('Finished creating CCT dictionaries in {}'.format(
      humanfriendly.format_timespan(elapsed)))
    

#%% Create info struct

info = {}
info['year'] = 2019
info['version'] = 1
info['description'] = 'Save the Elephants Survey B'
info['contributor'] = 'Save the Elephants'


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_json_file, 'w'), indent=2)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))


#%% Validate output

from data_management.databases import sanity_check_json_db

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = image_directory
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False
    
sortedCategories, data = sanity_check_json_db.sanity_check_json_db(output_json_file,options)


#%% Preview labels

from visualization import visualize_db
from data_management.databases import sanity_check_json_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 1000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file,image_db = visualize_db.process_images(db_path=output_json_file,
                                                        output_dir=os.path.join(output_base,'preview'),
                                                        image_base_dir=image_directory,
                                                        options=viz_options)
os.startfile(html_output_file)


#%% Scrap

if False:

    pass
    
    #%% Find unique photo types
    
    annotations = image_db['annotations']
    photo_types = set()
    for ann in tqdm(annotations):
        photo_types.add(ann['photo_type'])
#
# carrizo_trail_cam_2017.py
#
# Convert the .csv files provided for the "Trail Cam Carrizo" 2017 data set to 
# a COCO-camera-traps .json file.
#

#%% Constants and environment

import pandas as pd
import os
import json
import uuid
import time
import humanfriendly
from PIL import Image
import numpy as np

from tqdm import tqdm
from path_utils import find_images

input_base = r'Z:\Trail Cam Carrizo 2017'
open_metadata_file = os.path.join(input_base, 'Carrizo open 2017.csv')
shrub_metadata_file = os.path.join(input_base, 'Carrizo Shrub 2017.csv')

output_base = r'G:\carrizo-mojave'
output_json_file = os.path.join(output_base, 'carrizo trail cam 2017.json')
image_directory = input_base
input_metadata_files = [open_metadata_file, shrub_metadata_file]

load_width_and_height = False

assert(os.path.isdir(image_directory))

category_replacements = {'unidnetifiable':'unidentifiable','unidentifiable animal':'unidentifiable'}

annotation_fields_to_copy = ['rep','photo rep','timeblock','night.day','observations']


#%% Read source data

final_data = pd.DataFrame()

for inp_file in input_metadata_files:
    
    print("Reading: {0}".format(inp_file))
    input_metadata = pd.read_csv(inp_file)
    
    print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns), 
                                                                  len(input_metadata)))
    
    # Removing the empty records
    input_metadata = input_metadata[~np.isnan(input_metadata['rep'])]
    
    input_metadata['file'] = input_metadata.groupby(["rep", "week"]).cumcount()
    week_folder_format = {1: 'week 1- 2017', 2: 'week2- 2017', 3: 'week3-2017'}
    
    input_metadata['file'] = input_metadata[['file', 'rep', 'week', 'microsite']].apply(
        lambda x: "{3}/{4}{1}-week{0}-carrizo-2017/IMG_{2}.JPG".format(int(x[2]), int(x[1]), 
                                                                       str(int(x[0]+1)).zfill(4), 
                                                                       week_folder_format[int(x[2])], 
                                                                       x[3].lower()), axis=1)
    
    final_data = final_data.append(input_metadata)

print('Read {} metadata rows'.format(len(final_data)))

    
#%% Map filenames to rows, verify image existence
    
start_time = time.time()
filenames_to_rows = {}
image_filenames = input_metadata.file

missing_files = []
duplicate_rows = []

# Build up a map from filenames to a list of rows, checking image existence as we go
for iFile, fn in tqdm(enumerate(image_filenames),total=len(image_filenames)):
    if (fn in filenames_to_rows):
        duplicate_rows.append(iFile)
        filenames_to_rows[fn].append(iFile)
    else:
        filenames_to_rows[fn] = [iFile]
        image_path = os.path.join(image_directory, fn)
        if not os.path.isfile(image_path):
            missing_files.append(fn)

elapsed = time.time() - start_time

print('Finished verifying image existence in {}, found {} missing files (of {})'.format(
    humanfriendly.format_timespan(elapsed), 
    len(missing_files),len(image_filenames)))

assert len(duplicate_rows) == 0

# 908 missing files (of 60562)


#%% Check for images that aren't included in the metadata file

image_full_paths = find_images(image_directory, bRecursive=True)
images_missing_from_metadata = []

for iImage, image_path in tqdm(enumerate(image_full_paths),total=len(image_full_paths)):

    relative_path = os.path.relpath(image_path,input_base).replace('\\','/')
    if relative_path not in filenames_to_rows:
        images_missing_from_metadata.append(relative_path)
    
print('{} of {} files are not in metadata'.format(len(images_missing_from_metadata),len(image_full_paths)))

# 105329 of 164983 files are not in metadata


#%% Create CCT dictionaries

images = []
annotations = []

# Map categories to integer IDs
#
# The category '0' is reserved for 'empty'

categories_to_category_id = {}
categories_to_counts = {}
categories_to_category_id['empty'] = 0
categories_to_counts['empty'] = 0

next_category_id = 1

# For each image
#
# Because in practice images are 1:1 with annotations in this data set,
# this is also a loop over annotations.

start_time = time.time()

for image_name in tqdm(image_filenames):
    
    rows = filenames_to_rows[image_name]
    
    # Each filename should just match one row
    assert(len(rows) == 1)   
    
    iRow = rows[0]
    row = input_metadata.iloc[iRow]
    im = {}
    im['id'] = image_name.replace('\\','/').replace('/','_').replace(' ','_')
    im['file_name'] = image_name
    im['region'] = row['region']
    im['site']= row['site']
    im['mircosite'] = row['microsite']
    im['datetime'] = row['calendar date']
    im['location'] = "{0}_{1}_{2}".format(row['region'], row['site'], row['microsite'])
    
    image_path = os.path.join(image_directory, image_name)
    
    # Don't include images that don't exist on disk
    if not os.path.isfile(image_path):
        continue
    
    if load_width_and_height:
        pilImage = Image.open(image_path)
        width, height = pilImage.size
        im['width'] = width
        im['height'] = height
    else:
        im['width'] = -1
        im['height'] = -1
        
    images.append(im)
    
    is_image = row['animal.capture']
    
    if (is_image == 0):
        category = 'empty'
    else:
        if row['animal'] is np.nan:
            category = 'unidentifiable'
        else:
            category = row['animal'].strip()
    
    if category in category_replacements:
        category = category_replacements[category]
        
    # Have we seen this category before?
    if category in categories_to_category_id:
        categoryID = categories_to_category_id[category]
        categories_to_counts[category] += 1
    else:
        categoryID = next_category_id
        categories_to_category_id[category] = categoryID
        categories_to_counts[category] = 1
        next_category_id += 1
    
    # Create an annotation
    ann = {}
    
    # The Internet tells me this guarantees uniqueness to a reasonable extent, even
    # beyond the sheer improbability of collisions.
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']    
    ann['category_id'] = categoryID
    
    for fieldname in annotation_fields_to_copy:
        ann[fieldname] = row[fieldname]
        if ann[fieldname] is np.nan:
            ann[fieldname] = ''
        ann[fieldname] = str(ann[fieldname])
        
    annotations.append(ann)
    
# ...for each image
    
# Convert categories to a CCT-style dictionary

categories = []

for category in categories_to_counts:
    print('Category {}, count {}'.format(category,categories_to_counts[category]))
    categoryID = categories_to_category_id[category]
    cat = {}
    cat['name'] = category
    cat['id'] = categoryID
    categories.append(cat)    
    
elapsed = time.time() - start_time
print('Finished creating CCT dictionaries in {}'.format(
      humanfriendly.format_timespan(elapsed)))
    

#%% Create info struct

info = {}
info['year'] = 2017
info['version'] = 1
info['description'] = 'Carrizo Trail Cam 2017'
info['contributor'] = 'York University'


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_json_file, 'w'), indent=1)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))


#%% Validate output

from data_management.databases import sanity_check_json_db

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = image_directory
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False
data = sanity_check_json_db.sanity_check_json_db(output_json_file,options)


#%% Preview labels

from visualization import visualize_db
from data_management.databases import sanity_check_json_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = None
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.classes_to_exclude = ['empty']
html_output_file,image_db = visualize_db.process_images(db_path=output_json_file,
                                                        output_dir=os.path.join(
                                                        output_base, 'carrizo trail cam 2017/preview'),
                                                        image_base_dir=image_directory,
                                                        options=viz_options)
os.startfile(html_output_file)


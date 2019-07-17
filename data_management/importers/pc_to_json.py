#
# pc_to_json.py
#
# Convert a particular collection of .csv files to CCT format.
#

#%% Constants and environment

import pandas as pd
import uuid
import json
import time

import numpy as np
from tqdm import tqdm

import humanfriendly
import os
import PIL

from visualization import visualize_db
import path_utils

input_base = r"g:\20190715"
output_file = r"D:\wildlife_data\parks_canada\pc_20190715.json"
preview_base = r"D:\wildlife_data\parks_canada\preview"

filename_replacements = {}
category_mappings = {'':'unlabeled'}

csv_prefix = 'ImageData_Microsoft___'

expected_columns = 'Location,DateImage,TimeImage,Species,Total,Horses,DogsOnLeash,DogsOffLeash,AdultFemale,AdultMale,AdultUnknown,Subadult,YLY,YOY,ImageName'.split(',')
columns_to_copy = {'Total':'count','Horses':'horses','DogsOnLeash':'dogsonleash','DogsOffLeash':'dogsoffleash',
                   'AdultFemale':'adultfemale','AdultMale':'adultmale','AdultUnknown':'adultunknown',
                   'Subadult':'subadult','YLY':'yearling','YOY':'youngofyear'}

retrieve_image_sizes = False


#%% Read and concatenate source data

# List files
input_files = os.listdir(input_base)

# List of dataframes, one per .csv file; we'll concatenate later
all_input_metadata = []

# i_file = 87; fn = input_files[i_file]
for i_file,fn in enumerate(input_files):
    
    if not fn.endswith('.csv'):
        continue
    if not fn.startswith(csv_prefix):
        continue
    dirname = fn.replace(csv_prefix,'').replace('.csv','')
    dirfullpath = os.path.join(input_base,dirname)
    if not os.path.isdir(dirfullpath):
        dirname = fn.replace(csv_prefix,'').replace('.csv','').replace('  ',' ')
        dirfullpath = os.path.join(input_base,dirname)    
    assert(os.path.isdir(dirfullpath))
    
    metadata_fullpath = os.path.join(input_base,fn)
    print('Reading {}'.format(metadata_fullpath))
    df = pd.read_csv(metadata_fullpath)
    assert list(df.columns) == expected_columns
    df['DirName'] = dirname
    all_input_metadata.append(df)

# Concatenate into a giant data frame
input_metadata = pd.concat(all_input_metadata)

print('Read {} rows total'.format(len(input_metadata)))


#%% List files

print('Listing images...')
image_full_paths = path_utils.find_images(input_base,bRecursive=True)
print('Finished listing {} images'.format(len(image_full_paths)))

image_relative_paths = []
for s in image_full_paths:
    image_relative_paths.append(os.path.relpath(s,input_base))
image_relative_paths = set(image_relative_paths)

image_relative_paths_lower = set()
for s in image_relative_paths:
    image_relative_paths_lower.add(s.lower())
    
    
#%% Main loop over labels (prep)

start_time = time.time()

relative_path_to_image = {}

images = []
annotations = []
category_name_to_category = {}
missing_files = []

# Force the empty category to be ID 0
empty_category = {}
empty_category['name'] = 'empty'
empty_category['id'] = 0
category_name_to_category['empty'] = empty_category
next_category_id = 1


#%% Main loop over labels (loop)

# iRow = 0; row = input_metadata.iloc[iRow]
for iRow,row in tqdm(input_metadata.iterrows(),total=len(input_metadata)):
    
    # ImageID,FileName,FilePath,SpeciesID,CommonName
    image_id = str(uuid.uuid1())
    relative_path = os.path.normpath(row['ImageName'])
    
    if relative_path not in image_relative_paths:
        if relative_path.lower() in image_relative_paths_lower:
            print('Warning: lower-case version of {} in path list'.format(relative_path))
        else:
            missing_files.append(relative_path)
            continue
        
    full_path = os.path.join(input_base,relative_path)

    # assert os.path.isfile(full_path)    
 
    if relative_path in relative_path_to_image:
        
        im = relative_path_to_image[relative_path]

    else:
        
        im = {}
        im['id'] = image_id
        im['file_name'] = relative_path
        im['seq_id'] = '-1'
        im['location'] = row['Location']
        im['datetime'] = row['DateImage'] + ' ' + row['TimeImage']        
        
        images.append(im)
        relative_path_to_image[relative_path] = im
        
        if retrieve_image_sizes:
        
            # Retrieve image width and height
            pil_im = PIL.Image.open(full_path)
            width, height = pil_im.size
            im['width'] = width
            im['height'] = height

    species = row['Species']
    if isinstance(species,float):
        assert np.isnan(species)
        species = 'unlabeled'
    category_name = species.lower()
    if category_name in category_mappings:
        category_name = category_mappings[category_name]
        
    if category_name not in category_name_to_category:
        category = {}
        category['name'] = category_name
        category['id'] = next_category_id
        next_category_id += 1
        category_name_to_category[category_name] = category
    else:
        assert category_name_to_category[category_name]['name'] == category_name

    category_id = category['id']

    # Create an annotation
    ann = {}
    
    # The Internet tells me this guarantees uniqueness to a reasonable extent, even
    # beyond the sheer improbability of collisions.
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']    
    ann['category_id'] = category_id

    for col in columns_to_copy:
        ann[columns_to_copy[col]] = row[col]
        
    annotations.append(ann)
    
# ...for each image

categories = list(category_name_to_category.values())

elapsed = time.time() - start_time
print('Finished verifying file loop in {}, {} matched images, {} missing images'.format(
        humanfriendly.format_timespan(elapsed), len(images), len(missing_files))) 

#%%

dirnames = set()
# s = list(image_relative_paths)[0]
for s in image_relative_paths:
    image_dir = os.path.dirname(s)
    dirnames.add(image_dir)
        
n_missing_paths = 0

# s = missing_files[0]
for s in missing_files:
    assert s not in image_relative_paths
    dirname = os.path.dirname(s)
    if dirname not in dirnames:
        n_missing_paths += 1

print('Of {} missing files, {} are due to missing folders'.format(len(missing_files),n_missing_paths))    
    

#%% Check for images that aren't included in the metadata file

# Enumerate all images
# list(relative_path_to_image.keys())[0]

imageFullPaths = path_utils.find_images(image_base,bRecursive=True)
unmatchedFiles = []

for iImage,imagePath in enumerate(imageFullPaths):
    
    fn = os.path.relpath(imagePath,image_base)    
    if fn not in relative_path_to_image:
        unmatchedFiles.append(fn)

print('Finished checking {} images to make sure they\'re in the metadata, found {} mismatches'.format(
        len(imageFullPaths),len(unmatchedFiles)))


#%% Create info struct

info = {}
info['year'] = 2019
info['version'] = 1
info['description'] = 'COCO style database'
info['secondary_contributor'] = 'Converted to COCO .json by Dan Morris'
info['contributor'] = ''


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_file,'w'), indent=4)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))


#%% Sanity-check the database's integrity

from data_management.databases import sanity_check_json_db

options = sanity_check_json_db.SanityCheckOptions()
sortedCategories,data = sanity_check_json_db.sanity_check_json_db(output_file, options)

    
#%% Render a bunch of images to make sure the labels got carried along correctly

bbox_db_path = output_file
output_dir = preview_base

options = visualize_bbox_db.BboxDbVizOptions()
options.num_to_visualize = 1000
options.sort_by_filename = False

htmlOutputFile = visualize_bbox_db.process_images(bbox_db_path,output_dir,image_base,options)


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

from data_management.databases import sanity_check_json_db
from data_management.cct_json_utils import IndexedJsonDb
from visualization import visualize_db
from data_management import cct_json_to_filename_json
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

max_num_csvs = -1

db_sampling_scheme = 'preview' # 'labeled','all'
n_unlabeled_to_sample = -1
cap_unlabeled_to_labeled = True


#%% Read and concatenate source data

# List files
input_files = os.listdir(input_base)

# List of dataframes, one per .csv file; we'll concatenate later
all_input_metadata = []

# i_file = 87; fn = input_files[i_file]
for i_file,fn in enumerate(input_files):

    if max_num_csvs > 0 and len(all_input_metadata) >= max_num_csvs:
        break
    
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

labeled_images = []
unlabeled_images = []


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
        
    category_name = species.lower().strip()
    if category_name in category_mappings:
        category_name = category_mappings[category_name]
        
    if category_name not in category_name_to_category:
        category = {}
        category['name'] = category_name
        category['id'] = next_category_id
        next_category_id += 1
        category_name_to_category[category_name] = category
    else:
        category = category_name_to_category[category_name]
        assert category['name'] == category_name

    category_id = category['id']

    if category_name == 'unlabeled':
        unlabeled_images.append(im)
    else:
        labeled_images.append(im)
        
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
print('Finished verifying file loop in {}, {} matched images, {} missing images, {} unlabeled images'.format(
        humanfriendly.format_timespan(elapsed), len(images), len(missing_files), len(unlabeled_images))) 


#%% See what's up with missing files

dirnames = set()
# s = list(image_relative_paths)[0]
for s in image_relative_paths:
    image_dir = os.path.dirname(s)
    dirnames.add(image_dir)
        
missing_images_with_missing_dirs = []
missing_images_with_non_missing_dirs = []

missing_dirs = set()

# s = missing_files[0]
for s in missing_files:
    assert s not in image_relative_paths
    dirname = os.path.dirname(s)
    if dirname not in dirnames:
        missing_images_with_missing_dirs.append(s)
        missing_dirs.add(dirname)
    else:
        missing_images_with_non_missing_dirs.append(s)

print('Of {} missing files, {} are due to {} missing folders'.format(
        len(missing_files),len(missing_images_with_missing_dirs),len(missing_dirs)))
    

#%% Check for images that aren't included in the metadata file

unmatched_files = []

for i_image,relative_path in tqdm(enumerate(image_relative_paths),total=len(image_relative_paths)):
    
    if relative_path not in relative_path_to_image:
        unmatched_files.append(relative_path)

print('Finished checking {} images to make sure they\'re in the metadata, found {} mismatches'.format(
        len(image_relative_paths),len(unmatched_files)))


#%% Sample the database

images_all = images
annotations_all = annotations

#%%

if db_sampling_scheme == 'all':
    
    pass

elif db_sampling_scheme == 'labeled' or db_sampling_scheme == 'preview':
    
    json_data = {}
    json_data['images'] = images
    json_data['annotations'] = annotations
    json_data['categories'] = categories

    indexed_db = IndexedJsonDb(json_data)
    
    # Collect the images we want
    sampled_images = []
    for im in images:
        classes = indexed_db.get_classes_for_image(im)
        if 'unlabeled' in classes and len(classes) == 1:
            pass
        else:
            sampled_images.append(im)
    
    if db_sampling_scheme == 'preview':
        n_sample = n_unlabeled_to_sample
        if n_sample == -1:
            n_sample = len(labeled_images)
        if n_sample > len(labeled_images) and cap_unlabeled_to_labeled:
            n_sample = len(labeled_images)
        if n_sample > len(unlabeled_images):
            n_sample = len(unlabeled_images)
        print('Sampling {} of {} unlabeled images'.format(n_sample,len(unlabeled_images)))
        from random import sample
        sampled_images.extend(sample(unlabeled_images,n_sample))

    sampled_annotations = []
    for im in sampled_images:
        sampled_annotations.extend(indexed_db.get_annotations_for_image(im))
    
    print('Sampling {} of {} images, {} of {} annotations'.format(
            len(sampled_images),len(images),len(sampled_annotations),len(annotations)))
  
    images = sampled_images
    annotations = sampled_annotations      
    
else:
    
    raise ValueError('Unrecognized DB sampling scheme {}'.format(db_sampling_scheme))


#%% Create info struct

info = {}
info['year'] = 2019
info['version'] = 1
info['description'] = 'COCO style database'
info['secondary_contributor'] = 'Converted to COCO .json by Dan Morris'
info['contributor'] = 'Parks Canada'


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

json_data = json.load(open(output_file))
options = sanity_check_json_db.SanityCheckOptions()
sortedCategories,data = sanity_check_json_db.sanity_check_json_db(json_data, options)

    
#%% Render a bunch of images to make sure the labels got carried along correctly

output_dir = preview_base

options = visualize_db.DbVizOptions()
options.num_to_visualize = 100
options.sort_by_filename = False
# options.classes_to_exclude = ['unlabeled']
options.classes_to_exclude = None

htmlOutputFile,_ = visualize_db.process_images(json_data,output_dir,input_base,options)
os.startfile(htmlOutputFile)


#%% Write out a list of files to annotate

_,file_list = cct_json_to_filename_json.convertJsonToStringList(output_file,prepend="20190715/")
os.startfile(file_list)


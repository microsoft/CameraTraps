#
# ubc_to_json.py
#
# Convert the .csv file provided for the UBC data set to a 
# COCO-camera-traps .json file
#
# Images were provided in eight folders, each of which contained a .csv
# file with annotations.  Those annotations came in two slightly different 
# formats, the two formats corresponding to folders starting with "SC_" and 
# otherwise.
#

#%% Constants and environment

import pandas as pd
import os
import json
import uuid
import numpy as np
import shutil

from tqdm import tqdm
from PIL import Image

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
from path_utils import find_images

input_base = r'e:\ubc'
assert(os.path.isdir(input_base))

output_base = r'f:\data_staging\ubc'
output_json_file = os.path.join(output_base,'ubc.json')
file_list_file = os.path.join(output_base,'all_files.txt')

os.makedirs(output_base,exist_ok=True)

# Map Excel column names - which vary a little across spreadsheets - to a common set of names
mapped_fields = {"Survey.Name" : "survey_name",
                 "project_id": "survey_name",
                 "Camera.Name": "camera_name",
                 "station_id": "camera_name",
                 "Media.Filename": "filename",
                 "orig_file": "filename",
                 "timestamp_pst": "datetime",
                 "Date.Time": "datetime",
                 "Species": "species",
                 "latin_name": "species",
                 "common.name": "common_name",
                 "common_names": "common_name",
                 "Sighting.Quantity": "species_count"
                }

category_mappings = {
    'bird_spp.':'unknown_bird',
    'dog_dog':'dog',
    'hiker_hiker':'hiker',
    'quad_quad':'quad',
    'skier_skier':'skier',
    'snowshoer_snowshoer':'showshoer',
    'quad_quad':'quad'
    }

target_fields = ['species_count','group_count','behaviour']


#%% Enumerate images

# Load from file if we've already enumerated
if os.path.isfile(file_list_file):
    with open(file_list_file,'r') as f:
        files = f.readlines()
    files = [s.strip() for s in files]
    image_full_paths = files
    print('Loaded {} images from {}'.format(len(image_full_paths),file_list_file))
else:
    image_full_paths = find_images(input_base, recursive=True)
    with open(file_list_file,'w') as f:
        for line in image_full_paths:
            f.write(line + '\n')        
    print('Enumerated {} images from {}'.format(len(image_full_paths),input_base))
    
image_full_paths_set = set(image_full_paths)
image_relative_paths = [os.path.relpath(fn,input_base) for fn in image_full_paths]
image_relative_paths_set = set(image_relative_paths)


#%% Create CCT dictionaries

annotations = []
image_ids_to_images = {}
category_name_to_category = {}

# Force the empty category to be ID 0
empty_category = {}
empty_category['name'] = 'empty'
empty_category['id'] = 0
empty_category['common_name'] = 'empty'
category_name_to_category['empty'] = empty_category
next_category_id = 1

latin_to_common = {}

folders = os.listdir(input_base)

# To simplify debugging of the loop below
i_folder = 0; folder = folders[i_folder]; 


##%% Create CCT dictionaries (loop)

invalid_images = []

for i_folder,folder in enumerate(folders):
        
    ##%% 
    
    print('\nProcessing folder {} of {}: {}'.format(i_folder,len(folders),folder))
    
    filenames_to_rows = {}
    filenames_with_multiple_annotations = []
    missing_images = []
    image_directory = os.path.join(input_base, folder)
    files = os.listdir(image_directory)
    files = list(filter(lambda f: f.endswith('.csv'), files))
    input_metadata_file = os.path.join(input_base, folder, files[0])
    assert(os.path.isfile(input_metadata_file))
    
    # Read source data for this folder
    input_metadata = pd.read_csv(input_metadata_file)
    
    # Rename columns
    input_metadata.rename(columns=mapped_fields, inplace=True)
    print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns), len(input_metadata)))
    
    if folder.startswith("SC_"):
        # Folder name is the first two characters of the filename
        #
        # Create relative path names from the filename itself
        input_metadata['image_relative_path'] = input_metadata['filename'].apply(
            lambda x: os.path.join(folder, x[0:2], x.replace(".JPG", ".jpg")))
    else:
        # Folder name is the camera name
        #
        # Create relative path names from camera name and filename    
        input_metadata['image_relative_path'] = input_metadata[['camera_name', 'filename']].apply(
            lambda x: os.path.join(folder, x[0], x[1]), axis = 1)

    # Which of our images are in the spreadsheet?
    # i_row = 0; fn = input_metadata['image_relative_path'][i_row]        
    for i_row, image_relative_path in enumerate(input_metadata['image_relative_path']):
        
        if image_relative_path in filenames_to_rows:
            filenames_with_multiple_annotations.append(image_relative_path)
            filenames_to_rows[image_relative_path].append(i_row)
        else:
            filenames_to_rows[image_relative_path] = [i_row]
            image_full_path = os.path.join(input_base, image_relative_path)
            
            if not image_full_path in image_full_paths_set:
                assert image_relative_path not in image_relative_paths_set
                missing_images.append(image_full_path)
        
    print('Finished verifying image existence for {} files in {} rows\nFound {} filenames with multiple labels, {} missing images'.format(
        len(input_metadata), len(filenames_to_rows), 
        len(filenames_with_multiple_annotations), len(missing_images)))
    
    
    ##%% Check for images that aren't included in the metadata file

    # Find all the images in this folder
    image_relative_paths_this_folder = [s for s in image_relative_paths if s.startswith(folder)]

    # Which of these aren't in the spreadsheet?
    annotated_files_this_folder = list(filenames_to_rows.keys())
    annotated_files_this_folder_set = set(annotated_files_this_folder)
    unannotated_images = [s for s in image_relative_paths_this_folder if s not in annotated_files_this_folder_set]

    print('Found {} unannotated images (of {}) in this folder'.format(
            len(unannotated_images),len(image_relative_paths_this_folder)))
    
    
    ##%% Create entries in CCT dictionaries
    
    image_relative_path = list(filenames_to_rows.keys())[0]
    
    for image_relative_path in list(filenames_to_rows.keys()):
        
        # Only process images we have on disk
        if image_relative_path not in image_relative_paths_set:
            continue
        
        image_full_path = os.path.join(input_base,image_relative_path)
        
        # This is redundant, but doing this for clarity, at basically no performance
        # cost since we need to *read* the images below to check validity.
        assert os.path.isfile(image_full_path)
        
        img_id = image_relative_path.replace('\\','/').replace('/','_').replace(' ','_')
        row_indices = filenames_to_rows[image_relative_path]
        
        # i_row = row_indices[0]
        for i_row in row_indices:
            
            row = input_metadata.iloc[i_row]
            assert(row['image_relative_path'] == image_relative_path)
            timestamp = row['datetime']            
            location = row['survey_name'] + '_' + row['camera_name']
            
            if img_id in image_ids_to_images:
                im = image_ids_to_images[img_id]
                assert im['file_name'] == image_relative_path
                assert im['location'] == location
            else:
                im = {}
                
                try:
                    pil_image = Image.open(image_full_path)        
                    width, height = pil_image.size
                    im['width'] = width
                    im['height'] = height
                except:
                    # These generally represent zero-byte images in this data set, don't try
                    # to find the very small handful that might be other kinds of failures we 
                    # might want to keep around.
                    # print('Error opening image {}'.format(image_relative_path))
                    invalid_images.append(image_relative_path)
                    continue
                
                im['id'] = img_id
                im['file_name'] = image_relative_path
                im['datetime'] = timestamp
                im['location'] = location

                image_ids_to_images[img_id] = im

            species = row['species'].lower().strip().replace(' ','_')
            
            if (isinstance(species,float) or \
                (isinstance(species,str) and (len(species) == 0))):
                category_name = 'empty'
            else:
                category_name = species
            del species
            
            category_name = category_name.strip().lower()
            
            common_name = row['common_name']
            if isinstance(common_name,float) and np.isnan(common_name):
                common_name = ''
            else:
                common_name = str(common_name).lower().strip().replace(', ',',').replace(' ','_')
            
            for k,v in category_mappings.items():
                common_name = common_name.replace(k,v)
                category_name = category_name.replace(k,v)
            common_name = common_name.replace('.','').replace('spp','species')
            category_name = category_name.replace('.','').replace('spp','species')
                        
            if category_name == 'passerine_species' and common_name != '' and common_name != 'passerine_species':
                category_name = common_name
                
            # If we've seen this category before...
            if category_name in category_name_to_category:
                
                category = category_name_to_category[category_name]
                category_id = category['id'] 
                
                # ...make sure it used the same latin --> common mapping
                #
                # If the previous instance had no mapping, use the new one.
                if category['common_name'] == '':
                    category['common_name'] = common_name
                else:
                    # assert common_name == category['common_name']
                    if common_name != category['common_name']:
                        print('Warning: common name {} used for species {}, previously {}'.format(
                            common_name,category_name,category['common_name']))
                    
            else:
                
                category_id = next_category_id
                category = {}
                category['id'] = category_id
                category['name'] = category_name
                category['common_name'] = common_name
                category_name_to_category[category_name] = category
                next_category_id += 1
            
            # Create an annotation
            ann = {}        
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = im['id']    
            ann['category_id'] = category_id
                        
            for target_field in target_fields:
                if target_field in input_metadata.columns:
                    val = row[target_field]
                    if isinstance(val,float) and np.isnan(val):
                        val = ''
                    else:
                        val = str(val).strip()
                    ann[target_field] = val
                
            annotations.append(ann)

        # ...for each annotation we found for this image
            
    # ...for each image
             
# ...for each dataset
            
images = list(image_ids_to_images.values())
categories = list(category_name_to_category.values())

# Print all of our species mappings
for c in categories:
    print(c['name'].ljust(30) + c['common_name'])

print('Finished creating CCT dictionaries, loaded {} images of {} total on disk ({} invalid)'.format(
    len(images), len(image_relative_paths_set), len(invalid_images)))


#%% Copy images for which we actually have annotations to a new folder, lowercase everything

# im = images[0]
for im in tqdm(images):
    relative_filename = im['file_name']
    input_filename = os.path.join(input_base,relative_filename)
    output_filename = os.path.join(output_base,relative_filename).lower()
    os.makedirs(os.path.dirname(output_filename),exist_ok=True)
    
    shutil.copy(input_filename, output_filename)
    im['file_name'] = im['file_name'].lower()
    im['id'] = im['id'].lower()
    

#%% Create info struct

info = {}
info['year'] = 2020
info['version'] = 1
info['description'] = 'UBC Camera Traps'
info['contributor'] = 'UBC'


#%% Convert image IDs to lowercase in annotations, tag as sequence level

# While there isn't any sequence information, the nature of false positives
# here leads me to believe the images were labeled at the sequence level, so
# we should trust labels more when positives are verified.  Overall false
# positive rate looks to be between 1% and 5%.
    
for ann in annotations:
    ann['image_id'] = ann['image_id'].lower()
    ann['sequence_level_annotation'] = True


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_json_file, 'w'), indent=2)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
    len(images), len(annotations), len(categories)))


#%% Validate output

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = output_base
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = True

sortedCategories, data, errors = sanity_check_json_db.sanity_check_json_db(
    output_json_file, options)


#%% Preview labels

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 2000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file, image_db = visualize_db.process_images(db_path=output_json_file,
                                                         output_dir=os.path.join(
                                                             output_base, 'preview'),
                                                         image_base_dir=output_base,
                                                         options=viz_options)
os.startfile(html_output_file)

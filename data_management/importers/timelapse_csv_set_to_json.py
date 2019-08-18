###
#
# timelapse_csv_set_to_json.py
#
# Given a directory full of reasonably-consistent Timelapse-exported
# .csvs, assemble a CCT .json.
#
# Assumes that you have a list of all files in the directory tree, including 
# image and .csv files.
#
###

#%% Constants and imports

import pandas as pd
import copy
import uuid
import json
import time
from tqdm import tqdm
import humanfriendly
import os
import PIL
from visualization import visualize_db
import path_utils

# Text file with relative paths to all files (images and .csv files)
input_relative_file_list = r"F:\uw_gardner\all_files_2019.08.17.txt"
output_file = r"f:\uw_gardner\uw_gardner.2019.08.17.json"
file_base = 'y:\\'

expected_columns = 'File,RelativePath,Folder,Date,Time,ImageQuality,DeleteFlag,CameraLocation,StartDate,TechnicianName,Empty,Service,Species,HumanActivity,Count,AdultFemale,AdultMale,AdultUnknown,Offspring,YOY,UNK,Collars,Tags,NaturalMarks,Reaction,Illegal,GoodPicture,SecondOpinion,Comments'.split(',')
ignore_fields = 'Unnamed: 29'


#%% Read file list, make a list of all image files and all .csv files

with open(input_relative_file_list) as f:
    all_files = f.readlines()
    all_files = [x.strip() for x in all_files] 

image_files = set()
csv_files = []

for fn in all_files:
    fnl = fn.lower()
    if fnl.endswith('.csv'):
        csv_files.append(fn)
    elif (fnl.endswith('.jpg') or fnl.endswith('.png')):
        image_files.add(fn)
        
print('Found {} image files and {} .csv files'.format(len(image_files),len(csv_files)))            
csv_files_raw = copy.copy(csv_files)


#%% Verify column consistency, create a giant array with all rows from all .csv files

bad_csv_files = []
normalized_dataframes = []

# i_csv = 0; csv_filename = csv_files[0]
for i_csv,csv_filename in enumerate(csv_files):
    full_path = os.path.join(file_base,csv_filename)
    try:
        df = pd.read_csv(full_path)        
    except Exception as e:
        if 'invalid start byte' in str(e):
            try:
                print('Read error, reverting to fallback encoding')
                df = pd.read_csv(full_path,encoding='latin1')                
            except Exception as e:
                print('Can''t read file {}: {}'.format(csv_filename,str(e)))
                bad_csv_files.append(csv_filename)
                continue
    if not (len(df.columns) == len(expected_columns) and (df.columns == expected_columns).all()):
        extra_fields = ','.join(set(df.columns) - set(expected_columns))
        extra_fields = [x for x in extra_fields if x not in ignore_fields]
        missing_fields = ','.join(set(expected_columns) - set(df.columns))        
        missing_fields = [x for x in missing_fields if x not in ignore_fields]
        if not (len(missing_fields) == 0 and len(extra_fields) == 0):
            print('In file {}, extra fields {}, missing fields {}'.format(csv_filename,
                  extra_fields,missing_fields))
    normalized_df = df[expected_columns].copy()
    normalized_df['source_file'] = csv_filename
    normalized_dataframes.append(normalized_df)
    
print('Ignored {} of {} csv files'.format(len(bad_csv_files),len(csv_files)))
valid_csv_files = [x for x in csv_files if x not in bad_csv_files]

df = pd.concat(normalized_dataframes)
assert len(df.columns) == 1 + len(expected_columns)

print('Concatenated all .csv files into a dataframe with {} rows'.format(len(df)))


#%% Main loop over labels

startTime = time.time()

relativePathToImage = {}

images = []
annotations = []
categoryIDToCategories = {}
missingFiles = []

duplicateImageIDs = set()

# Force the empty category to be ID 0
emptyCat = {}
emptyCat['name'] = 'empty'
emptyCat['id'] = 0
categoryIDToCategories[0] = emptyCat

# iRow = 0; row = input_metadata.iloc[iRow]
for iRow,row in tqdm(input_metadata.iterrows(),total=len(input_metadata)):
    
    # ImageID,FileName,FilePath,SpeciesID,CommonName
    imageID = str(row['ImageID'])
    fn = row['FileName']
    for k in filename_replacements:
        dirName = row['FilePath'].replace(k,filename_replacements[k])
    relativePath = os.path.join(dirName,fn)
    
    # This makes an assumption of one annotation per image, which happens to be
    # true in this data set.
    if relativePath in relativePathToImage:

        im = relativePathToImage[relativePath]
        assert im['id'] == imageID
        duplicateImageIDs.add(imageID)
            
    else:
        im = {}
        im['id'] = imageID
        im['file_name'] = relativePath
        im['seq_id'] = '-1'
        images.append(im)
        relativePathToImage[relativePath] = im
        
        fullPath = os.path.join(image_base,relativePath)
        
        if not os.path.isfile(fullPath):
            
            missingFiles.append(fullPath)
        
        else:
            # Retrieve image width and height
            pilImage = PIL.Image.open(fullPath)
            width, height = pilImage.size
            im['width'] = width
            im['height'] = height

    categoryName = row['CommonName'].lower()
    if categoryName in category_mappings:
        categoryName = category_mappings[categoryName]
        
    categoryID = row['SpeciesID']
    assert isinstance(categoryID,int)
    
    if categoryID not in categoryIDToCategories:
        category = {}
        category['name'] = categoryName
        category['id'] = categoryID
        categoryIDToCategories[categoryID] = category
    else:
        assert categoryIDToCategories[categoryID]['name'] == categoryName
    
    # Create an annotation
    ann = {}
    
    # The Internet tells me this guarantees uniqueness to a reasonable extent, even
    # beyond the sheer improbability of collisions.
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']    
    ann['category_id'] = categoryID
    
    annotations.append(ann)
    
categories = list(categoryIDToCategories.values())

elapsed = time.time() - startTime
print('Finished verifying file loop in {}, {} images, {} missing images, {} repeat labels'.format(
        humanfriendly.format_timespan(elapsed), len(images), len(missingFiles), len(duplicateImageIDs)))    


#%% Check for images that aren't included in the metadata file

# Enumerate all images
# list(relativePathToImage.keys())[0]

imageFullPaths = path_utils.find_images(image_base,bRecursive=True)
unmatchedFiles = []

for iImage,imagePath in enumerate(imageFullPaths):
    
    fn = os.path.relpath(imagePath,image_base)    
    if fn not in relativePathToImage:
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


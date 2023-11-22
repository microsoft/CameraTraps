#
# awc_to_json.py
#
# Convert a particular .csv file to CCT format.
#

#%% Constants and environment

import pandas as pd
import uuid
import json
import time
from tqdm import tqdm
import humanfriendly
import os
import PIL
from visualization import visualize_db
import path_utils

input_metadata_file = r"D:\wildlife_data\awc\awc_imageinfo.csv"
output_file = r"D:\wildlife_data\awc\awc_imageinfo.json"
image_base = r"D:\wildlife_data\awc"
preview_base = r"D:\wildlife_data\awc\label_preview"

filename_replacements = {'D:\\Wet Tropics':'WetTropics'}
category_mappings = {'none':'empty'}


#%% Read source data

input_metadata = pd.read_csv(input_metadata_file)

print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns),
      len(input_metadata)))


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


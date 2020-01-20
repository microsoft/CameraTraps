#
# carrizo_trail_cam_2017.py
#
# Convert the .csv files provided for the Carrizo Trail Cam 2017 data set to a 
# COCO-camera-traps .json file
#

# %% Constants and environment

import pandas as pd
import os
import glob
import json
import re
import uuid
import time
import ntpath
import humanfriendly
import PIL
from PIL import Image
import numpy as np
import logging

from path_utils import find_images

input_base = r'/mnt/blobfuse/wildlifeblobssc/Trail Cam Carrizo 2017/'
open_metadata_file = os.path.join(input_base, 'Carrizo open 2017.csv')
shrub_metadata_file = os.path.join(input_base, 'Carrizo Shrub 2017.csv')

output_base = r'/home/gramener'
output_json_file = os.path.join(output_base, 'Trail Cam Carrizo 2017.json')
image_directory = input_base


input_metadata_files = [open_metadata_file, shrub_metadata_file]
# output_file = r'C:\Users\Gramener\Desktop\Projects\Microsoft\Camera Traps\carrizo_trail_cam_2017.json'
# image_directory = r'C:\Users\Gramener\Desktop\Projects\Microsoft\Camera Traps\Trail Cam Carrizo 2017'
final_data = pd.DataFrame()
assert(os.path.isdir(image_directory))
logging.basicConfig(filename='carrizo_camera_trap_2017.log', level=logging.INFO)

# %% Read source data

for inp_file in input_metadata_files:
    print("Reading : {0}".format(inp_file))
    input_metadata = pd.read_csv(inp_file)
    print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns), len(input_metadata)))
    # Removing the empty records
    input_metadata = input_metadata[~np.isnan(input_metadata['rep'])]
    # Filenames were provided as numbers, but images were *.JPG, converting here
    input_metadata['file'] = input_metadata.groupby(["rep", "week"]).cumcount()
    week_folder_format = {1: 'week 1- 2017', 2: 'week2- 2017', 3: 'week3-2017'}
    input_metadata['file'] = input_metadata[['file', 'rep', 'week', 'microsite']].apply(
        lambda x: "{3}/{4}{1}-week{0}-carrizo-2017/IMG_{2}.JPG".format(int(x[2]), int(x[1]), str(int(x[0]+1)).zfill(4), week_folder_format[int(x[2])], x[3].lower()), axis=1)
    print('Converted filenames in the dataframe')
    final_data = final_data.append(input_metadata)
# import pdb;pdb.set_trace()
# %% Map filenames to rows, verify image existence
# import pdb;pdb.set_trace()
startTime = time.time()
filenamesToRows = {}
imageFilenames = input_metadata.file

duplicateRows = []

logging.info("File names which are present in CSV but not in the directory")
# Build up a map from filenames to a list of rows, checking image existence as we go
for iFile, fn in enumerate(imageFilenames):
    if (fn in filenamesToRows):
        duplicateRows.append(iFile)
        filenamesToRows[fn].append(iFile)
    else:
        filenamesToRows[fn] = [iFile]
        imagePath = os.path.join(image_directory, fn)
        try:
            assert(os.path.isfile(imagePath))
        except Exception:
            logging.info(imagePath)

elapsed = time.time() - startTime
print('Finished verifying image existence in {}, found {} filenames with multiple labels'.format(
    humanfriendly.format_timespan(elapsed), len(duplicateRows)))

# I didn't expect this to be true a priori, but it appears to be true, and
# it saves us the trouble of checking consistency across multiple occurrences
# of an image.
assert(len(duplicateRows) == 0)    

# %% Check for images that aren't included in the metadata file

# Enumerate all images
# imageFullPaths = glob.glob(os.path.join(image_directory, '*\\*\\*.JPG'))
imageFullPaths = find_images(image_directory, bRecursive=True)
for iImage, imagePath in enumerate(imageFullPaths):
    # fn = ntpath.basename(imagePath)
    # parent_dir = os.path.basename(os.path.dirname(imagePath))
    # import pdb;pdb.set_trace()
    assert(imagePath.replace(input_base, '') in filenamesToRows)

print('Finished checking {} images to make sure they\'re in the metadata'.format(
        len(imageFullPaths)))

# %% Create CCT dictionaries

# Also gets image sizes, so this takes ~6 minutes
#
# Implicitly checks images for overt corruptness, i.e. by not crashing.

images = []
annotations = []

# Map categories to integer IDs (that's what COCO likes)
nextCategoryID = 0
categoriesToCategoryId = {}
categoriesToCounts = {}

# For each image
#
# Because in practice images are 1:1 with annotations in this data set,
# this is also a loop over annotations.

startTime = time.time()
print(imageFilenames)
imageName = imageFilenames[0]
for imageName in imageFilenames:
    rows = filenamesToRows[imageName]
    # As per above, this is convenient and appears to be true; asserting to be safe
    assert(len(rows) == 1)    
    iRow = rows[0]
    row = input_metadata.iloc[iRow]
    im = {}
    im['id'] = image_name.replace('\\','/').replace('/','_').replace(' ','_')
    im['file_name'] = imageName
    im['region'] = row['region']
    im['site']= row['site']
    im['mircosite'] = row['mircosite']
    im['datetime'] = row['calendar date']
    im['location'] = "{0}_{1}_{2}".format(row['region'], row['site'], row['microsite'])
    # Check image height and width
    imagePath = os.path.join(image_directory, parent_dir, fn)
    assert(os.path.isfile(imagePath))
    pilImage = Image.open(imagePath)
    width, height = pilImage.size
    im['width'] = width
    im['height'] = height

    images.append(im)
    
#     category = row['label'].lower()
    is_image = row['animal.capture']
    
    # Use 'empty', to be consistent with other data on lila    
    if (is_image == 0):
        category = 'empty'
    else:
        if row['animal'] is np.nan:
            category = 'unidentifiable'
        else:
            category = row['animal']
        
    # Have we seen this category before?
    if category in categoriesToCategoryId:
        categoryID = categoriesToCategoryId[category]
        categoriesToCounts[category] += 1
    else:
        categoryID = nextCategoryID
        categoriesToCategoryId[category] = categoryID
        categoriesToCounts[category] = 0
        nextCategoryID += 1
    
    # Create an annotation
    ann = {}
    
    # The Internet tells me this guarantees uniqueness to a reasonable extent, even
    # beyond the sheer improbability of collisions.
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']    
    ann['category_id'] = categoryID
    
    annotations.append(ann)
    
# ...for each image
    
# Convert categories to a CCT-style dictionary

categories = []

for category in categoriesToCounts:
    print('Category {}, count {}'.format(category,categoriesToCounts[category]))
    categoryID = categoriesToCategoryId[category]
    cat = {}
    cat['name'] = category
    cat['id'] = categoryID
    categories.append(cat)    
    
elapsed = time.time() - startTime
print('Finished creating CCT dictionaries in {}'.format(
      humanfriendly.format_timespan(elapsed)))
    

# %% Create info struct

info = {}
info['year'] = 2018
info['version'] = 1
info['description'] = 'COCO style database'
info['secondary_contributor'] = 'Converted to COCO .json by Dan Morris'
info['contributor'] = 'Vardhan Duvvuri'


# %% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_file, 'w'), indent=4)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))

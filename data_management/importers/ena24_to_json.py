#
# ena24_to_json_2017.py
#
# Convert the ENA24 data set to a COCO-camera-traps .json file
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

# input_metadata_file = r'C:\Users\Gramener\Desktop\Projects\Microsoft\Camera Traps\MNP 2017 .csv'
output_file = r'C:\Users\Gramener\Desktop\Projects\Microsoft\Camera Traps\ena24.json'
image_directory = r'C:\Users\Gramener\Desktop\Projects\Microsoft\Camera Traps\ena24\images'
label_directory = r'C:\Users\Gramener\Desktop\Projects\Microsoft\Camera Traps\ena24\labels'
labels = ['White_Tailed_Deer', 'Dog', 'Bobcat', 'Red_Fox', 'Horse', 'Domestic_Cat',
            'American_Black_Bear', 'Eastern_Cottontail', 'Grey_Fox', 'Coyote', 'Eastern Fox Squirrel',
            'Eastern_Gray_Squirre', 'Vehicle', 'Eastern_Chipmunk', 'Wild_Turkey', 'Northern Raccoon',
            'Striped_Skunk', 'Woodchuck', 'Virginia_Opossum', 'Human', 'Bird', 'American_Crow', 'Chicken']

assert(os.path.isdir(label_directory))
assert(os.path.isdir(image_directory))

# %% Read source data

dir_list = os.listdir(label_directory)

print('Read {} rows from metadata file'.format(len(dir_list)))
# %% Map filenames to rows, verify image existence

startTime = time.time()
filenamesToRows = {}

# Build up a map from filenames to a list of rows, checking image existence as we go
for iFile in dir_list:
    imagePath = os.path.join(image_directory, "{}.jpg".format(iFile.split(".")[0]))
    assert(os.path.isfile(imagePath))

elapsed = time.time() - startTime
print('Finished verifying image existence in {}'.format(
      humanfriendly.format_timespan(elapsed)))

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
for iFile in dir_list:
    file_data = open(os.path.join(label_directory, iFile), 'r').read()
    row = file_data.split()
    im = {}
    im['id'] = iFile.split('.')[0]
    fn = "{}.jpg".format(iFile.split('.')[0])
    im['file_name'] = fn
    # Check image height and width
    imagePath = os.path.join(image_directory, fn)
    assert(os.path.isfile(imagePath))
    pilImage = Image.open(imagePath)
    width, height = pilImage.size
    im['width'] = width
    im['height'] = height

    images.append(im)
    
    category = labels[int(row[0])-1]

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
    ann['bbox'] = [float(row[1]), float(row[2]), float(row[3])*width, float(row[4])*height]
    annotations.append(ann)
    
# ...for each image
    
# Convert categories to a CCT-style dictionary

categories = []

for category in categoriesToCounts:
    print('Category {}, count {}'.format(category, categoriesToCounts[category]))
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
info['year'] = 2016
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
print(json_data)
# json.dump(json_data, open(output_file, 'w'), indent=4)

# print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
#         len(images),len(annotations),len(categories)))

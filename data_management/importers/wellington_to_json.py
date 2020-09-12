#
# wellington_to_json.py
#
# Convert the .csv file provided for the Wellington data set to a 
# COCO-camera-traps .json file
#

#%% Constants and environment

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

input_metadata_file = r'e:\wellington_data\anton_wellington_metadata.csv'
output_file = r'e:\wellington_data\wellington_camera_traps.json'
image_directory = r'e:\wellington_data\images\images'

assert(os.path.isdir(image_directory))


#%% Read source data

input_metadata = pd.read_csv(input_metadata_file)

print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns),
      len(input_metadata)))

# Filenames were provided as *.jpg, but images were *.JPG, converting here
input_metadata['file'] = input_metadata['file'].apply(lambda x: x.replace('.jpg','.JPG'))

print('Converted extensions to uppercase')


#%% Map filenames to rows, verify image existence

# Takes ~30 seconds, since it's checking the existence of ~270k images

startTime = time.time()
filenamesToRows = {}
imageFilenames = input_metadata.file

duplicateRows = []

# Build up a map from filenames to a list of rows, checking image existence as we go
for iFile,fn in enumerate(imageFilenames):
    
    if (fn in filenamesToRows):
        duplicateRows.append(iFile)
        filenamesToRows[fn].append(iFile)
    else:
        filenamesToRows[fn] = [iFile]
        imagePath = os.path.join(image_directory,fn)
        assert(os.path.isfile(imagePath))

elapsed = time.time() - startTime
print('Finished verifying image existence in {}, found {} filenames with multiple labels'.format(
      humanfriendly.format_timespan(elapsed),len(duplicateRows)))

# I didn't expect this to be true a priori, but it appears to be true, and
# it saves us the trouble of checking consistency across multiple occurrences
# of an image.
assert(len(duplicateRows) == 0)    
    
        
    
#%% Check for images that aren't included in the metadata file

# Enumerate all images
imageFullPaths = glob.glob(os.path.join(image_directory,'*.JPG'))

for iImage,imagePath in enumerate(imageFullPaths):
    
    fn = ntpath.basename(imagePath)
    assert(fn in filenamesToRows)

print('Finished checking {} images to make sure they\'re in the metadata'.format(
        len(imageFullPaths)))


#%% Create CCT dictionaries

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

# imageName = imageFilenames[0]
for imageName in imageFilenames:
    
    rows = filenamesToRows[imageName]
    
    # As per above, this is convenient and appears to be true; asserting to be safe
    assert(len(rows) == 1)    
    iRow = rows[0]
    
    row = input_metadata.iloc[iRow]
    
    im = {}
    # Filenames look like "290716114012001a1116.jpg"
    im['id'] = imageName.split('.')[0]
    im['file_name'] = imageName
    
    # This gets imported as an int64
    im['seq_id'] = int(row['sequence'])
    
    # These appear as "image1", "image2", etc.
    frameID = row['image_sequence']
    m = re.match('^image(\d+)$',frameID)
    assert (m is not None)
    im['frame_num'] = int(m.group(1))-1
    
    # In the form "001a"
    im['site']= row['site']
    
    # Can be in the form '111' or 's46'
    im['camera'] = row['camera']
    
    # In the form "7/29/2016 11:40"
    im['datetime'] = row['date']
    
    # Check image height and width
    imagePath = os.path.join(image_directory,fn)
    assert(os.path.isfile(imagePath))
    pilImage = PIL.Image.open(imagePath)
    width, height = pilImage.size
    im['width'] = width
    im['height'] = height

    images.append(im)
    
    category = row['label'].lower()
    
    # Use 'empty', to be consistent with other data on lila    
    if (category == 'nothinghere'):
        category = 'empty'
        
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
    

#%% Create info struct

info = {}
info['year'] = 2018
info['version'] = 1
info['description'] = 'COCO style database'
info['secondary_contributor'] = 'Converted to COCO .json by Dan Morris'
info['contributor'] = 'Victor Anton'


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data,open(output_file,'w'),indent=4)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))



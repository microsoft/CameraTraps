#
# mcgill_to_json.py
#
# Convert the .csv file provided for the McGill test data set to a 
# COCO-camera-traps .json file
#

#%% Constants and environment

import pandas as pd
import os
import glob
import json
import uuid
import time
import ntpath
import humanfriendly
import PIL
import math

baseDir = r'D:\wildlife_data\mcgill_test'
input_metadata_file = os.path.join(baseDir, 'dan_500_photos_metadata.csv')
output_file = os.path.join(baseDir, 'mcgill_test.json')
image_directory = baseDir

assert(os.path.isdir(image_directory))
assert(os.path.isfile(input_metadata_file))


#%% Read source data

input_metadata = pd.read_csv(input_metadata_file)

print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns),
      len(input_metadata)))


#%% Map filenames to rows, verify image existence

# Create an additional column for concatenated filenames
input_metadata['relative_path'] = ''
input_metadata['full_path'] = ''

startTime = time.time()

# Maps relative filenames to rows
filenamesToRows = {}

duplicateRows = []

# Build up a map from filenames to a list of rows, checking image existence as we go
# row = input_metadata.iloc[0]
for iFile,row in input_metadata.iterrows():

    relativePath = os.path.join(row['site'],row['date_range'],str(row['camera']),
                                str(row['folder']),row['filename'])
    fullPath = os.path.join(baseDir,relativePath)
    
    if (relativePath in filenamesToRows):
        duplicateRows.append(iFile)
        filenamesToRows[relativePath].append(iFile)
    else:
        filenamesToRows[relativePath] = [iFile]
        assert(os.path.isfile(fullPath))

    row['relative_path'] = relativePath
    row['full_path'] = fullPath
    
    input_metadata.iloc[iFile] = row
    
elapsed = time.time() - startTime
print('Finished verifying image existence in {}, found {} filenames with multiple labels'.format(
      humanfriendly.format_timespan(elapsed),len(duplicateRows)))

# I didn't expect this to be true a priori, but it appears to be true, and
# it saves us the trouble of checking consistency across multiple occurrences
# of an image.
assert(len(duplicateRows) == 0)    
    
    
#%% Check for images that aren't included in the metadata file

# Enumerate all images
imageFullPaths = glob.glob(os.path.join(image_directory,'**/*.JPG'), recursive=True)

for iImage,imagePath in enumerate(imageFullPaths):
    
    imageRelPath = ntpath.relpath(imagePath, image_directory)
    assert(imageRelPath in filenamesToRows)

print('Finished checking {} images to make sure they\'re in the metadata'.format(
        len(imageFullPaths)))


#%% Create CCT dictionaries

# Also gets image sizes, so this takes ~6 minutes
#
# Implicitly checks images for overt corruptness, i.e. by not crashing.

images = []
annotations = []
categories = []

emptyCategory = {}
emptyCategory['id'] = 0
emptyCategory['name'] = 'empty'
emptyCategory['latin'] = 'empty'
emptyCategory['count'] = 0
categories.append(emptyCategory)

# Map categories to integer IDs (that's what COCO likes)
nextCategoryID = 1
labelToCategory = {'empty':emptyCategory}

# For each image
#
# Because in practice images are 1:1 with annotations in this data set,
# this is also a loop over annotations.

startTime = time.time()

# row = input_metadata.iloc[0]
for iFile,row in input_metadata.iterrows():

    relPath = row['relative_path'].replace('\\','/')
    im = {}
    # Filenames look like "290716114012001a1116.jpg"
    im['id'] = relPath.replace('/','_').replace(' ','_')
    
    im['file_name'] = relPath
    
    im['seq_id'] = -1
    im['frame_num'] = -1
    
    # In the form "001a"
    im['site']= row['site']
    
    # Can be in the form '111' or 's46'
    im['camera'] = row['camera']
    
    # In the form "7/29/2016 11:40"
    im['datetime'] = row['timestamp']
    
    otherFields = ['motion','temp_F','n_present','n_waterhole','n_contact','notes']
    
    for s in otherFields:
        im[s] = row[s]
        
    # Check image height and width
    fullPath = row['full_path']
    assert(os.path.isfile(fullPath))
    pilImage = PIL.Image.open(fullPath)
    width, height = pilImage.size
    im['width'] = width
    im['height'] = height

    images.append(im)
    
    label = row['species']
    if not isinstance(label,str):
        # NaN is the only thing we should see that's not a string        
        assert math.isnan(label)
        label = 'empty'
    else:
        label = label.lower()
    
    latin = row['binomial']
    if not isinstance(latin,str):
        # NaN is the only thing we should see that's not a string
        assert math.isnan(latin)
        latin = 'empty'
    else:
        latin = latin.lower()

    if label == 'empty':
        if latin != 'empty':
            latin = 'empty'

    if label == 'unknown':
        if latin != 'unknown':
            latin = 'unknown'
            
    if label not in labelToCategory:
        print('Adding category {} ({})'.format(label,latin))
        category = {}
        categoryID = nextCategoryID
        category['id'] = categoryID        
        nextCategoryID += 1
        category['name'] = label
        category['latin'] = latin
        category['count'] = 1
        labelToCategory[label] = category
        categories.append(category)
    else:
        category = labelToCategory[label]
        category['count'] = category['count'] + 1
        categoryID = category['id']        
        
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


for category in categories:
    print('Category {}, count {}'.format(category['name'],category['count']))
    
elapsed = time.time() - startTime
print('Finished creating CCT dictionaries in {}'.format(
      humanfriendly.format_timespan(elapsed)))
    

#%% Create info struct

info = {}
info['year'] = 2019
info['version'] = 1
info['description'] = 'COCO style database'
info['secondary_contributor'] = 'Converted to COCO .json by Dan Morris'
info['contributor'] = 'McGill University'


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_file,'w'), indent=4)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))





#
# jb_csv_to_json.py
#
# Convert a particular .csv file to CCT format.  Images were not available at
# the time I wrote this script, so this is much shorter than other scripts 
# in this folder.
#

#%% Constants and environment

import pandas as pd
import uuid
import json

input_metadata_file = r'd:\temp\pre_bounding_box.csv'
output_file = r'd:\temp\pre_bounding_box.json'
filename_col = 'filename'
label_col = 'category'


#%% Read source data

input_metadata = pd.read_csv(input_metadata_file)

print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns),
      len(input_metadata)))


#%% Confirm filename uniqueness (this data set has one label per image)

imageFilenames = input_metadata[filename_col]

duplicateRows = []
filenamesToRows = {}

# Build up a map from filenames to a list of rows, checking image existence as we go
for iFile,fn in enumerate(imageFilenames):
    
    if (fn in filenamesToRows):
        duplicateRows.append(iFile)
        filenamesToRows[fn].append(iFile)
    else:
        filenamesToRows[fn] = [iFile]

assert(len(duplicateRows) == 0)    
    

#%% Create CCT dictionaries

images = []
annotations = []

# Map categories to integer IDs (that's what COCO likes)
nextCategoryID = 1
categories = []
categoryNamesToCategories = {}

cat = {}
cat['name'] = 'empty'
cat['id'] = 0
categories.append(cat)    
categoryNamesToCategories['empty'] = cat

# For each image
#
# Because in practice images are 1:1 with annotations in this data set,
# this is also a loop over annotations.

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
    im['seq_id'] = '-1'
    
    images.append(im)
    
    categoryName = row[label_col].lower()
    
    # Have we seen this category before?
    if categoryName in categoryNamesToCategories:
        categoryID = categoryNamesToCategories[categoryName]['id']
    else:
        cat = {}
        categoryID = nextCategoryID 
        cat['name'] = categoryName
        cat['id'] = nextCategoryID 
        categories.append(cat)    
        categoryNamesToCategories[categoryName] = cat
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
        
print('Finished creating dictionaries')


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


#%% Sanity-check

from data_management.databases import sanity_check_json_db

options = sanity_check_json_db.SanityCheckOptions()
sortedCategories,data = sanity_check_json_db.sanity_check_json_db(output_file, options)



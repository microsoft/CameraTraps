#
# save_the_elephants_survey_B.py
#
# Convert the .csv file provided for the Save the Elephants Survey B data set to a 
# COCO-camera-traps .json file
#

#%% Constants and environment

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
import pandas as pd
import os
import glob
import json
import uuid
import time
import humanfriendly
from PIL import Image
import numpy as np
import logging
from tqdm import tqdm
import shutil


input_metadata_file = r'/mnt/blobfuse/wildlifeblobssc/ste_2019_08_drop/SURVEY B.xlsx'
output_file = r'/data/home/gramener/SURVEY_B.json'
image_directory = r'/mnt/blobfuse/wildlifeblobssc/ste_2019_08_drop/SURVEY B with False Triggers'
log_file = r'/data/home/gramener/save_elephants_survey_b.log'
output_dir = r'/data/home/gramener/SURVEY_B'

os.mkdir(output_dir)
assert(os.path.isdir(image_directory))
logging.basicConfig(filename=log_file, level=logging.INFO)


#%% Read source data

input_metadata = pd.read_excel(input_metadata_file, sheet_name='9. CT Image')
input_metadata = input_metadata.iloc[2:]

print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns),
      len(input_metadata)))


#%% Map filenames to rows, verify image existence

# Takes ~30 seconds, since it's checking the existence of ~270k images

startTime = time.time()
filenamesToRows = {}
imageFilenames = input_metadata['Image Name']

duplicateRows = []

logging.info("File names which are present in CSV but not in the directory")
# Build up a map from filenames to a list of rows, checking image existence as we go
for iFile, fn in enumerate(imageFilenames):
    if (fn in filenamesToRows):
        # print(fn)
        duplicateRows.append(iFile)
        filenamesToRows[fn].append(iFile)
    else:
        filenamesToRows[fn] = [iFile]
        try:
            imagePath = os.path.join(image_directory, fn)
            assert(os.path.isfile(imagePath))
        except Exception:
            logging.info(imagePath)

elapsed = time.time() - startTime
print('Finished verifying image existence in {}, found {} filenames with multiple labels'.format(
      humanfriendly.format_timespan(elapsed), len(duplicateRows)))


#%% Check for images that aren't included in the metadata file

# Enumerate all images
imageFullPaths = glob.glob(os.path.join(image_directory, '*\\*\\*\\*.JPG'))
for iImage, imagePath in enumerate(imageFullPaths):
    # fn = ntpath.basename(imagePath)
    fn = imagePath.split(image_directory)[1]
    # parent_dir = os.path.basename(os.path.dirname(imagePath))
    assert(fn[1:] in filenamesToRows)

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
processed = []
startTime = time.time()
# print(imageFilenames)
# imageName = imageFilenames[0]
for imageName in tqdm(imageFilenames):
    
    try:        
        rows = filenamesToRows[imageName]
        iRow = rows[0]
        row = input_metadata.iloc[iRow+2]
        im = {}
        img_id = imageName.split('.')[0]
        if img_id in processed:
            continue
        processed.append(img_id)
        im['id'] = img_id
        im['file_name'] = imageName
        im['datetime'] = row['Date'].strftime("%d/%m/%Y")
        im['Camera Trap Station Label'] = row['Camera Trap Station Label']
        if row['No. of Animals in Photo'] is np.nan:
            im['No. of Animals in Photo'] = 0
        else:
            im['No. of Animals in Photo'] = row['No. of Animals in Photo']
        if row['Photo Type '] is np.nan:
            im['Photo Type '] = ""
        else:
            im['Photo Type'] = row['Photo Type ']

        # Check image height and width        
        imagePath = os.path.join(image_directory, imageName)
        assert(os.path.isfile(imagePath))
        pilImage = Image.open(imagePath)
        width, height = pilImage.size
        im['width'] = width
        im['height'] = height

        images.append(im)
        shutil.copy(imagePath, output_dir)
        # category = row['label'].lower()
        is_image = row['Species']
    
        # Use 'empty', to be consistent with other data on lila    
        if (is_image == np.nan or is_image == " " or type(is_image) == float):
            category = 'empty'
        else:
            category = row['Species']
        
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
    except Exception:
        continue
    
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
info['year'] = 2014
info['version'] = 1
info['description'] = ''
info['secondary_contributor'] = 'Converted to COCO .json by Vardhan Duvvuri'
info['contributor'] = 'Save the Elephants'


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_file, 'w'), indent=2)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))


#%% Validate output

fn = output_file
options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = image_directory
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True

sortedCategories, data = sanity_check_json_db.sanity_check_json_db(fn, options)


#%% Preview labels

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 1000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file, image_db = visualize_db.process_images(db_path=output_file,
                                                         output_dir='/home/gramener/previewB',
                                                         image_base_dir=image_directory,
                                                         options=viz_options)
os.startfile(html_output_file)

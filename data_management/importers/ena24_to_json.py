#
# ena24_to_json_2017.py
#
# Convert the ENA24 data set to a COCO-camera-traps .json file
#

#%% Constants and environment

import os
import json
import uuid
import time
import humanfriendly
import numpy as np
from PIL import Image
from tqdm import tqdm

# output_file = r'C:\Users\Gramener\Desktop\Projects\Microsoft\Camera Traps\ena24.json'
# image_directory = r'C:\Users\Gramener\Desktop\Projects\Microsoft\Camera Traps\ena24\images'
# label_directory = r'C:\Users\Gramener\Desktop\Projects\Microsoft\Camera Traps\ena24\labels'

base_directory = r'e:\wildlife_data\ena24'
output_file = os.path.join(base_directory,'ena24.json')
image_directory = os.path.join(base_directory,'images')
label_directory = os.path.join(base_directory,'labels')

labels = ['White_Tailed_Deer', 'Dog', 'Bobcat', 'Red Fox', 'Horse', 
          'Domestic Cat', 'American Black Bear', 'Eastern Cottontail', 'Grey Fox', 'Coyote', 
          'Eastern Fox Squirrel', 'Eastern Gray Squirrel', 'Vehicle', 'Eastern Chipmunk', 'Wild Turkey',
          'Northern Raccoon', 'Striped Skunk', 'Woodchuck', 'Virginia Opossum', 'Human', 
          'Bird', 'American Crow', 'Chicken']

assert(os.path.isdir(label_directory))
assert(os.path.isdir(image_directory))


#%% Read source data

image_list = os.listdir(label_directory)
print('Enumerated {} label files'.format(len(image_list)))


#%% Map filenames to rows, verify image existence

startTime = time.time()

# Build up a map from filenames to a list of rows, checking image existence as we go
for filename in image_list:
    imagePath = os.path.join(image_directory, "{}.jpg".format(filename.split(".")[0]))
    assert(os.path.isfile(imagePath))

elapsed = time.time() - startTime
print('Finished verifying image existence for {} files in {}'.format(
      len(image_list),humanfriendly.format_timespan(elapsed)))


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
for filename in tqdm(image_list):
    
    im = {}
    im['id'] = filename.split('.')[0]
    fn = "{}.jpg".format(filename.split('.')[0])
    im['file_name'] = fn
    
    # Check image height and width
    imagePath = os.path.join(image_directory, fn)
    assert(os.path.isfile(imagePath))
    pilImage = Image.open(imagePath)
    width, height = pilImage.size
    im['width'] = width
    im['height'] = height

    images.append(im)
    
    label_path = os.path.join(label_directory, filename)
    file_data = open(label_path, 'r').read()
    row = file_data.split()
    category = labels[int(row[0])-1]

    rows = np.loadtxt(label_path)
    
    # Each row is category, [box coordinates]    
    
    # If there's just one row, loadtxt reads it as a 1d array; make it a 2d array 
    # with one row 
    if len(rows.shape)==1:
        rows = rows.reshape(1,-5)
    
    assert (len(rows.shape)==2 and rows.shape[1] == 5)
    
    categories_this_image = set()

    # Each row is a bounding box
    for row in rows:
        
        i_category = int(row[0])-1        
        category = labels[i_category]
        categories_this_image.add(category)
        
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
        
        ann['id'] = str(uuid.uuid1())
        ann['image_id'] = im['id']    
        ann['category_id'] = categoryID
        ann['bbox'] = [row[1]*width, row[2]*height, row[3]*width, row[4]*height]
        annotations.append(ann)
        
    # ...for each bounding box
    
    # This was here for debugging; nearly every instance is Human+Horse, Human+Vehicle,
    # or Human+Dog, but there is one Rabbit+Opossium, and a few Deer+Chicken!
    if False:
        if len(categories_this_image) > 1:
            print('Image {} has multiple categories: '.format(filename),end='')
            for c in categories_this_image:
                print(c, end=',')
            print('')
            
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
    

#%% Create info struct

info = {}
info['year'] = 2016
info['version'] = 1
info['description'] = ''
info['secondary_contributor'] = 'Converted to COCO .json by Vardhan Duvvuri'
info['contributor'] = 'University of Missouri'


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

from data_management.databases import sanity_check_json_db

fn = output_file
options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = image_directory
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True
    
sortedCategories, data = sanity_check_json_db.sanity_check_json_db(fn,options)


#%% Preview labels

from visualization import visualize_db
from data_management.databases import sanity_check_json_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = None
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file,image_db = visualize_db.process_images(db_path=output_file,
                                                        output_dir=os.path.join(base_directory,'preview'),
                                                        image_base_dir=image_directory,
                                                        options=viz_options)
os.startfile(html_output_file)

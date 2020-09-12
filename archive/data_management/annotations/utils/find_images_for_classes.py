#
# find_images_for_classes.py
#
# Given a .json database, find images that are associated with one or more
# classes.
#

#%% Constants and imports

import os
import json


#%% Configuration

imageBaseDir = r'd:\temp\snapshot_serengeti_tfrecord_generation'
annotationFile = os.path.join(imageBaseDir,'imerit_batch7_renamed.json')
targetClasses = set(['lionMale'])


#%%  Read database and build up convenience mappings 

print("Loading json database")
with open(annotationFile, 'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']

# Category ID to category name
categoryIDToCategoryName = {cat['id']:cat['name'] for cat in categories}

# Image ID to image info
imageIDToImage = {im['id']:im for im in images}

# Image ID to image path
imageIDToPath = {}

for im in images:
    imageID = im['id']
    imageIDToPath[imageID] = os.path.join(imageBaseDir,im['file_name'])


#%% Look for target-class annotations

targetClassPaths = []
    
# ann = annotations[0]
for ann in annotations:
    
    imageID = ann['image_id']
    assert imageID in imageIDToPath
    assert imageID in imageIDToImage
    
    catID = ann['category_id']
    classname = categoryIDToCategoryName[catID]
    if classname in targetClasses:        
        targetClassPaths.append(imageIDToPath[imageID])
    
print('Found {} target-class images'.format(len(targetClassPaths)))

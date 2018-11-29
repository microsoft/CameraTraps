#
# filter_database.py
#
# Look through a COCO-ct database and find images matching some crtieria, writing
# a subset of images and annotations to a new file.
#

#%% Constants and imports

import os
import json


#%% Configuration

baseDir = r'd:\temp\snapshot_serengeti_tfrecord_generation'
imageBaseDir = os.path.join(baseDir,'imerit_batch7_images_renamed')
annotationFile = os.path.join(baseDir,'imerit_batch7_renamed_uncorrupted.json')
outputFile = os.path.join(baseDir,'imerit_batch7_renamed_uncorrupted_filtered.json')


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

print('Finished loading .json data')


#%% Filter

filteredImages = {}
filteredAnnotations = []

# ann = annotations[0]
for iAnn,ann in enumerate(annotations):
    
    if not 'bbox' in ann:
        continue
    
    # Is this a tiny box or a group annotation?
    MIN_BOX_SIZE = 50
    
    # x,y,w,h
    bbox = ann['bbox']
    w = bbox[2]; h = bbox[3]
    minsize = min(w,h)
    
    annotationType = ann['annotation_type']
    
    if ((minsize >= MIN_BOX_SIZE) and (annotationType != 'group')):
        imageID = ann['image_id']    
        imageInfo = imageIDToImage[imageID]
        filteredImages[imageID] = imageInfo
        filteredAnnotations.append(ann)
        
print('Filtered {} of {} images and {} of {} annotations'.format(
        len(filteredImages),len(images),
        len(filteredAnnotations),len(annotations)))
    

#%% Write output file

dataFiltered = data
dataFiltered['images'] = list(filteredImages.values())
dataFiltered['annotations'] = filteredAnnotations

json.dump(dataFiltered,open(outputFile,'w'))

#
# filter_database.py
#
# Look through a COCO-ct database and find images matching some crtieria, writing
# a subset of images and annotations to a new file.
#

#%% Constants and imports

import os
import json
import math


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
nEmpty = 0

# ann = annotations[0]
for iAnn,ann in enumerate(annotations):
    
    # Is this a tiny box or a group annotation?
    MIN_BOX_SIZE = 50
    
    minsize = math.inf
    
    if ('bbox' in ann):
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
        
    if (annotationType == 'empty'):
        nEmpty +=1
        assert 'bbox' not in ann
        # All empty annotations should be classified as either empty or ambiguous
        #
        # The ambiguous cases are basically minor misses on the annotators' part,
        # where two different small animals were present somewhere.
        assert ann['category_id'] == 0 or ann['category_id'] == 1001
        
print('Filtered {} of {} images and {} of {} annotations ({} empty)'.format(
        len(filteredImages),len(images),
        len(filteredAnnotations),len(annotations),nEmpty))
    

#%% Write output file

dataFiltered = data
dataFiltered['images'] = list(filteredImages.values())
dataFiltered['annotations'] = filteredAnnotations

json.dump(dataFiltered,open(outputFile,'w'))

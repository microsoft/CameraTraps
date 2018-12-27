#
# sanity_check_json_db.py
#
# Does some sanity-checking and computes basic statistics on a db, specifically:
#
# * Verifies that annotations refer to valid images
# * Verifies that annotations refer to valid categories
# * Verifies that image, category, and annotation IDs are unique 
#
# * Finds un-annotated images
# * Finds unused categories
#
# * Prints a list of categories sorted by count

#%% Constants and environment

import json
import os
from operator import itemgetter

jsonFile = r'd:\temp\wellington_camera_traps.json'
assert os.path.isfile(jsonFile)


#%% Read .json file, sanity-check fields

with open(jsonFile,'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']
info = data['info']


#%% Build dictionaries, checking ID uniqueness and internal validity as we go

imageIdToImage = {}
annIdToAnn = {}
catIdToCat = {}

for cat in categories:
    
    # Confirm that required fields are present
    assert 'name' in cat
    assert 'id' in cat
    
    catId = cat['id']
    
    # Confirm ID uniqueness
    assert catId not in catIdToCat
    catIdToCat[catId] = cat
    cat['_count'] = 0
    
# ...for each category
    
for image in images:
    
    # Confirm that required fields are present
    assert 'file_name' in image
    assert 'id' in image
    
    imageId = image['id']        
    
    # Confirm ID uniqueness
    assert imageId not in imageIdToImage
    imageIdToImage[imageId] = image
    image['_count'] = 0
    
# ...for each image

for ann in annotations:

    # Confirm that required fields are present
    assert 'image_id' in ann
    assert 'id' in ann
    assert 'category_id' in ann
    
    annId = ann['id']        
    
    # Confirm ID uniqueness
    assert annId not in annIdToAnn
    annIdToAnn[annId] = ann

    # Confirm validity
    assert ann['category_id'] in catIdToCat
    assert ann['image_id'] in imageIdToImage

    imageIdToImage[ann['image_id']]['_count'] += 1
    catIdToCat[ann['category_id']]['_count'] +=1 
    
# ...for each annotation
    
    
#%% Print statistics
    
# Find un-annotated images and multi-annotation images
nUnannotated = 0
nMultiAnnotated = 0

for image in images:
    if image['_count'] == 0:
        nUnannotated += 1
    elif image['_count'] > 1:
        nMultiAnnotated += 1
        
print('Found {} unannotated images, {} images with multiple annotations'.format(
        nUnannotated,nMultiAnnotated))

nUnusedCategories = 0

# Find unused categories
for cat in categories:
    if cat['_count'] == 0:
        print('Unused category: {}'.format(cat['name']))
        nUnusedCategories += 1

print('Found {} unused categories'.format(nUnusedCategories))
        
# Prints a list of categories sorted by count

# https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary

sortedCategories = sorted(categories, key=itemgetter('_count'), reverse=True)

print('Categories and counts:\n')

for cat in sortedCategories:
    print('{:6} {}'.format(cat['_count'],cat['name']))

print('')
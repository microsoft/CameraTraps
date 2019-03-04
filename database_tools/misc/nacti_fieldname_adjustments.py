#
# nacti_fieldname_adjustments.py
#
# NACTI metadata was posted with "filename" in images instead of "file_name", and
# used string (rather than int) category IDs (in categories, but not in annotations).
#
# This script fixes those issues and rev's the version number.
#

#%% Constants and environment

import json
import os

inputJsonFile = r'/datadrive1/nacti_metadata_orig.json'
outputJsonFile = r'/datadrive1/nacti_metadata.json'

assert os.path.isfile(inputJsonFile)


#%% Read .json file

with open(inputJsonFile,'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']
info = data['info']

print('Finished reading input .json')


#%% Rev version number, update field names and types

assert(info['version'] == 1.0)
info['version'] = 1.1
nFilenameConversions = 0
nCatConversions = 0
nAnnConversions = 0

for image in images:
    
    assert 'filename' in image
    image['file_name'] = image['filename']
    del image['filename']
    nFilenameConversions += 1
    assert 'SEQ_NO' in image
    del image['SEQ_NO']
    assert 'IMG_WIDTH' in image and isinstance(image['IMG_WIDTH'],str)
    assert 'IMG_HEIGHT' in image and isinstance(image['IMG_HEIGHT'],str)
    image['width'] = int(image['IMG_WIDTH']) 
    image['height'] = int(image['IMG_HEIGHT']) 
    del image['IMG_WIDTH']
    del image['IMG_HEIGHT']
    
for cat in categories:
    
    assert 'id' in cat
    assert isinstance(cat['id'],str)
    cat['id'] = int(cat['id'])
    nCatConversions += 1

for ann in annotations:
    
    assert 'id' in ann    
    assert isinstance(ann['id'],str)
    assert 'category_id' in ann
    assert isinstance(ann['category_id'],str)
    ann['category_id'] = int(ann['category_id'])
    nAnnConversions += 1
    
print('Finished checking data, converted {} filename fields, {} category IDs, {} annotation category IDs'.format(
        nFilenameConversions,nCatConversions,nAnnConversions))


#%% Write json file
    
json.dump(data, open(outputJsonFile, 'w'))

print('Finished writing output .json to {}'.format(outputJsonFile))


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
    
    assert 'path' in image and isinstance(image['path'],str)
    image['file_name'] = image['path']
    del image['path']
    nFilenameConversions += 1
    assert 'seq_no' in image
    del image['seq_no']
    assert 'width' in image and isinstance(image['width'],str)
    assert 'height' in image and isinstance(image['height'],str)
    image['width'] = int(image['width']) 
    image['height'] = int(image['height']) 
    
for cat in categories:
    
    assert 'id' in cat and isinstance(cat['id'],str)
    cat['id'] = int(cat['id'])
    nCatConversions += 1

for ann in annotations:
    
    assert 'id' in ann and isinstance(ann['id'],str)
    assert 'category_id' in ann and isinstance(ann['category_id'],str)
    ann['category_id'] = int(ann['category_id'])
    nAnnConversions += 1
    
print('Finished checking data, converted {} filename fields, {} category IDs, {} annotation category IDs'.format(
        nFilenameConversions,nCatConversions,nAnnConversions))


#%% Write json file
    
json.dump(data, open(outputJsonFile, 'w'), indent=4)

print('Finished writing output .json to {}'.format(outputJsonFile))


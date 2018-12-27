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

inputJsonFile = r'd:\temp\nacti_metadata_orig.json'
outputJsonFile = r'd:\temp\nacti_metadata.json'

assert os.path.isfile(inputJsonFile)


#%% Read .json file

with open(inputJsonFile,'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']
info = data['info']


#%% Rev version number, update field names and types

assert(info['version']==1.0)
info['version'] = 1.01

for image in images:
    assert 'filename' in image
    image['file_name'] = image['filename']
    del image['filename']

for cat in categories:
    assert 'id' in cat
    assert isinstance(cat['id'],str)
    cat['id'] = int(cat['id'])
    
    
#%% Write json file
    
json.dump(data, open(outputJsonFile, 'w'))

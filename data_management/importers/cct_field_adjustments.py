#
# cct_field_adjustments.py
#
# CCT metadata was posted with int locations instead of strings.
#
# This script fixes those issues and rev's the version number.
#

#%% Constants and environment

from data_management.databases import sanity_check_json_db
import json
import os

inputJsonFile = r"D:\temp\CaltechCameraTraps_v2.0.json"
outputJsonFile = r"D:\temp\CaltechCameraTraps_v2.1.json"

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

assert(info['version'] == 'Caltech Camera Traps - v2')
info['version'] = 2.1
info['description'] = 'Caltech Camera Traps: camera trap images collected from the NPS and the USGS with help from Justin Brown and Erin Boydston'

for image in images:
    
    assert 'location' in image and isinstance(image['location'],int)
    image['location'] = str(image['location'])
    

#%% Write json file
    
json.dump(data, open(outputJsonFile, 'w'), indent=4)

print('Finished writing output .json to {}'.format(outputJsonFile))


#%% Check output data file

sanity_check_json_db.sanity_check_json_db(outputJsonFile)


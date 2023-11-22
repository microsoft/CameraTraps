#
# add_url_to_database.py
#
# Adds a "url" field to a coco-camera-traps .json database, specifically to allow the db to
# be reviewed in the Visipedia annotation tool.
#

import json

datafile = '/ai4efs/annotations/modified_annotations/imerit_ss_annotations_1.json'
url_base = 'https://s3-us-west-2.amazonaws.com/snapshotserengeti/'

with open(datafile, 'r') as f:
    data = json.load(f)

for im in data['images']:
    im['url'] = url_base + im['id'] + '.JPG'

json.dump(data, open(datafile, 'w'))



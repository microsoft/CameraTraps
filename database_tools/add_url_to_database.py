import json

datafile = '/ai4efs/annotations/modified_annotations/imerit_ss_annotations_1.json'
url_base = 'https://s3-us-west-2.amazonaws.com/snapshotserengeti/'

with open(datafile, 'r') as f:
    data = json.load(f)

for im in data['images']:
    im['url'] = url_base + im['id'] + '.JPG'

json.dump(data, open(datafile, 'w'))



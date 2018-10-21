import json
from PIL import Image

datafile = '/datadrive/snapshotserengeti/databases/imerit_ss_annotations_1.json'
image_base = '/datadrive/snapshotserengeti/images/'

with open(datafile,'r') as f:
    data = json.load(f)


for im in data['images']:
    if 'height' not in im:
        im_w, im_h = Image.open(image_base+im['file_name']).size
        im['height'] = im_h
        im['width'] = im_w


json.dump(data, open(datafile,'w'))


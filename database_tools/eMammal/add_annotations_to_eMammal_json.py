import json
import os
import re
import sys
from datetime import datetime, date

from tqdm import tqdm

# add_annotations_to_eMammal_json.py
#
# This scripts takes in the incomplete COCO formatted database with only image information and possibly
# annotations added previously, and add new annotations to it, saving the new version.


# configurations and paths

# if previous annotations are overwritten
overwrite_previous_annotations = True

# original version of the database to add annotations to; this will NOT be overwritten by this script
# unless OUTPUT_DB_PATH is set to the same path
json_db_path = '/home/yasiyu/yasiyu_temp/eMammal_db/eMammal_images.json'

# path to the new version of the database with annotations added
output_db_path = '/home/yasiyu/yasiyu_temp/eMammal_db/eMammal_20180929.json'

# where to save other info
other_output_dir = '/home/yasiyu/yasiyu_temp/eMammal_db/eMammal_20180929_others'
os.makedirs(other_output_dir, exist_ok=True)

# annotations obtained from our annotation vendor to be added to this database
new_annotation_paths = [
    os.path.join('/home/yasiyu/yasiyu_temp/eMammal_annotations/', f) for f in os.listdir('/home/yasiyu/yasiyu_temp/eMammal_annotations/')
]

start_time = datetime.now()

# load the original database
# should be a dict with 'info' and 'images' as keys, possibly also 'annotations' and 'categories'
original_db = json.load(open(json_db_path))

db_images = original_db.get('images', [])
assert len(db_images) > 0

db_info = original_db.get('info', [])
assert len(db_info) >= 5
db_info['version'] = '0.0.2'
db_info['description'] = 'eMammal dataset containing deployments in the McShea, Kays, and Long collections, in COCO format.'
db_info['date_created'] = str(date.today())

# map of all images' IDs in the ORIGINAL_DB to the label/category assigned to that image
# (which is the category assigned to the image sequence from which the image comes from)
db_image_ids = {
    i['id'] for i in db_images
}

default_categories = [
    {'id': 0, 'name': 'empty'},  # identifies the image as empty, which create_tfrecords_format will get rid of
    {'id': 1, 'name': 'animal'},
    {'id': 2, 'name': 'person'}
]

if overwrite_previous_annotations:
    db_annotations = []
    db_categories = default_categories
else:
    db_annotations = original_db.get('annotations', [])
    db_categories = original_db.get('categories', default_categories)

print('Number of annotations in original DB: {}'.format(len(db_annotations)))

# for each annotation pseudo-json, check that the image it refers to exists in the
# original database; use the image-level species to assign a category to each bounding box;
# if there are multiple species on that image (semi-column separated string),
# just leave it as such on all bounding boxes on the image; these can then be cleaned up later
image_ids_not_found = []
pattern = re.compile('^datasetemammal\.project(.+?)\.deployment(.+?)\.seq(.+?)[-_]frame(.+?)\.img(.+?)\.')

for annotation_path in new_annotation_paths:
    print('Now processing annotation batch {}...'.format(annotation_path))
    with open(annotation_path, 'r') as f:
        content = f.readlines()

    # each row in this pseudo-json is a COCO formatted entry for an image sequence
    for row in tqdm(content):
        entry = json.loads(row)

        # check that entry is for the eMammal dataset
        file_name = entry['images'][0]['file_name']
        if not file_name.startswith('datasetemammal.'):
            continue

        # check that 1 is animal, 2 is person
        row_categories = entry['categories']
        try:
            assert row_categories[0]['id'] == '1' and row_categories[0]['name'] == 'animal'
            assert row_categories[1]['id'] == '2' and row_categories[1]['name'] == 'person'
        except Exception as e:
            print(row_categories)

        annotations_entry = entry.get('annotations', [])

        # need to include an annotation entry for images that came back from labeling but were determined to
        # be empty to serve as confirmed negative examples
        if len(annotations_entry) == 0:
            empty_images = entry['images']
            for img in empty_images:
                # add a empty bbox_entry
                annotations_entry.append({
                    'image_id': img['file_name'],
                    'category_id': 0,
                    'id': '',
                    'bbox': []  # in create_tfrecords_format.py, empty array marks an empty image
                    })

        for bbox_entry in annotations_entry:
            # only annotate animal and person categories (no group), also empty images
            if int(bbox_entry['category_id']) not in [0, 1, 2]:
                continue

            # our img_id doesn't contain frame info
            match = pattern.match(bbox_entry['image_id'])
            project_id, deployment_id, seq_id, _, image_id = match.group(1, 2, 3, 4, 5)
            full_img_id = 'datasetemammal.project{}.deployment{}.seq{}.img{}'.format(
                project_id, deployment_id, seq_id, image_id)

            if full_img_id in db_image_ids:  # the image should exist in the original database
                db_annotations.append({
                    'image_id': full_img_id,
                    'category_id': int(bbox_entry['category_id']),
                    'id': bbox_entry['id'],
                    'bbox': bbox_entry['bbox']  # [top left x, top left y, relative width, relative height]
                })
            else:
                image_ids_not_found.append(full_img_id)

coco_formatted_json = {
    'info': db_info,
    'images': db_images,
    'annotations': db_annotations,
    'categories': db_categories
}

print('Number of annotations saved: {}'.format(len(db_annotations)))
print('Number of categories: {}'.format(len(db_categories)))

print('Saving the new json database to disk...')
with open(output_db_path, 'w') as f:
    json.dump(coco_formatted_json, f, indent=4, sort_keys=True)

with open(os.path.join(other_output_dir, 'image_ids_in_annotations_not_found.json'), 'w') as f:
    json.dump(image_ids_not_found, f, indent=4)

print('Running the script took {}.'.format(datetime.now() - start_time))


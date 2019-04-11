#
# add_bounding_boxes_to_json.py
#
# This script takes a image database in the COCO Camera Traps format and merges in a set of bounding box annotations
# in the format that iMerit uses (a .json where actually only each row is a valid json).
#
# If you need to update an existing bbox database, please get all the original annotation files and re-generate
# from scratch
#

import json
import re
from datetime import datetime, date
from tqdm import tqdm


#%% Configurations and paths

# images database
image_db_path = '/Users/siyuyang/Source/temp_data/CameraTrap/databases_201904/emammal_mcshea/emammal_mcshea_20190409.json'

# output bboxes database
new_bbox_db_path = '/Users/siyuyang/Source/temp_data/CameraTrap/databases_201904/emammal_mcshea/emammal_mcshea_bboxes_20190409.json'

# annotation files (pseudo json) obtained from our annotation vendor that contain annotations for this dataset
annotation_paths = [
    '/Users/siyuyang/Source/temp_data/CameraTrap/annotations/201904/microsoft_wildlife_reprocessing_1apr2019/batch4and6_new_image_id.json'
]

version_str = '20190409'
description = 'Bounding box annotations for the 0Bill McShea collection obtained through eMammal.'

# None or a string that is the prefix to all image_ids of interest / in this dataset in the annotation files
dataset_prefix = 'datasetemammalmcshea'

# functions for mapping the image_id in the annotation files (pseudo jsons) to the image_id used in the image DB

old_emammal_pattern = re.compile('^datasetemammal\.project(.+?)\.deployment(.+?)\.seq(.+?)[-_]frame(.+?)\.img(.+?)\.')
def old_emammal_annotation_image_file_name_to_db_image_id(image_file_name):
    # our img_id doesn't contain frame info
    match = old_emammal_pattern.match(image_file_name)
    project_id, deployment_id, seq_id, _, image_id = match.group(1, 2, 3, 4, 5)
    full_img_id = 'datasetemammal.project{}.deployment{}.seq{}.img{}'.format(
        project_id, deployment_id, seq_id, image_id)
    return full_img_id

def default_annotation_image_file_name_to_db_image_id(image_file_name):
    return image_file_name.split('.jpg')[0]

# specify which one to use for your dataset here
annotation_image_file_name_to_db_image_id = default_annotation_image_file_name_to_db_image_id


#%% Load the image database and fill in DB info for the output bbox database

start_time = datetime.now()

# load the images database
print('Loading the image database...')
image_db = json.load(open(image_db_path))
all_images = { i['id']: i for i in image_db['images'] }

db_info = image_db.get('info', [])
assert len(db_info) >= 5

if len(version_str) > 0:
    db_info['version'] = version_str
if len(description) > 0:
    db_info['description'] = description

db_info['date_created'] = str(date.today())


#%% Create the bbox database from all annotation files pertaining to this dataset

# the four categories for bounding boxes - do not change
bbox_categories = [
    {'id': 0, 'name': 'empty'},
    {'id': 1, 'name': 'animal'},
    {'id': 2, 'name': 'person'},
    {'id': 3, 'name': 'group'},  # group of animals
    {'id': 4, 'name': 'vehicle'}
]
# for the incoming annotations, look up by category name (common) and convert to the numerical id used in our databases
bbox_cat_map = { x['name']: x['id'] for x in bbox_categories }

db_annotations = []
db_images = []  # Only contain images sent for annotation, which could be confirmed to be empty

# for each annotation pseudo-json, check that the image it refers to exists in the original database
image_ids_not_found = []
num_bboxes = 0

for i_batch, annotation_path in enumerate(annotation_paths):
    print('Now processing annotation batch {}...'.format(annotation_path))
    with open(annotation_path, 'r') as f:
        content = f.readlines()

    # each row in this pseudo-json is a COCO formatted entry for an image sequence
    for i_row, row in tqdm(enumerate(content)):
        entry = json.loads(row)

        # check that entry is for this dataset
        file_name = entry['images'][0]['file_name']
        if dataset_prefix is not None and not file_name.startswith(dataset_prefix):
            continue

        # category map for this entry in the annotation file - usually the same across all entries but just in case
        anno_bbox_cat_map = { x['id']: x['name'] for x in entry['categories'] }
        anno_bbox_cat_map[0] = 'empty'  # on line 124 we assign category 0 to empty images

        annotations_entry = entry.get('annotations', [])

        for bbox_entry in annotations_entry:
            img_id = annotation_image_file_name_to_db_image_id(bbox_entry['image_id'])
            if img_id in all_images:  # the image should exist in the image database
                # use the image length and width in the image DB
                im_width, im_height = all_images[img_id]['width'], all_images[img_id]['height']
                if len(bbox_entry['bbox']) == 0:
                    bbox = []
                else:
                    # [top left x, top left y, width, height] in relative coordinates
                    rel_x, rel_y, rel_width, rel_height = bbox_entry['bbox']
                    x = rel_x * im_width
                    y = rel_y * im_height
                    w = rel_width * im_width
                    h = rel_height * im_height
                    bbox = [x, y, w, h]
                    num_bboxes += 1

                db_annotations.append({
                    'id': bbox_entry['id'],
                    'image_id': img_id,
                    'category_id': bbox_cat_map[anno_bbox_cat_map[bbox_entry['category_id']]],
                    'bbox': bbox  # [top left x, top left y, width, height] in absolute coordinates (floats)
                })
            else:
                image_ids_not_found.append(img_id)
                print('! Image in annotation not found in image DB: ', img_id)

        images_entry = entry.get('images', [])
        for im in images_entry:
            db_images.append(all_images[annotation_image_file_name_to_db_image_id(im['file_name'])])


coco_formatted_json = {
    'info': db_info,
    'images': db_images,
    'annotations': db_annotations,
    'categories': bbox_categories,
    'categories': bbox_categories
}

print('Number of bbox annotations in the bbox DB: {}'.format(num_bboxes))
print('Number of annotation entries in the bbox DB (some are empty of bbox): {}'.format(len(db_annotations)))
print('Number of images in the bbox DB: {}'.format(len(db_images)))

print('Saving the new json database to disk...')
with open(new_bbox_db_path, 'w') as f:
    json.dump(coco_formatted_json, f, indent=1)

if len(image_ids_not_found):
    print('Image ids in annotation but not found in the original DB:')
    for i in image_ids_not_found:
        print(i)

print('Running the script took {}.'.format(datetime.now() - start_time))

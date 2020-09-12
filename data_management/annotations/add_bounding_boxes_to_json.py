#
# add_bounding_boxes_to_json.py
#
# This script takes a image database in the COCO Camera Traps format and merges in a set of bounding 
# box annotations in the format that iMerit uses (a .json where actually only each row is a valid json).
#
# If you need to update an existing bbox database, please get all the original annotation files and 
# re-generate from scratch
#

#%% Imports

import json
import re
from datetime import datetime, date

from tqdm import tqdm

from data_management.annotations import annotation_constants


#%% Configurations and paths

# images database
image_db_path = '/Users/siyuyang/Source/temp_data/CameraTrap/databases_201904/caltechcameratraps/caltech_20190409.json'
image_db_has_dims = True  # does the image DB image entries have the height and width of the images

# output bboxes database
new_bbox_db_path = '/Users/siyuyang/Source/temp_data/CameraTrap/databases_201904/caltechcameratraps/caltech_bboxes_20190409.json'

# annotation files (pseudo json) obtained from our annotation vendor that contain annotations for this dataset
annotation_paths = [
    '/Users/siyuyang/Source/temp_data/CameraTrap/annotations/201904/microsoft_wildlife_reprocessing_1apr2019/batch1.json',
    '/Users/siyuyang/Source/temp_data/CameraTrap/annotations/201904/microsoft_wildlife_reprocessing_1apr2019/batch2.json'
]

version_str = '20190409'
description = 'Database of camera trap images and their bounding box annotations collected from the NPS and the USGS with help from Justin Brown and Erin Boydston.'

# None or a string or tuple of strings that is the prefix to all file_name of interest / in this dataset in the annotation files
dataset_prefix = ''

# functions for mapping the image_id in the annotation files (pseudo jsons) to the image_id used in the image DB

old_emammal_pattern = re.compile('^datasetemammal\.project(.+?)\.deployment(.+?)\.seq(.+?)[-_]frame(.+?)\.img(.+?)\.')
def old_emammal_anno_image_file_name_to_db_image_id(image_file_name):
    # our img_id doesn't contain frame info
    match = old_emammal_pattern.match(image_file_name)
    project_id, deployment_id, seq_id, _, image_id = match.group(1, 2, 3, 4, 5)
    full_img_id = 'datasetemammal.project{}.deployment{}.seq{}.img{}'.format(
        project_id, deployment_id, seq_id, image_id)
    return full_img_id

def idfg_anno_image_file_name_to_db_image_id(file_name):
    return file_name.replace('~', '/')

def ss_anno_image_file_name_to_db_image_id(file_name):
    # batch3 - "file_name":"ASG0000019_0_S1_B06_R1_PICT0007.JPG"
    # batch5 and 7 - "file_name":"datasetsnapshotserengeti.seqASG000002m-frame0.imgS1_B06_R1_PICT0056.JPG"
    # sometimes - 'datasetsnapshotserengeti.seqASG000001a.frame0.imgS1_B06_R1_PICT0008.JPG'
    # id in DB (old_token): 'S6/J01/J01_R1/S6_J01_R1_IMAG0001', 'S1/B05/B05_R1/S1_B05_R1_PICT0036'
    try:
        file_name = file_name.replace('-', '.')  # particularity with the frame section
        parts = file_name.split('.')

        if len(parts) == 5:
            tokens = parts[3].split('img')[1].split('_')
        elif len(parts) == 2:
            tokens = parts[0].split('_')[2:]

        old_token = tokens[0] + '/' + tokens[1] + '/' + tokens[1] + '_' + tokens[2] + '/' + '_'.join(tokens)
        return old_token
    except Exception as e:
        raise RuntimeError('Error with {}: {}'.format(file_name, e))

def rspb_annotation_image_file_name_to_db_image_id(image_file_name):
    return image_file_name.split('.JPG')[0]

def caltech_annotation_image_file_name_to_db_image_id(image_file_name):
    jpg_name = image_file_name.split('.img')[1]
    return jpg_name.split('.')[0]

def default_annotation_image_file_name_to_db_image_id(image_file_name):
    return image_file_name.split('.jpg')[0]

# specify which one to use for your dataset here
annotation_image_file_name_to_db_image_id = caltech_annotation_image_file_name_to_db_image_id


#%% Load the image database and fill in DB info for the output bbox database

start_time = datetime.now()

# load the images database
print('Loading the image database...')
image_db = json.load(open(image_db_path))
print('Image database loaded.')
all_images = { i['id']: i for i in image_db['images'] }

print(list(all_images.keys())[:10])

db_info = image_db.get('info', [])
assert len(db_info) >= 5

if len(version_str) > 0:
    db_info['version'] = version_str
if len(description) > 0:
    db_info['description'] = description

db_info['date_created'] = str(date.today())


#%% Find the height and width of images from the annotation files
#
# ...if they are not available in the images DB

if not image_db_has_dims:
    height_width_from_anno = {}
    for i_batch, annotation_path in enumerate(annotation_paths):
        with open(annotation_path, 'r') as f:
            content = f.readlines()

        # each row in this pseudo-json is a COCO formatted entry for an image sequence
        for i_row, row in tqdm(enumerate(content)):
            entry = json.loads(row)
            images_entry = entry.get('images', [])
            for im in images_entry:
                height_width_from_anno[im['file_name']] = {
                    'height': im['height'],
                    'width': im['width']
                }


#%% Other functions required by specific datasets

def idfg_add_image_entry(file_name):
    # the IDFG image database does not include images from unlabeled folders that were annotated with bounding boxes
    parts = file_name.split('/')
    if file_name.startswith('CrowCreek'):
        location = parts[1]
    elif file_name.startswith('Statewide'):
        location = parts[2]
    elif file_name.startswith('Focal'):
        location = parts[1]
    else:
        raise RuntimeError('image not in image DB and also not in the two specified folders:', file_name)
    all_images[file_name] = {
        'id': file_name,
        'file_name': file_name,
        'location': location  # TODO bug - location should be region_name + camera_location
    }

original_rspb_db = json.load(open('/Users/siyuyang/Source/temp_data/CameraTrap/databases_2018/rspb/rspb_gola.json'))
original_images = { i['id']: i for i in original_rspb_db['images']}
def rspb_add_image_entry(image_id):
    if image_id in original_images:
        all_images[image_id] = original_images[image_id]
    else:
        parts = image_id.split('__')
        file_name = image_id + '.JPG'
        all_images[image_id] = {
            'id': image_id,
            'file_name': file_name,
            'location': parts[0] + '__' + parts[1],
            'width': height_width_from_anno[file_name]['width'],
            'height': height_width_from_anno[file_name]['height']
        }


#%% Create the bbox database from all annotation files pertaining to this dataset

bbox_categories = annotation_constants.bbox_categories

# for the incoming annotations, look up by category name (common) and convert to the numerical id used in our databases
bbox_cat_map = { x['name']: x['id'] for x in bbox_categories }

db_annotations = []
db_images_dict = {}  # Only contain images sent for annotation, which could be confirmed to be empty

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
            print('Skipping ', file_name)
            continue

        # category map for this entry in the annotation file - usually the same across all entries but just in case
        anno_bbox_cat_map = { int(x['id']): x['name'] for x in entry['categories'] }

        annotations_entry = entry.get('annotations', [])

        for bbox_entry in annotations_entry:
            img_id = annotation_image_file_name_to_db_image_id(bbox_entry['image_id'])
            if img_id not in all_images:
                print('img_id not found: {}'.format(img_id))
                continue
                #rspb_add_image_entry(img_id)

            # use the image length and width in the image DB
            if image_db_has_dims:
                im_width, im_height = all_images[img_id]['width'], all_images[img_id]['height']
            else:
                im_width = height_width_from_anno[bbox_entry['image_id']]['width']
                im_height = height_width_from_anno[bbox_entry['image_id']]['height']

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
                'category_id': bbox_cat_map[anno_bbox_cat_map[int(bbox_entry['category_id'])]],
                'bbox': bbox  # [top left x, top left y, width, height] in absolute coordinates (floats)
            })

        # add all images that have been sent to annotation, some of which may be empty of bounding boxes
        images_entry = entry.get('images', [])
        for im in images_entry:
            db_image_id = annotation_image_file_name_to_db_image_id(im['file_name'])
            if db_image_id not in all_images:
                continue
                # rspb_add_image_entry(db_image_id)
            db_images_dict[db_image_id] = all_images[db_image_id]


coco_formatted_json = {
    'info': db_info,
    'images': list(db_images_dict.values()),
    'annotations': db_annotations,
    'categories': bbox_categories
}

print('Number of bbox annotations in the bbox DB: {}'.format(num_bboxes))
print('Number of annotation entries in the bbox DB (should be same as last line): {}'.format(len(db_annotations)))
print('Number of images in the bbox DB: {}'.format(len(db_images_dict)))

print('Saving the new json database to disk...')
with open(new_bbox_db_path, 'w') as f:
    json.dump(coco_formatted_json, f, indent=1)

if len(image_ids_not_found):
    print('{} image_ids with annotations are not found in the images DB.'.format(len(image_ids_not_found)))

print('Running the script took {}.'.format(datetime.now() - start_time))

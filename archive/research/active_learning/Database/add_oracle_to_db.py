'''
add_oracle_to_db.py

Uses crops.json file (from crop_images_from_batch_api_detection.py) and
COCO .json file to initialize and populate Oracle table in a PostgreSQL database.

'''

import argparse, json, os, string, time
from peewee import *
from DB_models import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_json', type=str, required=True, help='Path to a .json file with information about crops in crop_dir.')
    parser.add_argument('--coco_json', type=str, required=True, help='Path to .json file with COCO formatted information about detections.')
    parser.add_argument('--img_base', type=str, help='Path to add as prefix to filenames in COCO json.')
    parser.add_argument('--class_list', type=str, required=True, help='Path to .txt file containing a list of classes in the dataset.')
    parser.add_argument('--db_name', default='missouricameratraps', type=str, help='Name of the output Postgres DB.')
    parser.add_argument('--db_user', default='new_user', type=str, help='Name of the user accessing the Postgres DB.')
    parser.add_argument('--db_password', default='new_user_password', type=str, help='Password of the user accessing the Postgres DB.')
    args = parser.parse_args()

    crop_json = json.load(open(args.crop_json, 'r'))

    # Get class names from .txt list
    class_list = ['empty'] + [cname.lower() for cname in open(args.class_list, 'r').read().splitlines()]

    # Initialize Oracle table
    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password
    target_db = PostgresqlDatabase(DB_NAME, user=USER, password=PASSWORD, host='localhost')
    db_proxy.initialize(target_db)
    target_db.create_tables([Oracle])

    # Map filenames to classes (NOTE: we assume a single image does not contain more than one class)
    coco_json = json.load(open(args.coco_json, 'r'))
    coco_categories = {cat['id']:cat['name'].replace('_', ' ') for cat in coco_json['categories']}
    coco_imgid_to_fn = {im['id']: os.path.join(args.img_base,  im['file_name'].replace('\\', '/')) for im in coco_json['images']}
    coco_imgfn_to_catname = {}
    for ann in coco_json['annotations']:
        ann_imgid = ann['image_id']
        ann_imgfn = coco_imgid_to_fn[ann_imgid]
        ann_catid = ann['category_id']
        ann_catname = coco_categories[ann_catid]
        coco_imgfn_to_catname[ann_imgfn] = ann_catname

    # For each detection, use source image path to get class
    counter = 0
    timer = time.time()
    for crop in crop_json:
        counter += 1
        crop_info = crop_json[crop]
        source_image_file_name = crop_info['source_file_name']
        crop_class = coco_imgfn_to_catname[os.path.join(args.img_base , source_image_file_name)]
        existing_cat_entries = Category.select().where(Category.name == crop_class)
        try:
            existing_category_entry = existing_cat_entries.get()
            labelval = existing_category_entry.id
        except:
            print('Class %s not found in database Category table.'%crop_class)

        existing_oracle_entries = Oracle.select().where(Oracle.detection == crop)
        try:
            existing_oracle_entry = existing_oracle_entries.get()
        except:
            oracle_entry = Oracle.create(detection=crop, label=labelval)
            oracle_entry.save()
        
        if counter%100 == 0:
            print('Updated database with Oracle table entries for %d out of %d detections in %0.2f seconds'%(counter, len(crop_json), time.time() - timer))


if __name__ == '__main__':
    main()
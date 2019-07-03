'''
add_oracle_to_db.py

Uses .json file (from crop_images_from_batch_api_detection.py) to initialize and populate Oracle table in a PostgreSQL database.

'''

import argparse, json, os, string, time
from peewee import *
from DB_models import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_dir', type=str, required=True,
                        help='Path to a directory containing cropped images.')
    parser.add_argument('--crop_json', type=str, required=True,
                        help='Path to a .json file with information about crops in crop_dir.')
    parser.add_argument('--image_dir', type=str,
                        help='Path to root directory containing original images from which crops were generated.')
    parser.add_argument('--db_name', default='missouricameratraps', type=str,
                        help='Name of the output Postgres DB.')
    parser.add_argument('--db_user', default='new_user', type=str,
                        help='Name of the user accessing the Postgres DB.')
    parser.add_argument('--db_password', default='new_user_password', type=str,
                        help='Password of the user accessing the Postgres DB.')
    args = parser.parse_args()

    crop_json = json.load(open(args.crop_json, 'r'))
    sample_key = list(crop_json.keys())[0]
    print(crop_json[sample_key].keys())
    print(crop_json[sample_key]['source_file_name'])
    print(args.crop_dir)

    # Get class names from images root directory
    class_folders = os.listdir(args.image_dir)
    class_names_list = []
    for cf in class_folders:
        class_name = ''.join([i for i in cf if not i.isdigit()]).lower() # strip numbers
        class_name = class_name.lstrip(string.punctuation) # strip leading punctuation
        # class_name = class_name.replace('_', ' ') # replace underscore between words with space
        class_names_list.append(class_name)
    class_names_list = sorted(class_names_list)

    # Initialize Oracle table
    DB_NAME = args.db_name
    USER = args.db_user
    PASSWORD = args.db_password
    target_db = PostgresqlDatabase(DB_NAME, user=USER, password=PASSWORD, host='localhost')
    db_proxy.initialize(target_db)
    target_db.create_tables([Oracle])

    # For each detection, use source image path to get class
    counter = 0
    timer = time.time()
    for crop in crop_json:
        counter += 1
        crop_info = crop_json[crop]
        source_image_file_name = crop_info['source_file_name']
        crop_class = None
        for cn in class_names_list:
            cn_present = source_image_file_name.lower().find(cn) >= 0
            if cn_present:
                crop_class = cn
                break
        if crop_class is None:
            print('Class not identified for crop %s, from image %s'%(crop, source_image_file_name))
            continue
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
            print('Updated Oracle table entries for %d out of %d detections in %0.2f seconds'%(counter, len(crop_json), time.time() - timer))


if __name__ == '__main__':
    main()
'''
initialize_target_db.py

Creates a PostgreSQL database of camera trap images for use in active learning for classification.

Prerequisite steps:
- Create a PostgreSQL user and database:
    sudo -u postgres psql -c "CREATE USER <db_user> WITH PASSWORD <db_password>;"
    sudo -u postgres psql -c "CREATE DATABASE <db_name> WITH OWNER <db_user> CONNECTION LIMIT -1;"
    sudo -u postgres psql -c "GRANT CONNECT ON DATABASE <db_name> TO <db_user>;"
    sudo -u postgres psql -d <db_name> -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
- Create a folder containing crops as well as a crops.json file using crop_images_from_batch_api_detections.py.
- Create a .txt file containing 

Produces:
- A PostgreSQL database with the following tables:
    * info: information about the dataset
    * category: class names and corresponding ids
    * image: images present in the dataset
    * detections: crops of images with detections with confidence greater than the specified detector threshold

'''

import argparse, glob, json, os, psycopg2, time, uuid
from datetime import datetime
from peewee import *
from DB_models import *


parser = argparse.ArgumentParser(description='Initialize a PostgreSQL database for a dataset of camera trap images to use for active learning for classification.')
parser.add_argument('--db_name', default='missouricameratraps', type=str,
                    help='Name of the output Postgres DB.')
parser.add_argument('--db_user', type=str, required=True,
                    help='Name of the user accessing the Postgres DB.')
parser.add_argument('--db_password', type=str, required=True,
                    help='Password of the user accessing the Postgres DB.')
parser.add_argument('--crop_dir', metavar='DIR', required=True,
                    help='Path to dataset directory containing all cropped images')
parser.add_argument('--class_list', type=str, required=True,
                    help='Path to .txt file containing a list of classes in the dataset.')
args = parser.parse_args()

# Connect to database DB_NAME as USER and initialize tables
DB_NAME = args.db_name
USER = args.db_user
PASSWORD = args.db_password
target_db = PostgresqlDatabase(DB_NAME, user=USER, password=PASSWORD, host='localhost')
db_proxy.initialize(target_db)
target_db.create_tables([Info, Category, Image, Detection])



# Populate Info table
info_name = args.crop_dir
info_desc = 'Active learning for classification database of cropped images to classify via active learning (target dataset).'
info_contrib = 'Amrita'
info_version = 0
info_year = 2019
info_date = datetime.today().date()
existing_info_entries = Info.select().where(Info.name == info_name)
try:
    existing_info_entry = existing_info_entries.get()
except:
    info_entry = Info.create(name=info_name, description=info_desc, contributor=info_contrib, version=info_version, year=info_year, date_created=info_date)
    info_entry.save()

# Populate Category table
## For now, we have a predefined list of species we expect to see in the camera trap database (e.g. maybe from a quick look through the images)
## TODO: allow user to update the class list through the labeling tool UI as they see different species
class_list = ['empty'] + [cname.lower() for cname in open(args.class_list, 'r').read().splitlines()]
for i, cat in enumerate(class_list):
    existing_cat_entries = Category.select().where(Category.name == cat)
    try:
        existing_cat_entry = existing_cat_entries.get()
    except:
        cat_entry = Category.create(id=i, name=cat)
        cat_entry.save()

# Populate Image and Detection tables
with open(os.path.join(args.crop_dir,'crops.json'), 'r') as infile:
    crops_json = json.load(infile)

counter = 0
timer = time.time()
num_detections = len(crops_json)
for detectionid in crops_json:
    counter += 1
    detection_data = crops_json[detectionid]

    # Image entry data
    existing_image_entries = Image.select().where((Image.file_name == detection_data['file_name']))
    try:
        existing_image_entry = existing_image_entries.get()
    except:
        image_entry = Image.create(id=detectionid, file_name=detection_data['file_name'], width=detection_data['width'], height=detection_data['height'], grayscale=detection_data['grayscale'],
                                    source_file_name=detection_data['source_file_name'], relative_size=detection_data['relative_size'], 
                                    seq_id=detection_data['seq_id'], seq_num_frames=detection_data['seq_num_frames'], frame_num=detection_data['frame_num'])
        image_entry.save()
    
        # Detection entry data
        detection_entry = Detection.create(id=detectionid, image=detectionid, bbox_confidence=detection_data['bbox_confidence'], 
                                            bbox_X1=detection_data['bbox_X1'], bbox_Y1=detection_data['bbox_Y1'], bbox_X2=detection_data['bbox_X2'], bbox_Y2=detection_data['bbox_Y2'],
                                            kind=DetectionKind.ModelDetection.value)
        detection_entry.save()
    
    if counter%100 == 0:
        print('Updated database with Image and Detection table entries for %d out of %d crops in %0.2f seconds'%(counter, num_detections, time.time() - timer))


'''
initialize_target_db.py

Creates a PostgreSQL database of camera trap images for use in active learning for classification. Assumes that crops have already
been generated for the images using make_active_learning_classification_dataset.py. The created DB contains tables:
    - info: information about the dataset
    - image: images present in the dataset
    - detections: crops of images with detections with confidence greater than a specified threshold

'''

import argparse, glob, json, os, psycopg2, uuid
from datetime import datetime
from peewee import *
from DB_models import *


parser = argparse.ArgumentParser(description='Initialize a PostgreSQL database for a dataset of camera trap images to use for active learning for classification.')
parser.add_argument('--db_name', default='missouricameratraps', type=str,
                    help='Name of the output Postgres DB.')
parser.add_argument('--db_user', default='new_user', type=str,
                    help='Name of the user accessing the Postgres DB.')
parser.add_argument('--db_password', default='new_user_password', type=str,
                    help='Password of the user accessing the Postgres DB.')
parser.add_argument('--image_dir', metavar='DIR',
                    help='Path to dataset directory containing all images')
parser.add_argument('--coco_json', metavar='DIR',
                    help='Path to COCO Camera Traps json file if available', default=None)

args = parser.parse_args()

# Database connection credentials
DB_NAME = args.db_name
USER = args.db_user
PASSWORD = args.db_password
#HOST = 'localhost'
#PORT = 5432

# Options for getting dataset image data
COCO_CAMERA_TRAPS_JSON = args.coco_json # if image data is available in Coco Camera Traps JSON format
coco_json = None

# TODO: MAKE THIS SMOOTHER/MORE SECURE
# # DATABASE INITIALIZATION CODE
# # Connect to postgres database owned by suClass defining tperuser postgres
# conn = psycopg2.connect(dbname='postgres', user='postgres', password='postgres', host='localhost')
# conn.autocommit = True # this is needed to create database if it does not exist
# cursor = conn.cursor()

# # Check if the user USER with password PASSWORD exists
# query = "SELECT 1 FROM pg_roles WHERE rolname='%s'"%(USER)
# cursor.execute(query)
# qresult = cursor.fetchone()
# if qresult is None: # if not then create this user
#     query = "CREATE USER %s PASSWORD '%s';"%(USER, PASSWORD)
#     cursor.execute(query)
# else:
#     query = "ALTER USER %s WITH PASSWORD '%s'"%(USER, PASSWORD) # update the password in case it doesn't match, need to fix this later
#     cursor.execute(query)


# # Check if the database DB_NAME already exists
# query = "SELECT 1 FROM pg_catalog.pg_database WHERE datname = '%s'"%(DB_NAME)
# cursor.execute(query)
# qresult = cursor.fetchone()
# if qresult is None: # if not then create the database
#     query = "CREATE DATABASE %s"%(DB_NAME)
#     cursor.execute(query)
    
# # Grant USER access to DB_NAME
# query = "GRANT ALL PRIVILEGES ON DATABASE %s TO %s;"%(DB_NAME, USER)
# cursor.execute(query)

# # Close connection
# conn.close()


# SET UP TABLES
# Try to connect as USER to database DB_NAME through peewee
db = PostgresqlDatabase(DB_NAME, user=USER, password=PASSWORD, host='localhost')
db_proxy.initialize(db)
db.create_tables([Info, Image])

if COCO_CAMERA_TRAPS_JSON:
    # Use the metadata stored in COCO Camera Traps JSON formatted file
    coco_json = json.load(open(COCO_CAMERA_TRAPS_JSON, 'r'))

    # Info Table
    coco_json_info = coco_json['info']
    v = eval(coco_json_info['version'])
    y = coco_json_info['year']
    d = datetime.strptime(coco_json_info['date_created'], '%Y-%m-%d').date()

    # check if the info has already been entered (match by name, description and date created)
    info_entries = Info.select().where((Info.name==DB_NAME) & (Info.description == coco_json_info['description']) & (Info.date_created == d))
    try:
        info_entry = info_entries.get()
    except:
        info_entry = Info.create(name=DB_NAME, description=coco_json_info['description'], contributor= coco_json_info['contributor'], 
                        version=v, year=y, date_created=d)
        info_entry.save()


    # Image Table
    coco_json_images = coco_json['images'] # Data on all images in dataset
    imgs_in_dir = glob.glob(os.path.join(args.image_dir, '**/*.JPG'), recursive=True) # All images in directory (may be a subset of the dataset)
    for imgjson in sorted(coco_json_images, key = lambda i: i['file_name']):
        img_fn = imgjson['file_name'].replace('\\', '/')
        img_path = args.image_dir + img_fn
        if img_path in imgs_in_dir:
            img_id = imgjson['id']
            w = imgjson['width']
            h = imgjson['height']
            seq_id = imgjson['seq_id']
            seq_nf = imgjson['seq_num_frames']
            frame_num = imgjson['frame_num']

            # check if the image has already been entered into the table (match by id and file name)
            image_entries = Image.select().where((Image.id == img_id) & (Image.file_name == img_fn))
            try:
                image_entry = image_entries.get()
            except:
                image_entry = Image.create(id=img_id, file_name=img_fn, width=w, height=h, 
                                            seq_id=seq_id, seq_num_frames=seq_nf, frame_num=frame_num)
                image_entry.save()
    
            break # don't add all the images now while testing
    

    # Detections Table
    coco_json_anns = coco_json['annotations'] # Data on all annotations for all images in dataset
    print(coco_json_anns[0])

    
else:
    raise NotImplementedError('TODO: creating DB tables without using a COCO Camera Traps JSON file')
    # Info Table
    info = Info.create(name=DB_NAME, description='', contributor=USER, 
                    version=0.0, year=2019, date_created=datetime.today().date())
    info.save()

    # Image Table
    imgs_in_dir = glob.glob(os.path.join(args.source, '**/*.JPG'), recursive=True) # All images in directory
    for img in imgs_in_dir:
        img_id = uuid.uuid4()

# # img_paths = glob.glob(os.path.join(args.source, '**/*.JPG'), recursive=True)
# # coco_json = json.load(open(COCO_JSON, 'r'))
# # print(coco_json.keys())
# # print(coco_json['info'])




# RECYCLING
# db.connect()
# # Define some tables
# class Person(Model):
#     name = CharField()
#     birthday = DateField()
#     class Meta:
#         database = db

# class Pet(Model):
#     owner = ForeignKeyField(Person, backref='pets')
#     name = CharField()
#     animal_type = CharField()
#     class Meta:
#         database = db
# db.create_tables([Person, Pet])
# uncle_bob = Person(name='Bob', birthday=date(1960, 1, 15))
# uncle_bob.save()



# db = PostgresqlDatabase(
#     'active_learning_classification',
#     user='postgres',
#     password='postgres',
#     host='localhost',
#     port=5432
# )

# # db.connect()

# for person in Person.select():
#     print(person.name)

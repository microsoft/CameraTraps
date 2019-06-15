'''
initialize_pretrain_db.py

Creates a PostgreSQL database of camera trap images for use in active learning for classification. The script assumes that the images have been
used to prepare a classification dataset (i.e. the images have been cropped and organized into subfolders named by class)
a COCO JSON file is available for the dataset.
#  TODO update: Assumes that crops have already
# been generated for the images using make_active_learning_classification_dataset.py. The created DB contains tables:
#     - info: information about the dataset
#     - image: images present in the dataset
#     - detections: crops of images with detections with confidence greater than a specified threshold

'''

import argparse, glob, json, os, PIL.Image, psycopg2, sys, uuid
import numpy as np
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

# Initialize Database
## database connection credentials
DB_NAME = args.db_name
USER = args.db_user
PASSWORD = args.db_password
#HOST = 'localhost'
#PORT = 5432

## first, make sure the (user, password) has been created
## sudo -u postgres psql -c "CREATE USER <db_user> WITH PASSWORD <db_password>;"
## sudo -u postgres psql -c "CREATE DATABASE <db_name> WITH OWNER <db_user> CONNECTION LIMIT -1;"
## sudo -u postgres psql -c "GRANT CONNECT ON DATABASE <db_name> TO <db_user>;"
## sudo -u postgres psql -d <db_name> -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"

## Try to connect as USER to database DB_NAME through peewee
pretrain_db = PostgresqlDatabase(DB_NAME, user=USER, password=PASSWORD, host='localhost')
db_proxy.initialize(pretrain_db)
pretrain_db.create_tables([Info, Category, Image, Detection, Oracle])



# Populate Tables
## get class names
if sys.version_info >= (3, 5):
    # Faster anD available in Python 3.5 and above
    classes = [d.name for d in os.scandir(args.image_dir) if d.is_dir()]
else:
    classes = [d for d in os.listdir(args.image_dir) if os.path.isdir(os.path.join(args.image_dir, d))]
classes.sort()
class_to_idx = {classes[i]: i for i in range(len(classes))}

## iterate through images in each class folder
for cat in sorted(class_to_idx.keys()):
    existing_cat_entries = Category.select().where((Category.name == cat))
    try:
        existing_cat_entry = existing_cat_entries.get()
    except:
        category = Category.create(id = class_to_idx[cat], name = cat)
        category.save()
    
    image_list = []
    for root, _, fnames in sorted(os.walk(os.path.join(args.image_dir, cat))):
        for i, fname in enumerate(sorted(fnames)):
            if fname.endswith(".JPG"):
                image_uid = str(uuid.uuid4())
                image_fname = os.path.join(root, fname)
                image = np.array(PIL.Image.open(os.path.join(root, fname)))
                if image.dtype != np.uint8:
                    print('Failed to load image ' + fname)
                    continue
                img_width = image.shape[1]
                img_height = image.shape[0]
                # if mean of each channel is about the same, image is likely grayscale
                if (abs(np.mean(image[:,:,0]) - np.mean(image[:,:,1])) < 1e-1) & (abs(np.mean(image[:,:,1]) - np.mean(image[:,:,2])) < 1e-1):
                    img_grayscale = True
                else:
                    img_grayscale = False
                image_list.append((image_uid, image_fname, img_width, img_height, img_grayscale))
                
                # still have no info on these:
                # relative_size = FloatField(null = True) # cropped image size relative to original image size
                # source_file_name = CharField()      # full image file name from which cropped image was generated
                # seq_id = CharField(null= True)      # sequence identifier for the full image
                # seq_num_frames = IntegerField(null = True)  # number of frames in sequence
                # frame_num = IntegerField(null = True)       # which frame number in sequence
                # location = CharField(null = True)   # location of camera trap
                # datetime = DateTimeField(null = True)
                    
                

print(classes)
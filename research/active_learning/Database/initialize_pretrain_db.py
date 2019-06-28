'''
initialize_pretrain_db.py

(Largely draws from CameraTraps/research/active_learning/import_folder.py)

Creates a PostgreSQL database of camera trap images for use in active learning for classification. The script assumes that the images have been
used to prepare a classification dataset (i.e. the images have been cropped and organized into subfolders named by class).


a COCO JSON file is available for the dataset.
#  TODO update: Assumes that crops have already
# been generated for the images using make_active_learning_classification_dataset.py. The created DB contains tables:
#     - info: information about the dataset
#     - image: images present in the dataset
#     - detections: crops of images with detections with confidence greater than a specified threshold

'''

import argparse, glob, json, os, PIL.Image, psycopg2, sys, time, uuid
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
parser.add_argument('--crop_dir', metavar='DIR', required=True,
                    help='Path to dataset directory containing all cropped images')
parser.add_argument('--image_dir', metavar='DIR',
                    help='Path to dataset directory containing all original images')
parser.add_argument('--verbose', default=False, type=bool,
                    help='Output print messages while running')
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

## create Info table
info_name = args.crop_dir
info_desc = 'Active learning for classification database of cropped images used to train embedding model (pretraining dataset).'
info_contrib = 'Amrita'
info_version = 0
info_year = 2019
info_date = datetime.today().date()
existing_info_entries = Info.select().where(Info.name == info_name)
try:
    existing_info_entry = existing_info_entries.get()
except:
    info_entry = Info.create(name = info_name, description = info_desc, contributor = info_contrib, version = info_version, year = info_year, date_created = info_date)
    info_entry.save()

## get class names for Category table
if sys.version_info >= (3, 5):
    # Faster anD available in Python 3.5 and above
    classes = [d.name for d in os.scandir(args.crop_dir) if d.is_dir()]
else:
    classes = [d for d in os.listdir(args.crop_dir) if os.path.isdir(os.path.join(args.crop_dir, d))]
classes.sort()
class_to_idx = {classes[i]: i for i in range(len(classes))}

## iterate through images in each class folder
for cat in sorted(class_to_idx.keys()):
    # killing this process after over 38 hours adding over 500k white-tailed deer crops from emammal
    # resuming for remaining classes
    if cat not in ['wild turkey']:
        continue
    if args.verbose:
        print("Start processing "+cat)
        timer = time.time()

    existing_cat_entries = Category.select().where((Category.name == cat))
    try:
        existing_cat_entry = existing_cat_entries.get()
    except:
        category_entry = Category.create(id = class_to_idx[cat], name = cat)
        category_entry.save()
    
    for root, _, fnames in sorted(os.walk(os.path.join(args.crop_dir, cat))):
        for i, fname in enumerate(sorted(fnames)):
            if fname.endswith(".JPG"):
                ## get cropped image data for Image table
                img_uid = str(uuid.uuid4())
                img_fname = os.path.join(root, fname)
                img = np.array(PIL.Image.open(os.path.join(root, fname)))
                if img.dtype != np.uint8:
                    print('Failed to load image ' + fname)
                    continue
                img_width = img.shape[1]
                img_height = img.shape[0]
                # if mean of each channel is about the same, image is likely grayscale
                if (abs(np.mean(img[:,:,0]) - np.mean(img[:,:,1])) < 1e-1) & (abs(np.mean(img[:,:,1]) - np.mean(img[:,:,2])) < 1e-1):
                    img_grayscale = True
                else:
                    img_grayscale = False
                if len(fname.split('.JPG')[-2].split('_')) > 1:
                    img_source_fname = os.path.join(args.image_dir, os.path.basename(root), fname.split('.JPG')[-2].split('_')[0]+'.JPG')
                else:
                    img_source_fname = os.path.join(args.image_dir, os.path.basename(root), fname)

                orig_img = np.array(PIL.Image.open(img_source_fname))
                orig_img_width = orig_img.shape[1]
                orig_img_height = orig_img.shape[0]
                img_relative_size = (img_height*img_width)/float(orig_img_height*orig_img_width)
                ## still have no info on these:
                # seq_id = CharField(null= True)                # sequence identifier for the original image
                # seq_num_frames = IntegerField(null = True)    # number of frames in sequence
                # frame_num = IntegerField(null = True)         # which frame number in sequence
                # location = CharField(null = True)             # location of camera trap
                # datetime = DateTimeField(null = True)
                
                existing_image_entries = Image.select().where((Image.file_name == img_fname) & (Image.width == img_width) & (Image.height == img_height))
                try:
                    existing_image_entry = existing_image_entries.get()
                except:
                    image_entry = Image.create(id = img_uid, file_name = img_fname, width = img_width, height = img_height, grayscale = img_grayscale,
                                                source_file_name = img_source_fname, relative_size = img_relative_size)
                    image_entry.save()

                    ## store info about the detection corresponding to this image
                    detection_uid = str(uuid.uuid4())
                    detection_img = img_uid
                    detection_kind = DetectionKind.UserDetection.value  # pretrain dataset has user-provided annotations; ALTHOUGH crops were actually produced using detector...
                    detection_cat = class_to_idx[cat]
                    detection_cat_conf = 1
                    detection_entry = Detection.create(id = detection_uid, image = detection_img, kind = detection_kind, category = detection_cat, category_confidence = detection_cat_conf)
                    detection_entry.save()

                    ## store info about the true labels for the detection
                    ##  - for pretrain dataset this is the same as the detection_category if the detection categories
                    oracle_entry = Oracle.create(detection = detection_uid, label = detection_cat)
                
            if i%10==0 and i>0 and args.verbose:
                print('Processed %d files in %0.2f seconds'%(i, time.time() - timer))
                
                    
                

# print(classes)
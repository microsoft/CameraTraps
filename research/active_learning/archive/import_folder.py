from peewee import *
import sys
import os
import argparse
import datetime
import uuid
import gc
from shutil import copyfile

from UIComponents.DBObjects import *

parser = argparse.ArgumentParser(description='Import folder to DB')
parser.add_argument('--source', metavar='DIR',
                    help='path to the images folder', default='all_crops/crops_train')
parser.add_argument('--db_name', default='destination', type=str, metavar='PATH',
                    help='Name of the output SQLite DB.')


args = parser.parse_args()
os.mkdir(args.db_name)
os.mkdir(os.path.join(args.db_name,"crops"))
db = SqliteDatabase(os.path.join(args.db_name,args.db_name+'.db'))
proxy.initialize(db)
db.create_tables([Detection, Category, Image, Info, Oracle])
if sys.version_info >= (3, 5):
    # Faster and available in Python 3.5 and above
    classes = [d.name for d in os.scandir(args.source) if d.is_dir()]
else:
    classes = [d for d in os.listdir(args.source) if os.path.isdir(os.path.join(args.source, d))]
classes.sort()
class_to_idx = {classes[i]: i for i in range(len(classes))}
info = Info.create(name=args.source, description= '', contributor= 'Arash', version= 0, year= 2019, date_created= datetime.datetime.today())
info.save()

for cat in sorted(class_to_idx.keys()):
    print("Start processing "+cat)
    category= Category.create(id= class_to_idx[cat], name= cat, abbr= cat[0:2])
    category.save()
    image_list= []
    detection_list= []
    oracle_list= []
    for root, _, fnames in sorted(os.walk(os.path.join(args.source, cat))):
        for i, fname in enumerate(fnames):
            if fname.endswith(".JPG"):
                path = os.path.join(root, fname)
                image_id= str(uuid.uuid1())
                image_list.append((image_id, fname))
                copyfile(path,os.path.join(args.db_name,"crops",image_id+".JPG"))
                detection_list.append((image_id, image_id, DetectionKind.UserDetection.value, category.id, 1, 1, 0, 0, 1, 1))
                oracle_list.append((image_id, category.id))
            if i%1000==0 and i>0:
                print('Processed %d of %d '%(i,len(fnames)))
                Image.insert_many(image_list, fields= [Image.id, Image.file_name]).execute()
                Detection.insert_many(detection_list, fields= [Detection.id, Detection.image, Detection.kind, Detection.category, Detection.category_confidence, Detection.bbox_confidence,
                    Detection.bbox_X1, Detection.bbox_Y1, Detection.bbox_X2, Detection.bbox_Y2]).execute()
                Oracle.insert_many(oracle_list, fields= [Oracle.detection, Oracle.label]).execute()
                image_list= []
                detection_list= []
                oracle_list= []
                gc.collect()
        Image.insert_many(image_list, fields= [Image.id, Image.file_name]).execute()
        Detection.insert_many(detection_list, fields= [Detection.id, Detection.image, Detection.kind, Detection.category, Detection.category_confidence, Detection.bbox_confidence,
        Detection.bbox_X1, Detection.bbox_Y1, Detection.bbox_X2, Detection.bbox_Y2]).execute()
        Oracle.insert_many(oracle_list, fields= [Oracle.detection, Oracle.label]).execute()
        image_list= []
        detection_list= []
        oracle_list= []
        gc.collect()


    print(cat+" is done.")

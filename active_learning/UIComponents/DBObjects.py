#
# DBObjects.py
#
# Classes connecting the active learning framework to the underlying SQLite
# database; this file defines the DB schema.
#

#%% Constants, globals, and imports

from peewee import *
from enum import Enum

proxy = Proxy()


#%% Classes defining DB elements

class DetectionKind(Enum):
    ModelDetection = 0
    ActiveDetection = 1
    UserDetection = 2


class Image(Model):
    id = CharField(primary_key=True)
    file_name = CharField()
    #label= CharField()
    width = IntegerField(null = True)
    height = IntegerField(null= True)
    location = CharField(null = True)
    datetime = DateTimeField(null = True)
    frame_num = IntegerField(null = True)
    seq_id = CharField()
    seq_num_frames = IntegerField(null = True)
    
    class Meta:
      database= proxy


class Info(Model):
    name = CharField()
    description = CharField()
    contributor = CharField()
    version = IntegerField()
    year = IntegerField()
    date_created = DateField()
    RM = FloatField(null = True, default= -1)
    GM = FloatField(null = True, default= -1)
    BM = FloatField(null = True, default= -1)
    RS = FloatField(null = True, default= -1)
    GS = FloatField(null = True, default= -1)
    BS = FloatField(null = True, default= -1)

    class Meta:
      database = proxy


class Category(Model):
    id = IntegerField(primary_key=True)
    name = CharField()

    class Meta:
      database= proxy

class Detection(Model):
    id = CharField(primary_key=True)
    image = ForeignKeyField(Image)
    kind= IntegerField()
    category = ForeignKeyField(Category)
    category_confidence = FloatField(null= True)
    bbox_confidence = FloatField(null= True)
    bbox_X1 = FloatField(null = True)
    bbox_Y1 = FloatField(null = True)
    bbox_X2 = FloatField(null = True)
    bbox_Y2 = FloatField(null = True)

    class Meta:
      database = proxy


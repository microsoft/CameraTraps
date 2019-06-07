'''
DB_models.py

Defines model classes representing database tables used in active learning for classification.

'''

from peewee import *


db_proxy = Proxy() # create a proxy backend for the database


class Info(Model):
    '''
    Class defining table containing information about the target dataset.
    '''
    name = CharField() # name of dataset
    description= CharField()
    contributor= CharField()
    version= IntegerField()
    year= IntegerField()
    date_created = DateField()
    RM= FloatField(null = True, default= -1) # mean in RGB channels(?)
    GM= FloatField(null = True, default= -1)
    BM= FloatField(null = True, default= -1)
    RS= FloatField(null = True, default= -1) # standard deviation in RGB channels (?)
    GS= FloatField(null = True, default= -1)
    BS= FloatField(null = True, default= -1)

    class Meta:
        database = db_proxy
#######
#
# cct_json_utils.py
#
# Utilities for working with COCO Camera Traps .json databases
#
# Format spec:
#
# https://github.com/Microsoft/CameraTraps/blob/master/data_management/README.md#coco-cameratraps-format
#
#######


#%% Constants and imports

import os
import json
from collections import defaultdict


#%% Classes

class CameraTrapJsonUtils:
    """
    Miscellaneous utility functions for working with COCO Camera Traps databases
    """
    
    def annotationsToString(annotations,cat_id_to_name):
        """
        Given a list of annotations and a mapping from class IDs to names, produces
        a concatenated class list, always sorting alphabetically.
        """
        class_names = set()
        for ann in annotations:
            category_id = ann['category_id']
            category_name = cat_id_to_name[category_id]
            class_names.add(category_name)
        class_names = list(class_names)
        class_names.sort()
        return ','.join(class_names)
            
        
class IndexedJsonDb:
    """
    Wrapper for a COCO Camera Traps database.
    
    Handles boilerplate dictionary creation that we do almost every time we load 
    a .json database.
    """
    
    # The underlying .json db
    db = None
    
    # Useful dictionaries
    cat_id_to_name = None
    cat_name_to_id = None
    filename_to_id = None
    image_id_to_annotations = None

    def __init__(self,jsonFilename,b_normalize_paths=False,filename_replacements={}):
       
        self.db = json.load(open(jsonFilename))
    
        assert 'images' in self.db, 'Could not find image list in file {}, are you sure this is a COCO camera traps file?'.format(jsonFilename)
        
        if b_normalize_paths:
            # Normalize paths to simplify comparisons later
            for im in self.db['images']:
                im['file_name'] = os.path.normpath(im['file_name'])
        
        for s in filename_replacements:
            r = filename_replacements[s]
            for im in self.db['images']:
                im['file_name'] = im['file_name'].replace(s,r)
        
        ### Build useful mappings to facilitate working with the DB
        
        # Category ID <--> name
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.db['categories']}
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in self.db['categories']}
        
        # Image filename --> ID
        self.filename_to_id = {im['file_name']: im['id'] for im in self.db['images']}
        
        # Each image can potentially multiple annotations, hence using lists
        self.image_id_to_annotations = defaultdict(list)
        
        # Image ID --> image object
        self.image_id_to_image = {im['id'] : im for im in self.db['images']}
        
        # Image ID --> annotations
        for ann in self.db['annotations']:
            self.image_id_to_annotations[ann['image_id']].append(ann)
            
            

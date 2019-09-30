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
from collections import defaultdict, OrderedDict


#%% Classes

class CameraTrapJsonUtils:
    """
    Miscellaneous utility functions for working with COCO Camera Traps databases
    """
    @staticmethod
    def annotations_to_string(annotations, cat_id_to_name):
        """
        Given a list of annotations and a mapping from class IDs to names, produces
        a concatenated class list, always sorting alphabetically.
        """
        class_names = CameraTrapJsonUtils.annotationsToClassnames(annotations, cat_id_to_name)
        return ','.join(class_names)

    @staticmethod
    def annotations_to_classnames(annotations, cat_id_to_name):
        """
        Given a list of annotations and a mapping from class IDs to names, produces
        a list of class names, always sorting alphabetically.
        """
        # Collect all names
        class_names = [cat_id_to_name[ann['category_id']] for ann in annotations]
        # Make names unique and sort
        class_names = sorted(list(set(class_names)))
        return class_names

    @staticmethod
    def order_db_keys(db):
        """
        Given a dict representing a JSON database in the COCO Camera Trap format, return an
        OrderedDict with keys in the order of 'info', 'categories', 'annotations' and 'images'.
        When this OrderedDict is serialized with json.dump(), the order of the keys are preserved.
        Args:
            db: a dict representing a JSON database in the COCO Camera Trap format

        Returns:
            the same db but as an OrderedDict with keys ordered for readability.
        """
        ordered = OrderedDict([
            ('info', db['info']),
            ('categories', db['categories']),
            ('annotations', db['annotations']),
            ('images', db['images'])])
        return ordered

    @staticmethod
    def get_entries_from_locations(db, locations):
        """
        Given a dict representing a JSON database in the COCO Camera Trap format, return a dict
        with the 'images' and 'annotations' fields in the CCT format, each is an array that only
        includes entries in the original `db` that are in the `locations` set.
        Args:
            db: a dict representing a JSON database in the COCO Camera Trap format
            locations: a set or list of locations to include; each item is a string

        Returns:
            a dict with the 'images' and 'annotations' fields in the CCT format
        """
        locations = set(locations)
        print('Original DB has {} image and {} annotation entries.'.format(len(db['images']), len(db['annotations'])))
        new_db = {
            'images': [],
            'annotations': []
        }
        new_images = set()
        for i in db['images']:
            # cast location to string as the entries in locations are strings
            if str(i['location']) in locations:
                new_db['images'].append(i)
                new_images.add(i['id'])
        for a in db['annotations']:
            if a['image_id'] in new_images:
                new_db['annotations'].append(a)
        print(
            'New DB has {} image and {} annotation entries.'.format(len(new_db['images']), len(new_db['annotations'])))
        return new_db


class IndexedJsonDb:
    """
    Wrapper for a COCO Camera Traps database.

    Handles boilerplate dictionary creation that we do almost every time we load
    a .json database.
    """

    def get_annotations_for_image(self, image):
        """
        Returns a list of annotations associated with [image]
        
        Returns None is the db has not been loaded, [] if no annotations are available
        """
        
        if self.db is None:
            return None
    
        if image['id'] not in self.image_id_to_annotations:
            return []
        
        image_annotations = self.image_id_to_annotations[image['id']]
        return image_annotations
    
    
    def get_classes_for_image(self, image):
        """
        Returns a list of class names associated with [image]
        
        Returns None is the db has not been loaded, [] if no annotations are available
        """
        
        if self.db is None:
            return None
    
        if image['id'] not in self.image_id_to_annotations:
            return []
        
        class_ids = []
        image_annotations = self.image_id_to_annotations[image['id']]
        for ann in image_annotations:
            class_ids.append(ann['category_id'])
        class_ids = list(set(class_ids))
        class_ids.sort()
        class_names = [self.cat_id_to_name[x] for x in class_ids]
        
        return class_names
        
        
    def __init__(self, json_filename, b_normalize_paths=False, filename_replacements={}):
        '''
        json_filename can also be an existing json db
        '''
        
        if isinstance(json_filename,str):
            self.db = json.load(open(json_filename))
        else:
            self.db = json_filename
    
        assert 'images' in self.db, 'Could not find image list in file {}, are you sure this is a COCO camera traps file?'.format(json_filename)
        
        if b_normalize_paths:
            # Normalize paths to simplify comparisons later
            for im in self.db['images']:
                im['file_name'] = os.path.normpath(im['file_name'])

        for s in filename_replacements:
            r = filename_replacements[s]
            for im in self.db['images']:
                im['file_name'] = im['file_name'].replace(s, r)
        
        ### Build useful mappings to facilitate working with the DB

        # Category ID <--> name
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.db['categories']}
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in self.db['categories']}

        # Image filename --> ID
        self.filename_to_id = {im['file_name']: im['id'] for im in self.db['images']}

        # Each image can potentially multiple annotations, hence using lists
        self.image_id_to_annotations = defaultdict(list)

        # Image ID --> image object
        self.image_id_to_image = {im['id']: im for im in self.db['images']}
        
        # Image ID --> annotations
        for ann in self.db['annotations']:
            self.image_id_to_annotations[ann['image_id']].append(ann)

    # ...__init__

# ...class IndexedJsonDb

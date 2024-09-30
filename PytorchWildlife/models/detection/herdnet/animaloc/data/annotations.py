__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import pandas
import warnings

from typing import List, Optional, Union
from operator import itemgetter

from .types import Point, BoundingBox

__all__ = ['Annotations', 'AnnotationsFromCSV','objects_from_df', 'dict_from_objects']

def objects_from_df(df: pandas.DataFrame) -> List[Union[Point, BoundingBox]]:
    ''' Function to convert object coordinates to an appropried object type

    Args:
        df (pandas.DataFrame): DataFrame with 
            a header of this type for Point:
                | 'x' | 'y' |
            and a header of this type for BoundingBox
                | 'x_min' | 'y_min' | 'x_max' | 'y_max' |
    
    Returns:
        list:
            list of objects (Point or BoundingBox)
    '''

    # Point
    if {'x','y'}.issubset(df.columns):
        data = df[['x','y']]
        objects = [Point(r.x, r.y) for i,r in data.iterrows()]

    # BoundingBox
    elif {'x_min','y_min','x_max','y_max'}.issubset(df.columns):
        data = df[['x_min','y_min','x_max','y_max']]
        objects = [BoundingBox(r.x_min, r.y_min, r.x_max, r.y_max) for i,r in data.iterrows()]
    
    else:
        raise Exception('Wrong columns\' names for defining the objects in DataFrame. ' \
                        'Define x and y columns for Point object ' \
                        'or define x_min, y_min, x_max and y_max columns for BoundingBox object.')
    
    return objects

def dict_from_objects(obj_list: List[Union[Point, BoundingBox]]) -> List[dict]:
    ''' Function to convert a list of objects to corresponding coordinates, 
    stored in a list of dict

    Args:
        obj_list (list): list of objects (Point or BoundingBox)

    Returns:
        List[dict]:
            List of dict with a header keys of this type for Point:
                | 'x' | 'y' |
            and a header keys of this type for BoundingBox
                | 'x_min' | 'y_min' | 'x_max' | 'y_max' |
    '''

    assert all(isinstance(o, (Point, BoundingBox)) for o in obj_list) is True, \
        'Objects must be Point or BoundingBox instances.'
    
    # Point
    if isinstance(obj_list[0], Point):
        data = [{'x': o.x, 'y': o.y} for o in obj_list]

    # BoundingBox
    elif isinstance(obj_list[0], BoundingBox):
        data = [
            {'x_min': o.x_min, 'y_min': o.y_min, 'x_max': o.x_max, 'y_max': o.y_max}
            for o in obj_list
        ]
    
    return data

class Annotations:
    ''' Class to create an Annotations object '''

    def __init__(
        self, 
        images: Union[str, List[str]], 
        annos: List[Union[Point, BoundingBox]], 
        labels: List[int], 
        **kwargs
        ) -> None:
        '''
        Args:
            images (str or list): image name (str) or list of images' names. If
                image name is given, it will be used to define all annotations
            annos (list): list of Point or BoundingBox objects
            labels (list): list of labels id
            **kwargs (optional): additional data, value must be a list of the same 
                length than mandatory arguments
        '''

        self.images = images
        self.annos = annos
        self.labels = labels
        self.__dict__.update(kwargs)

        if all([len(v) > 0 for v in self.__dict__.values()]):
            assert all(isinstance(o, BoundingBox) for o in annos) is True \
                or all(isinstance(o, Point) for o in annos), \
                'annos argument must be a list composed of Point or BoundingBox instances'
            
            assert all(isinstance(lab, int) for lab in labels) is True, \
                'labels argument must be a list composed of integer only'

            self.images = images
            if isinstance(images, list):
                assert all(isinstance(im, str) for im in images) is True, \
                    'images argument must be a list composed of string only'
            elif isinstance(images, str):
                self.images = [images]*len(annos)
            else: 
                raise ValueError('images argument must be a string or a list of string')

            self.annos = annos
            self.labels = labels
            self.__dict__.update(kwargs)
                
            # assert that all variables have the same length
            assert len(set([len(v) for v in self.__dict__.values()])) == 1, \
                'All arguments must have the same length'
        
        else:
            warnings.warn('Empty Annotations object created')
    
    @property
    def dataframe(self) -> pandas.DataFrame:
        ''' To get annotations in Pandas DataFrame 

        Returns:
            pandas.DataFrame
        '''
        
        return pandas.DataFrame(data = self.__dict__)
    
    def sort(
        self, 
        attr: str, 
        keep: Optional[str] = None, 
        reverse: bool = False
        ) -> None:
        ''' Sort the object attributes while keeping the values of an attribute 
        grouped or not.

        Args:
            attr (str): attribute to sort
            keep (str, optional): attribute to keep grouped. Defaults to None
            reverse (bool, optional): set to True for descending sort. Defaults to
                False
        '''

        assert attr in self.__dict__.keys(), \
            f'{attr} is not an attribute of the object'
        
        if isinstance(keep, str):
            assert keep in self.__dict__.keys(), \
                f'{keep} is not an attribute of the object'

        all_attr = [a for a in self.__iter__()]
        
        # sort attr first, then the attribute to keep (ascending order)
        specs = [(attr, reverse)]
        if keep is not None : 
            specs = [(attr, reverse), (keep, False)]

        for key, reverse in specs:
            all_attr.sort(key=itemgetter(key), reverse=reverse)
        
        # update
        keys = self.__dict__.keys()
        for key in keys:
            sorted_list = [row[key] for row in all_attr]
            self.__dict__.update({key: sorted_list})
    
    def sub(self, image_name: str):
        ''' Returns an Annotations sub-object by selecting the items that 
        contain the specified image name

        Args: 
            image_name (str): the image name with extension
        
        Returns:
            Annotations
        '''

        new_kwargs = {}

        image_idx = [i for i, _ in enumerate(self.images) if self.images[i]==image_name]
        for key, values in self.__dict__.items():
            new_values = [values[i] for i in image_idx]
            new_kwargs.update({key: new_values})
        
        return Annotations(**new_kwargs)
    
    def get_supp_args_names(self):
        supp_args_names = []
        for key, values in self.__dict__.items():
            if key not in ['annos','images','labels']:
                supp_args_names.append(key)
        
        return supp_args_names

    def __iter__(self) -> dict:
        for i in range(len(self.images)):
            out_dict = {}
            for key in self.__dict__.keys():
                out_dict.update({key: self.__dict__[key][i]})
            
            yield out_dict
    
    def __getitem__(self, index) -> dict:
        out_dict = {}
        for key in self.__dict__.keys():
            out_dict.update({key: self.__dict__[key][index]})

        return out_dict
    
    def __len__(self) -> int:
        return len(self.images)

class AnnotationsFromCSV(Annotations):
    ''' Class to create annotations object from a CSV file
    
    Inheritance of Annotations class.

    The CSV file must have, at least, a header of this type for points:
    | 'images' | 'x' | 'y' | 'labels' |

    and of this type for bounding box:
    | 'images' | 'x_min' | 'y_min' | 'x_max' | 'y_max' | 'labels' |

    Other columns containing other information may be present. 
    In such a case, these will be kept and linked to the necessary basic content.
    '''

    def __init__(self, csv: Union[str,pandas.DataFrame]) -> None:
        '''
        Args:
            csv (str or pandas.DataFrame): absolute path to the CSV file (with extension),
                or DataFrame object.
        '''

        assert isinstance(csv, (str, pandas.DataFrame)), \
            'csv argument must be a string (absolute path with extension) ' \
            'or a pandas.DataFrame.'
        
        data_df = csv
        if isinstance(csv, str):
            data_df = pandas.read_csv(csv)

        assert {'images','labels'}.issubset(data_df.columns), \
            'File must contain at least images and labels columns name'
        
        images = list(data_df['images'])
        labels = list(data_df['labels'])
        annos = objects_from_df(data_df)

        # get other information
        supp_arg = {}
        for column, content in data_df.items():
            if column not in ['images','labels'] and column.startswith(('x','y')) is False:
                supp_arg.update({column: list(content)})

        super(AnnotationsFromCSV, self).__init__(images, annos, labels)
        if supp_arg:
            super(AnnotationsFromCSV, self).__init__(images, annos, labels, **supp_arg)
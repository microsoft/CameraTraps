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


from typing import Union

from .types import Point, BoundingBox

__all__ = ['PointProcessor', 'BboxProcessor', 'object_processor']

class PointProcessor(Point):
    ''' Class to apply geometric transformations to a Point object '''

    def __init__(self, point: Point) -> None:
        '''
        Args:
            point (Point): Point instance
        '''

        assert isinstance(point, Point), 'point argument must be a Point instance'

        super(PointProcessor, self).__init__(point.x, point.y)
    
    def intersect(self, other: Union[Point, BoundingBox]) -> Point:
        ''' Compute intersection with another point (useless) or a BoundingBox
        
        Args:
            other (Point or BoundingBox): Point or BoundingBox instance
        
        Returns:
            Point:
                intersected Point object
        '''

        if isinstance(other, BoundingBox):
            if self.x >= other.x_max or self.x <= other.x_min \
                or self.y >= other.y_max or self.y <= other.y_min:

                pt = Point(self.x, self.y)
                pt.area = 0.
                return pt
            else:
                return Point(self.x, self.y)
                 
        else:
            return Point(self.x, self.y)   

    def shift(self, limits: BoundingBox) -> Point:
        ''' Shift point coordinates to limits frame coordinates

        The two objects must overlap in the global frame coordinates !

        Args:
            limits (BoundingBox): BoundingBox instance representing the limits
        
        Returns:
            Point:
                shifted Point object
        '''

        assert isinstance(limits, BoundingBox), \
            'limits argument must be a BoundingBox instance'

        assert self.x >= limits.x_min and self.y >= limits.y_min, \
            'Point not in the specified limits'

        adjusted_x = max(0, self.x - limits.x_min)
        adjusted_y = max(0, self.y - limits.y_min)

        return Point(adjusted_x, adjusted_y)
    
    def dist(self, other_point: Point) -> float:
        ''' Return the euclidean distance between the two points

        Args:
            other_point (Point): Point instance
        
        Returns:
            float:
                the euclidean distance between the two points
        '''

        assert isinstance(other_point, Point), \
            f'\'other_point\' must be a Point object, got {type(other_point)}'

        return ((self.x - other_point.x)**2 + (self.y - other_point.y)**2)**0.5


class BboxProcessor(BoundingBox):
    ''' Class to apply geometric transformations to a BoundingBox object '''

    def __init__(self, bbox: BoundingBox) -> None:
        '''
        Args:
            bbox (BoundingBox): BoundingBox instance
        '''

        assert isinstance(bbox, BoundingBox), \
            'bbox argument must be a BoundingBox instance'

        super(BboxProcessor,self).__init__(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)

    def intersect(self, other_bbox: BoundingBox) -> BoundingBox:
        ''' Compute intersection with another BoundingBox instance

        Args:
            other_bbox (BoundingBox): BoundingBox instance
        
        Returns:
            BoundingBox:
                intersected BoundingBox object
        '''

        assert isinstance(other_bbox, BoundingBox), \
            'other_bbox argument must be a BoundingBox instance'

        top_x = max(self.x_min,other_bbox.x_min)	
        top_y = max(self.y_min,other_bbox.y_min)	
        bot_x = min(self.x_max,other_bbox.x_max)	
        bot_y = min(self.y_max,other_bbox.y_max)

        if top_x >= bot_x or top_y >= bot_y:
            top_x, top_y, bot_x, bot_y = 0, 0, 0, 0

        return BoundingBox(top_x,top_y,bot_x,bot_y)
       
    def shift(self, limits: BoundingBox) -> BoundingBox:
        ''' Shift BoundingBox coordinates to limits frame coordinates

        The two objects must overlap in the global frame coordinates !

        Args:
            limits (BoundingBox): BoundingBox instance representing the limits
        
        Returns:
            BoundingBox:
                shifted BoundingBox object
        '''

        assert isinstance(limits, BoundingBox), \
            'limits argument must be a BoundingBox instance'

        assert self.intersect(limits).area > 0, \
            'The two objects must overlap in the global frame coordinates !'

        top_x = max(0, self.x_min - limits.x_min)
        top_y = max(0, self.y_min - limits.y_min)
        bot_x = max(0, self.x_max - limits.x_min)
        bot_y = max(0, self.y_max - limits.y_min)

        return BoundingBox(top_x,top_y,bot_x,bot_y)


def object_processor(obj: Union[Point, BoundingBox]) -> Union[PointProcessor, BboxProcessor]:
    '''
    Function that returns correct processor to use according to object type in input

    Args:
        obj (Point, BoundingBox): Point or BoundingBox instance
    
    Returns:
        PointProcessor or BboxProcessor:
            processor object
    '''
    
    if isinstance(obj, Point):
        return PointProcessor(obj)

    elif isinstance(obj, BoundingBox):
        return BboxProcessor(obj)
    
    else:
        raise Exception('Object must be a Point or a BoundingBox instance')
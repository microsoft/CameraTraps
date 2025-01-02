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

from typing import Union, Tuple

__all__ = ['Point', 'BoundingBox']

class Point:
    ''' Class to define a Point object in a 2D Cartesian 
    coordinate system.
    '''

    def __init__(self, x: Union[int,float], y: Union[int,float]) -> None:
        '''
        Args:
            x (int, float): x coordinate
            y (int, float): y coordinate
        '''

        assert x >= 0 and y >= 0, f'Coordinates must be positives, got x={x} and y={y}'

        self.x = x
        self.y = y
        self.area = 1 # always 1 pixel
    
    # @property
    # def area(self) -> int:
    #     ''' To get area '''
    #     return 1 # always 1 pixel
    
    @property
    def get_tuple(self) -> Tuple[Union[int,float],Union[int,float]]:
        ''' To get point's coordinates in tuple '''
        return (self.x,self.y)

    @property
    def atype(self) -> str:
        ''' To get annotation type string '''
        return 'Point'
    
    def __repr__(self) -> str:
        return f'Point(x: {self.x}, y: {self.y})'
    
    def __eq__(self, other) -> bool:
        return all([
            self.x == other.x,
            self.y == other.y
            ])

class BoundingBox:
    ''' Class to define a BoundingBox object in a 2D Cartesian 
    coordinate system.
    '''

    def __init__(
            self, 
            x_min: Union[int,float], 
            y_min: Union[int,float], 
            x_max: Union[int,float], 
            y_max: Union[int,float]
        ) -> None:
        '''
        Args:
            x_min (int, float): x bbox top-left coordinate
            y_min (int, float): y bbox top-left coordinate
            x_max (int, float): x bbox bottom-right coordinate
            y_max (int, float): y bbox bottom-right coordinate
        '''

        assert all([c >= 0 for c in [x_min,y_min,x_max,y_max]]), \
            f'Coordinates must be positives, got x_min={x_min}, y_min={y_min}, ' \
                f'x_max={x_max} and y_max={y_max}'

        assert x_max >= x_min and y_max >= y_min, \
            'Wrong bounding box coordinates.'

        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
    
    @property
    def area(self) -> Union[int,float]:
        ''' To get bbox area '''
        return max(0, self.width) * max(0, self.height)
    
    @property
    def width(self) -> Union[int,float]:
        ''' To get bbox width '''
        return max(0, self.x_max - self.x_min)
    
    @property
    def height(self) -> Union[int,float]:
        ''' To get bbox height '''
        return max(0, self.y_max - self.y_min)
    
    @property
    def get_tuple(self) -> Tuple[Union[int,float],...]:
        ''' To get bbox coordinates in tuple type '''
        return (self.x_min,self.y_min,self.x_max,self.y_max)
    
    @property
    def atype(self) -> str:
        ''' To get annotation type string '''
        return 'BoundingBox'

    def __repr__(self) -> str:
        return f'BoundingBox(x_min: {self.x_min}, y_min: {self.y_min}, x_max: {self.x_max}, y_max: {self.y_max})'

    def __eq__(self, other) -> bool:
        return all([
            self.x_min == other.x_min,
            self.y_min == other.y_min,
            self.x_max == other.x_max,
            self.y_max == other.y_max
            ])
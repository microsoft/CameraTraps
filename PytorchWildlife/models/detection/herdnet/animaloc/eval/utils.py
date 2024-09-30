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


import torch

from typing import List

from ..data import Point, BoundingBox, PointProcessor, BboxProcessor


def bboxes_iou(bboxes_a: List[BoundingBox], bboxes_b: List[BoundingBox]) -> List[List[float]]:
    ''' Return Intersect-over-Union (IoU) of 2 sets of boxes.

    Both sets of boxes are expected to be in (x_min, y_min, x_max, y_max)
    format.

    Args:
        bboxes_a (list): list of BoundingBox objects (N)
        bboxes_b (list): list of BoundingBox objects (M)
    
    Returns:
        List[List[float]]:
            pairwise Intersect-over-Union (IoU) values stored in a list of
            shape NxM.
    '''

    assert all(isinstance(a, BoundingBox) for a in bboxes_a) is True, \
        'bboxes_a must contains BoundingBox objects only'
    
    assert all(isinstance(b, BoundingBox) for b in bboxes_b) is True, \
        'bboxes_b must contains BoundingBox objects only'

    nxm_shape_list = []
    for bbox_a in bboxes_a:

        m_shape_list = []
        for bbox_b in bboxes_b:
            intersect = BboxProcessor(bbox_a).intersect(bbox_b).area
            union = bbox_a.area + bbox_b.area - intersect

            m_shape_list.append(intersect/union)
        
        nxm_shape_list.append(m_shape_list)
    
    return nxm_shape_list

def points_dist(points_a: List[Point], points_b: List[Point]) -> List[List[float]]:
    ''' Return euclidean distances of 2 sets of points.

    Both sets of points are expected to be in (x, y) format.

    Args:
        points_a (list): list of Point objects (N)
        points_b (list): list of Point objects (M)
    
    Returns:
        List[List[float]]:
            pairwise euclidean distances values stored in a list of
            shape NxM.
    '''

    assert all(isinstance(a, Point) for a in points_a) is True, \
        'points_a must contains Point objects only'
    
    assert all(isinstance(b, Point) for b in points_b) is True, \
        'points_b must contains Point objects only'

    nxm_shape_list = []
    for point_a in points_a:

        m_shape_list = []
        for point_b in points_b:          
            dist = PointProcessor(point_a).dist(point_b)

            m_shape_list.append(dist)
        
        nxm_shape_list.append(m_shape_list)
    
    return nxm_shape_list

def hann_window_2D(size: int) -> torch.Tensor:
    ''' 2D Hann window
    Args:
        size (int): size of the square window.
    
    Returns:
        torch.Tensor
            (1,H,W)
    '''    
    w1d = torch.hann_window(size)
    w2d = torch.outer(w1d, w1d).unsqueeze(0)
    return w2d

class HannWindow2D:

    def __init__(self, size: int) -> None:
        self.size = size
    
    def get_window(self, aspect: str = 'original', direction: str = 'up') -> torch.Tensor:

        assert aspect in ['original', 'corner', 'edge'], \
            'aspect argument must be either \'original\', \'corner\' or \'edge\', ' \
                f'got {aspect}'
        
        assert direction in ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right'], \
            'direction argument must be either \'up\', \'down\', \'left\', \'right\', ' \
            '\'up_left\', \'up_right\', \'down_left\' or \'down_right\', ' \
                f'got {direction}'
        
        if aspect == 'original':
            output = self._original_case(self.size)
        
        elif aspect == 'edge':
            output = self._edge_case(self.size)

            if direction == 'down':
                output = torch.rot90(output, k=2, dims=[1,2])

            elif direction == 'left':
                output = torch.rot90(output, k=1, dims=[1,2])

            elif direction == 'right':
                output = torch.rot90(output, k=-1, dims=[1,2])
        
        elif aspect == 'corner':
            output = self._corner_case(self.size)

            if direction == 'up_right':
                output = torch.rot90(output, k=-1, dims=[1,2])

            elif direction == 'down_left':
                output = torch.rot90(output, k=1, dims=[1,2])

            elif direction == 'down_right':
                output = torch.rot90(output, k=-2, dims=[1,2])
        
        return output
    
    def _original_case(self, size: int) -> torch.Tensor:
        return hann_window_2D(size)
    
    def _edge_case(self, size: int) -> torch.Tensor:
        zeros = torch.zeros((size // 2, size))
        w1d = torch.hann_window(size).repeat(size // 2, 1)
        output = torch.cat([w1d, zeros], dim=0).unsqueeze(0)
        output = torch.max(output, hann_window_2D(size))
        return output
    
    def _corner_case(self, size: int) -> torch.Tensor:
        zeros = torch.zeros((size // 2, size))
        ones = torch.zeros((size, size))
        ones[:size//2,:size//2] = 1.
        w1d = torch.hann_window(size).repeat(size // 2, 1)
        w1d = torch.cat([w1d, zeros], dim=0).unsqueeze(0)
        w2d = w1d.permute(0,2,1)
        output = torch.max(w1d, ones)
        output = torch.max(output, w2d)
        output = torch.max(output, hann_window_2D(size))
        return output
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


import os
import PIL

from PIL import ImageDraw
from typing import Dict, Tuple, Optional, Union

from ..data import BoundingBox, Point, Annotations

__all__ = ['draw_image_gt', 'draw_image_det', 'draw_image_gt_det']

def _text_box(
    draw: ImageDraw, 
    xy: Tuple[int,int], 
    label: Union[int,str], 
    text_color: Tuple[int,int,int],
    text_bg_color: Tuple[int,int,int],
    score: Optional[float] = None
    ) -> None:
    '''
    Draw a white box with label name (and score)
    on top of the bounding box.
    '''

    text = f' {label} '
    if score is not None:
        text = f' {label} | {score:.2f} '

    text_size = draw.textsize(text)

    x , y = xy
    xy_offset = (x,y - text_size[1])
    col_box = [x, y - text_size[1], x + text_size[0], y]

    draw.rectangle(col_box, fill=text_bg_color)
    draw.text(xy_offset, text = text, fill=text_color)

def _draw_image(
    draw: ImageDraw,
    annos: Annotations,
    score: bool = False,
    labels_names: Optional[Dict[int,str]] = None,
    color: Tuple[int,int,int] = (0,255,0),
    thick: int = 2,
    text_color: Tuple[int,int,int] = (0,0,0),
    text_bg_color: Tuple[int,int,int] = (0,255,0)
    ) -> None:
    ''' Draw annotations on image '''

    for anno_record in annos:

        anno , label = anno_record['annos'] , anno_record['labels']
        score_value = None
        if score: score_value = anno_record['scores']

        if labels_names is not None:
            label = labels_names[label-1]

        if isinstance(anno, Point):
            draw.point(anno.get_tuple, fill=color)
            _text_box(draw, anno.get_tuple, label, color=color)
        
        elif isinstance(anno, BoundingBox):
            draw.rectangle(anno.get_tuple, outline=color, width=thick)
            _text_box(draw = draw, xy = tuple([anno.x_min,anno.y_min]), label = label, 
                text_color = text_color, text_bg_color=text_bg_color, score=score_value)

def draw_image_gt(
    image: PIL.Image.Image,
    groundtruth: Annotations,
    save_dir: str,
    labels_names: Optional[Dict[int,str]] = None,
    gt_color: Tuple[int,int,int] = (0,255,0),
    gt_thick: int = 2,
    text_color: Tuple[int,int,int] = (0,0,0),
    text_bg_color: Tuple[int,int,int] = (0,255,0),
    quality: int = 95,
    show: bool = False
    ) -> None:
    ''' Draw ground truth on PIL image

    Args:
        image (PIL.Image.Image): PIL image to draw ground truth on, must contain a
            filename attribute that contains image name with extension
        groundtruth (Annotations): an Annotations instances containing the ground truth
        save_dir (str): directory path where image will be saved
        labels_names (dict, optional): a dict containing correspondence between labels 
            id (int) and labels names (str). If specified, theses will be written instead
            of labels id. Defaults to None
        gt_color (tuple, optional): annotations color. Defaults to (0,255,0) (lime)
        gt_thick (int, optional): thickness of annotation drawn, in pixels. 
            Defaults to 2
        text_color (tuple, optional): text color. Defaults to (0,0,0) (black)
        text_bg_color (tuple, optional): text background color. Defaults to 
            (0,255,0) (lime)
        quality (int, optional): the saved image quality, on a scale from 1 (worst) to 
            95 (best). Defaults to 95
        show (bool, optional): set to True to show the image. Defaults to False
    '''

    assert isinstance(image, PIL.Image.Image), \
        'image argument must contain a PIL.Image.Image instance'
    assert isinstance(groundtruth, Annotations), \
        'groundtruth argument must contain an Annotations instance'

    img_name = os.path.basename(image.filename)
    groundtruth = groundtruth.sub(img_name)

    with image as img:
        draw = ImageDraw.Draw(img)

        _draw_image(draw, groundtruth, False, labels_names, gt_color, gt_thick, 
            text_color, text_bg_color)

        if show: img.show(title=img_name)
        img.save(os.path.join(save_dir,img_name), "JPEG", quality=quality)

def draw_image_det(
    image: PIL.Image.Image,
    detections: Annotations,
    save_dir: str,
    labels_names: Optional[Dict[int,str]] = None,
    det_color: Tuple[int,int,int] = (255,0,0),
    det_thick: int = 2,
    text_color: Tuple[int,int,int] = (0,0,0),
    text_bg_color: Tuple[int,int,int] = (255,0,0),
    quality: int = 95,
    show: bool = False
    ) -> None:
    ''' Draw detections on PIL image

    Args:
        image (PIL.Image.Image): PIL image to draw detections on, must contain a
            filename attribute that contains image name with extension
        detections (Annotations): an Annotations instances containing the detections, must
            contains a 'scores' attribute
        save_dir (str): directory path where image will be saved
        labels_names (dict, optional): a dict containing correspondence between labels 
            id (int) and labels names (str). If specified, theses will be written instead
            of labels id. Defaults to None
        det_color (tuple, optional): detections color. Defaults to (255,0,0) (red)
        det_thick (int, optional): thickness of detection drawn, in pixels. 
            Defaults to 2
        text_color (tuple, optional): text color. Defaults to (0,0,0) (black)
        text_bg_color (tuple, optional): text background color. Defaults to 
            (255,0,0) (red)
        quality (int, optional): the saved image quality, on a scale from 1 (worst) to 
            95 (best). Defaults to 95
        show (bool, optional): set to True to show the image. Defaults to False
    '''

    assert isinstance(image, PIL.Image.Image), \
        'image argument must contain a PIL.Image.Image instance'
    assert isinstance(detections, Annotations), \
        'detections argument must contain an Annotations instance'
    assert hasattr(detections, 'scores'), \
        'Annotations instance must contain a \'scores\' attribute'

    img_name = os.path.basename(image.filename)
    detections = detections.sub(img_name)

    with image as img:
        draw = ImageDraw.Draw(img)

        _draw_image(draw, detections, True, labels_names, det_color, det_thick, 
            text_color, text_bg_color)

        if show: img.show(title=img_name)
        img.save(os.path.join(save_dir,img_name), "JPEG", quality=quality)

def draw_image_gt_det(
    image: PIL.Image.Image,
    groundtruth: Annotations,
    detections: Annotations,
    save_dir: str,
    labels_names: Optional[Dict[int,str]] = None,
    gt_color: Tuple[int,int,int] = (0,255,0),
    det_color: Tuple[int,int,int] = (255,0,0),
    thick: int = 2,
    text_color: Tuple[int,int,int] = (0,0,0),
    text_gt_bg_color: Tuple[int,int,int] = (0,255,0),
    text_det_bg_color: Tuple[int,int,int] = (255,0,0),
    quality: int = 95,
    show: bool = False
    ) -> None:
    ''' Draw ground truth and detections on PIL image

    Args:
        image (PIL.Image.Image): PIL image to draw ground truth on, must contain a
            filename attribute that contains image name with extension
        groundtruth (Annotations): an Annotations instances containing the ground truth
        detections (Annotations): an Annotations instances containing the detections, must
            contains a 'scores' attribute
        save_dir (str): directory path where image will be saved
        labels_names (dict, optional): a dict containing correspondence between labels 
            id (int) and labels names (str). If specified, theses will be written instead
            of labels id. Defaults to None
        gt_color (tuple, optional): annotations color. Defaults to (0,255,0) (lime)
        det_color (tuple, optional): detections color. Defaults to (255,0,0) (red)
        thick (int, optional): thickness of annotation drawn, in pixels. 
            Defaults to 2
        text_color (tuple, optional): text color. Defaults to (0,0,0) (black)
        text_gt_bg_color (tuple, optional): ground truth's text background color. Defaults to 
            (0,255,0) (lime)
        text_det_bg_color (tuple, optional): detections' text background color. Defaults to 
            (255,0,0) (red)
        quality (int, optional): the saved image quality, on a scale from 1 (worst) to 
            95 (best). Defaults to 95
        show (bool, optional): set to True to show the image. Defaults to False
    '''

    assert isinstance(image, PIL.Image.Image), \
        'image argument must contain a PIL.Image.Image instance'
    assert isinstance(groundtruth, Annotations), \
        'groundtruth argument must contain an Annotations instance'
    assert isinstance(detections, Annotations), \
        'detections argument must contain an Annotations instance'
    assert hasattr(detections, 'scores'), \
        'detections Annotations instance must contain a \'scores\' attribute'

    img_name = os.path.basename(image.filename)
    groundtruth = groundtruth.sub(img_name)
    detections = detections.sub(img_name)

    with image as img:
        draw = ImageDraw.Draw(img)

        _draw_image(draw, groundtruth, False, labels_names, gt_color, thick, 
            text_color, text_gt_bg_color)
        
        _draw_image(draw, detections, True, labels_names, det_color, thick, 
            text_color, text_det_bg_color)
        
        if show: img.show(title=img_name)
        img.save(os.path.join(save_dir,img_name), "JPEG", quality=quality)
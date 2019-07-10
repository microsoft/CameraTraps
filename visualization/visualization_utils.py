#####
#
# visualization_utils.py
#
# Core rendering functions shared across visualization scripts
#
#####

#%% Constants and imports

import numpy as np
from PIL import Image, ImageFile, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from data_management.annotations import annotation_constants
ImageFile.LOAD_TRUNCATED_IMAGES = True


#%% Functions

def open_image(input):
    """
    Opens an image in binary format using PIL.Image and convert to RGB mode. This operation is lazy; image will
    not be actually loaded until the first operation that needs to load it (for example, resizing), so file opening
    errors can show up later.

    Args:
        input: an image in binary format read from the POST request's body or
            path to an image file (anything that PIL can open)

    Returns:
        an PIL image object in RGB mode
    """

    image = Image.open(input)
    if image.mode not in ('RGBA', 'RGB'):
        raise AttributeError('Input image not in RGBA or RGB mode and cannot be processed.')
    if image.mode == 'RGBA':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')
    return image


def resize_image(image, target_width, target_height=-1):
    """
    Resizes a PIL image object to the specified width and height; does not resize
    in place.  If either width or height are -1, resizes with aspect ratio preservation.
    If both are -1, returns the original image (does not copy in this case).
    """

    # Null operation
    if target_width == -1 and target_height == -1:

        return image

    elif target_width == -1 or target_height == -1:

        # Aspect ratio as width over height
        aspect_ratio = image.size[0] / image.size[1]

        if target_width != -1:
            # ar = w / h
            # h = w / ar
            target_height = int(target_width / aspect_ratio)

        else:
            # ar = w / h
            # w = ar * h
            target_width = int(aspect_ratio * target_height)

    resized_image = image.resize((target_width, target_height), Image.ANTIALIAS)
    return resized_image


def render_iMerit_boxes(boxes, classes, image,
                        label_map=annotation_constants.bbox_category_id_to_name):
    """
    Renders bounding boxes and their category labels on a PIL image.

    Args:
        boxes: bounding box annotations from iMerit, format is [x_rel, y_rel, w_rel, h_rel] (rel = relative coords)
        image: PIL.Image object to annotate on
        label_map: optional dict mapping classes to a string for display

    Returns:
        image will be altered in place
    """

    display_boxes = []
    display_strs = []  # list of list, one list of strings for each bounding box (to accommodate multiple labels)
    for box, clss in zip(boxes, classes):
        x_rel, y_rel, w_rel, h_rel = box
        ymin, xmin = y_rel, x_rel
        ymax = ymin + h_rel
        xmax = xmin + w_rel

        display_boxes.append([ymin, xmin, ymax, xmax])

        if label_map:
            clss = label_map[int(clss)]
        display_strs.append([clss])

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs)


def render_db_bounding_boxes(boxes, classes, image, original_size=None,
                             label_map=None, thickness=4):
    """
    Render bounding boxes (with class labels) on [image].  This is a wrapper for
    draw_bounding_boxes_on_image, allowing the caller to operate on a resized image
    by providing the original size of the image; bboxes will be scaled accordingly.
    """

    display_boxes = []
    display_strs = []

    if original_size is not None:
        image_size = original_size
    else:
        image_size = image.size

    img_width, img_height = image_size

    for box, clss in zip(boxes, classes):

        x_min_abs, y_min_abs, width_abs, height_abs = box

        ymin = y_min_abs / img_height
        ymax = ymin + height_abs / img_height

        xmin = x_min_abs / img_width
        xmax = xmin + width_abs / img_width

        display_boxes.append([ymin, xmin, ymax, xmax])

        if label_map:
            clss = label_map[int(clss)]
        display_strs.append([clss])

    display_boxes = np.array(display_boxes)

    draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs,
                                 thickness=thickness)



def render_detection_bounding_boxes(detections, image,
                                    label_map={},
                                    classification_label_map={},
                                    confidence_threshold=0.8, thickness=4):
    """
    Renders bounding boxes, label and confidence on an image if confidence is above the threshold.
    This is works with the output of the detector batch processing API.
    Supports classification, if the detection contains classification results according to the detector
    API output version 1.0. The detection label is ignored then, if the detection category is '1'.

    Args:
        detections: detections on the image, example content:
            {
                [
                    "category": "2",
                    "conf": 0.996,
                    "bbox": [
                        0.0,
                        0.2762,
                        0.1234,
                        0.2458
                    ]
                ]
            }
            where the bbox coordinates are [x, y, width_box, height_box], (x, y) is upper left.
            Supports classification results, if *detections* have the format
            {
                [
                    "category": "2",
                    "conf": 0.996,
                    "bbox": [
                        0.0,
                        0.2762,
                        0.1234,
                        0.2458
                    ]
                    "classifications": [
                        ["3", 0.901],
                        ["1", 0.071],
                        ["4", 0.025]
                    ]
                ]
            }
        image: PIL.Image object, output of generate_detections.
        label_map: optional, mapping the numerical label to a string name. The type of the numerical label
            (default string) needs to be consistent with the keys in label_map; no casting is carried out.
        classification_label_map: optional, mapping of the string class labels to the actual class names.
            The type of the numerical label (default string) needs to be consistent with the keys in
            label_map; no casting is carried out.
        confidence_threshold: optional, threshold above which the bounding box is rendered.
        thickness: optional, rendering line thickness.

    image is modified in place.
    """

    display_boxes = []
    display_strs = []  # list of lists, one list of strings for each bounding box (to accommodate multiple labels)
    classes = []

    for detection in detections:

        score = detection['conf']
        if score > confidence_threshold:
            x1, y1, w_box, h_box = detection['bbox']
            display_boxes.append([y1, x1, y1 + h_box, x1 + w_box])
            clss = detection['category']
            label = label_map[clss] if clss in label_map else clss
            displayed_label = ['Detected as {}: {}%'.format(label, round(100 * score))]
            if clss == '1' and 'classifications' in detection:
                # To avoid duplicate colors with detection-only visualization offset
                # the classification class index by the number of detection classes
                clss = len(annotation_constants.bbox_categories) + int(detection['classifications'][0][0])
                for classification in detection['classifications']:
                    class_key = classification[0]
                    if class_key in classification_label_map:
                        class_name = classification_label_map[class_key]
                    else:
                        class_name = class_key
                    displayed_label += ['{} with confidence {:5.1%}'.format(class_name, classification[1])]
            display_strs.append(displayed_label)
            classes.append(clss)

    display_boxes = np.array(display_boxes)

    draw_bounding_boxes_on_image(image, display_boxes, classes,
                                 display_strs=display_strs, thickness=thickness)


# The following functions are modified versions of those at:
#
# https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

COLORS = [
    'AliceBlue', 'Red', 'RoyalBlue', 'Gold', 'Chartreuse', 'Aqua',  'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'RosyBrown', 'Aquamarine', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 classes,
                                 thickness=4,
                                 display_strs=()):
    """
    Draws bounding boxes on image.

    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      classes: a list of ints or strings (which can be cast to ints) corresponding to the class labels of the boxes.
             This is only used for selecting the color to render the bounding box in.
      thickness: line thickness. Default value is 4.
      display_strs: list of list of strings.
                             a list of strings for each bounding box.
                             The reason to pass a list of strings for a
                             bounding box is that it might contain
                             multiple labels.
    """

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        # print('Input must be of size [N, 4], but is ' + str(boxes_shape))
        return  # no object detection on this image, return
    for i in range(boxes_shape[0]):
        if display_strs:
            display_str_list = display_strs[i]
            draw_bounding_box_on_image(image,
                                       boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3],
                                       classes[i],
                                       thickness=thickness, display_str_list=display_str_list)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               clss,
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """
    Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      clss: the class of the object in this bounding box - will be cast to an int.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """

    color = COLORS[int(clss) % len(COLORS)]

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

    try:
        font = ImageFont.truetype('arial.ttf', 32)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)

        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def plot_confusion_matrix(matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                         vmax=None,
                         use_colorbar=True,
                         y_label = True):

    """
    This function plots a confusion matrix.
    Normalization can be applied by setting `normalize=True`.


    """

    if normalize:
        matrix = matrix.astype(np.double) / (matrix.sum(axis=1, keepdims=True) + 1e-7)

    plt.imshow(matrix, interpolation='nearest', cmap=cmap, vmax=vmax)
    plt.title(title) #,fontsize=22)
    if use_colorbar:
        plt.colorbar(fraction=0.046, pad=0.04, ticks=[0.0,0.25,0.5,0.75,1.0]).set_ticklabels(['0%','25%','50%','75%','100%'])
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    if y_label:
        plt.yticks(tick_marks, classes)
    else:
        plt.yticks(tick_marks, ['' for cn in classes])

    thresh = 0.5
    for i, j in np.ndindex(matrix.shape):
        plt.text(j, i, '{:.0f}%'.format(matrix[i, j]*100),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black",
                fontsize='x-small')
    if y_label:
        plt.ylabel('Ground-truth class')
    plt.xlabel('Predicted class')
    #plt.grid(False)
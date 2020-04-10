"""
visualization_utils.py

Core rendering functions shared across visualization scripts
"""

#%% Constants and imports

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import requests
from PIL import Image, ImageFile, ImageFont, ImageDraw
from io import BytesIO
from data_management.annotations import annotation_constants

ImageFile.LOAD_TRUNCATED_IMAGES = True


#%% Functions

def open_image(input_file):
    """
    Opens an image in binary format using PIL.Image and convert to RGB mode. This operation is lazy; image will
    not be actually loaded until the first operation that needs to load it (for example, resizing), so file opening
    errors can show up later.

    Args:
        input_file: an image in binary format read from the POST request's body or
            path to an image file (anything that PIL can open)

    Returns:
        an PIL image object in RGB mode
    """

    if input_file.startswith('http://') or input_file.startswith('https://'):
        response = requests.get(input_file)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L'):
        raise AttributeError('Input image {} uses unsupported mode {}'.format(input_file, image.mode))
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')
    return image


def load_image(input_file):
    """
    Loads the image at input_file as a PIL Image into memory; Image.open() used in open_image() is lazy and
    errors will occur downstream if not explicitly loaded

    Args:
        input_file: an image in binary format read from the POST request's body or
            path to an image file (anything that PIL can open)

    Returns:
        an PIL image object in RGB mode
    """
    image = open_image(input_file)
    image.load()
    return image


def resize_image(image, target_width, target_height=-1):
    """
    Resizes a PIL image object to the specified width and height; does not resize
    in place. If either width or height are -1, resizes with aspect ratio preservation.
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


def show_images_in_a_row(images):
    num = len(images)
    assert num > 0

    if isinstance(images[0], str):
        images = [Image.open(img) for img in images]

    fig, axarr = plt.subplots(1, num, squeeze=False)  # number of rows, number of columns
    fig.set_size_inches((num * 5, 25))  # each image is 2 inches wide
    for i, img in enumerate(images):
        axarr[0, i].set_axis_off()
        axarr[0, i].imshow(img)
    return fig


# The following three functions are modified versions of those at:
# https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

COLORS = [
        'AliceBlue', 'Red', 'RoyalBlue', 'Gold', 'Chartreuse', 'Aqua', 'Azure',
        'Beige', 'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
        'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson',
        'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
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


def render_detection_bounding_boxes(detections, image,
                                    label_map={},
                                    classification_label_map={},
                                    confidence_threshold=0.8, thickness=4, expansion=0,
                                    classification_confidence_threshold=0.3,
                                    max_classifications=3):
    """
    Renders bounding boxes, label, and confidence on an image if confidence is above the threshold.

    This works with the output of the batch processing API.

    Supports classification, if the detection contains classification results according to the
    API output version 1.0.

    Args:

        detections: detections on the image, example content:
            [
                {
                    "category": "2",
                    "conf": 0.996,
                    "bbox": [
                        0.0,
                        0.2762,
                        0.1234,
                        0.2458
                    ]
                }
            ]

            ...where the bbox coordinates are [x, y, box_width, box_height].

            (0, 0) is the upper-left.  Coordinates are normalized.

            Supports classification results, if *detections* have the format
            [
                {
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
                }
            ]

        image: PIL.Image object, output of generate_detections.

        label_map: optional, mapping the numerical label to a string name. The type of the numerical label
            (default string) needs to be consistent with the keys in label_map; no casting is carried out.

        classification_label_map: optional, mapping of the string class labels to the actual class names.
            The type of the numerical label (default string) needs to be consistent with the keys in
            label_map; no casting is carried out.

        confidence_threshold: optional, threshold above which the bounding box is rendered.
        thickness: line thickness in pixels. Default value is 4.
        expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
        classification_confidence_threshold: confidence above which classification result is retained.
        max_classifications: maximum number of classification results retained for one image.

    image is modified in place.
    """

    display_boxes = []
    display_strs = []  # list of lists, one list of strings for each bounding box (to accommodate multiple labels)
    classes = []  # for color selection

    for detection in detections:

        score = detection['conf']
        if score >= confidence_threshold:

            x1, y1, w_box, h_box = detection['bbox']
            display_boxes.append([y1, x1, y1 + h_box, x1 + w_box])
            clss = detection['category']
            label = label_map[clss] if clss in label_map else clss
            displayed_label = ['{}: {}%'.format(label, round(100 * score))]

            if 'classifications' in detection:

                # To avoid duplicate colors with detection-only visualization, offset
                # the classification class index by the number of detection classes
                clss = annotation_constants.NUM_DETECTOR_CATEGORIES + int(detection['classifications'][0][0])
                classifications = detection['classifications']
                if len(classifications) > max_classifications:
                    classifications = classifications[0:max_classifications]
                for classification in classifications:
                    p = classification[1]
                    if p < classification_confidence_threshold:
                        continue
                    class_key = classification[0]
                    if class_key in classification_label_map:
                        class_name = classification_label_map[class_key]
                    else:
                        class_name = class_key
                    displayed_label += ['{}: {:5.1%}'.format(class_name.lower(), classification[1])]

            # ...if we have detection results
            display_strs.append(displayed_label)
            classes.append(clss)

        # ...if the confidence of this detection is above threshold

    # ...for each detection
    display_boxes = np.array(display_boxes)

    draw_bounding_boxes_on_image(image, display_boxes, classes,
                                 display_strs=display_strs, thickness=thickness, expansion=expansion)


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 classes,
                                 thickness=4,
                                 expansion=0,
                                 display_strs=()):
    """
    Draws bounding boxes on image.

    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      classes: a list of ints or strings (that can be cast to ints) corresponding to the class labels of the boxes.
             This is only used for selecting the color to render the bounding box in.
      thickness: line thickness in pixels. Default value is 4.
      expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
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
                                       thickness=thickness, expansion=expansion,
                                       display_str_list=display_str_list)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               clss=None,
                               thickness=4,
                               expansion=0,
                               display_str_list=(),
                               use_normalized_coordinates=True,
                               label_font_size=16):
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
    ymin: ymin of bounding box - upper left.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    clss: str, the class of the object in this bounding box - will be cast to an int.
    thickness: line thickness. Default value is 4.
    expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
    display_str_list: list of strings to display in box
        (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    label_font_size: font size to attempt to load arial.ttf with
    """
    if clss is None:
        color = COLORS[1]
    else:
        color = COLORS[int(clss) % len(COLORS)]

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    if expansion > 0:
        left -= expansion
        right += expansion
        top -= expansion
        bottom += expansion

    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

    try:
        font = ImageFont.truetype('arial.ttf', label_font_size)
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

        text_bottom -= (text_height + 2 * margin)


def render_iMerit_boxes(boxes, classes, image,
                        label_map=annotation_constants.bbox_category_id_to_name):
    """
    Renders bounding boxes and their category labels on a PIL image.

    Args:
        boxes: bounding box annotations from iMerit, format is [x_rel, y_rel, w_rel, h_rel] (rel = relative coords)
        classes: the class IDs of the predicted class of each box/object
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


def render_megadb_bounding_boxes(boxes_info, image):
    """
    each item in boxes_info is
    {
        "category": "animal",
        "bbox": [
          0.739,
          0.448,
          0.187,
          0.198
        ]
    }
    """
    display_boxes = []
    display_strs = []
    classes = []  # ints, for selecting colors

    for b in boxes_info:
        x_rel, y_rel, w_rel, h_rel = b['bbox']
        ymin, xmin = y_rel, x_rel
        ymax = ymin + h_rel
        xmax = xmin + w_rel
        display_boxes.append([ymin, xmin, ymax, xmax])
        display_strs.append([b['category']])
        classes.append(annotation_constants.bbox_category_name_to_id[b['category']])

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs)


def render_db_bounding_boxes(boxes, classes, image, original_size=None,
                             label_map=None, thickness=4, expansion=0):
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
        display_strs.append([str(clss)])  # need to be a string here because PIL needs to iterate through chars

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs,
                                 thickness=thickness, expansion=expansion)
    
    
def plot_confusion_matrix(matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          vmax=None,
                          use_colorbar=True,
                          y_label=True):

    """
    This function plots a confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        matrix: confusion matrix as a numpy 2D matrix. Rows are ground-truth classes
            and columns the predicted classes. Number of rows and columns have to match
        classes: list of strings, which contain the corresponding class names for each row/column
        normalize: boolean indicating whether to perform row-wise normalization to sum 1
        title: string which will be used as title
        cmap: pyplot colormap, default: matplotlib.pyplot.cm.Blues
        vmax: float, specifies the value that corresponds to the largest value of the colormap.
            If None, the maximum value in *matrix* will be used. Default: None
        use_colorbar: boolean indicating if a colorbar should be plotted
        y_label: boolean indicating whether class names should be plotted on the y-axis as well

    Returns a reference to the figure
    """

    assert matrix.shape[0] == matrix.shape[1]
    fig = plt.figure(figsize=[3 + 0.5 * len(classes)] * 2)

    if normalize:
        matrix = matrix.astype(np.double) / (matrix.sum(axis=1, keepdims=True) + 1e-7)

    plt.imshow(matrix, interpolation='nearest', cmap=cmap, vmax=vmax)
    plt.title(title)   # ,fontsize=22)

    if use_colorbar:
        plt.colorbar(fraction=0.046, pad=0.04,
                     ticks=[0.0,0.25,0.5,0.75,1.0]).set_ticklabels(['0%','25%','50%','75%','100%'])

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)

    if y_label:
        plt.yticks(tick_marks, classes)
    else:
        plt.yticks(tick_marks, ['' for cn in classes])

    for i, j in np.ndindex(matrix.shape):

        plt.text(j, i, '{:.0f}%'.format(matrix[i, j]*100),
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='white' if matrix[i, j] > 0.5 else 'black',
                fontsize='x-small')

    if y_label:
        plt.ylabel('Ground-truth class')

    plt.xlabel('Predicted class')
    #plt.grid(False)
    plt.tight_layout()

    return fig


def plot_precision_recall_curve(precisions, recalls, title='Precision/Recall curve'):

    """
    Plots the precision recall curve given lists of (ordered) precision
    and recall values
    Args:
        precisions: list of floats, the precision for the corresponding recall values.
            Should have same length as *recalls*.
        recalls: list of floats, the recall values for corresponding precision values.
            Should have same length as *precisions*.
        title: string that will be as as plot title

    Returns a reference to the figure
    """

    step_kwargs = ({'step': 'post'})
    fig = plt.figure()
    plt.title(title)
    plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
    plt.fill_between(recalls, precisions, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])

    return fig


def plot_stacked_bar_chart(data, series_labels, col_labels=None, x_label=None, y_label=None, log_scale=False):
    """
    For plotting e.g. species distribution across locations.
    Reference: https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
    Args:
        data: a 2-dimensional numpy array or nested list containing data for each series (species)
              in rows (1st dimension) across locations (columns, 2nd dimension)

    Returns:
        the plot that can then be saved as a png.
    """

    fig = plt.figure()
    ax = plt.subplot(111)

    data = np.array(data)
    num_series, num_columns = data.shape
    ind = list(range(num_columns))

    colors = cm.rainbow(np.linspace(0, 1, num_series))

    cumulative_size = np.zeros(num_columns)  # stacked bar charts are made with each segment starting from a y position

    for i, row_data in enumerate(data):
        ax.bar(ind, row_data, bottom=cumulative_size, label=series_labels[i], color=colors[i])
        cumulative_size += row_data

    if col_labels and len(col_labels) < 25:
        ax.set_xticks(ind)
        ax.set_xticklabels(col_labels, rotation=90)
    elif col_labels:
        ax.set_xticks(list(range(0, len(col_labels), 20)))
        ax.set_xticklabels(col_labels, rotation=90)

    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if log_scale:
        ax.set_yscale('log')

    # To fit the legend in, shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(0.99, 0.5), frameon=False)

    return fig


#####
#
# visualization_utils.py
#
# Core rendering functions shared across visualization scripts
#
#####

#%% Constants and imports

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageFile
from detection.run_tf_detector import DetectorUtils

from data_management.annotations import annotation_constants

ImageFile.LOAD_TRUNCATED_IMAGES = True


#%% Classes


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
    if image.mode not in ('RGBA', 'RGB', 'L'):
        raise AttributeError('Input image {} uses unsupported mode {}'.format(input,image.mode))
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')
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
    DetectorUtils.draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs)


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
    DetectorUtils.draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs)


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
    DetectorUtils.draw_bounding_boxes_on_image(image, display_boxes, classes, display_strs=display_strs,
                                 thickness=thickness,expansion=expansion)


def render_detection_bounding_boxes(detections, image,
                                        label_map={},
                                        classification_label_map={},
                                        confidence_threshold=0.8, thickness=4, expansion=0,
                                        classification_confidence_threshold=0.3,
                                        max_classifications=3):
    """
    Renders bounding boxes, label, and confidence on an image if confidence is above the threshold.
    
    Simple wrapper around the corresponding DetectorUtils function.
    """
    DetectorUtils.render_detection_bounding_boxes(detections,image,label_map,classification_label_map,
                                                  confidence_threshold,thickness,expansion,
                                                  classification_confidence_threshold,
                                                  max_classifications)
    
    
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
    plt.title(title) #,fontsize=22)

    if use_colorbar:
        plt.colorbar(fraction=0.046, pad=0.04, ticks=[0.0,0.25,0.5,0.75,1.0]).set_ticklabels(['0%','25%','50%','75%','100%'])

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
    plt.step(recalls, precisions, color='b', alpha=0.2,
                where='post')
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


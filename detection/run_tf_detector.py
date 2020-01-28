"""
Module to run a TensorFlow animal detection model on images.

Contains functions to load a TensorFlow detection model, run inference,
render bounding boxes on images, and write out the resulting
images (with bounding boxes).

THIS SCRIPT IS NOT A GOOD WAY TO PROCESS LOTS OF IMAGES.

Did we mention in all caps that this script is not a good way to process lots
of images? It does not facilitate checkpointing the results so if it crashes
you would have to start from scratch. If you want to run a detector (e.g. ours)
on lots of images, you should check out:

1) run_tf_detector_batch.py (for local execution)

2) https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing
   (for running large jobs on Azure ML)

The good news: this script depends on nothing else in our repo, just standard pip
installs (the list of packages you need to install via pip is listed in `envrionment-detector.yml`
at the root of the repo); alternatively, we recommend you set up a conda virtual environment
following the Installation section on the main README, using `envrionment-detector.yml` as the
environment file where asked).

It's a good way to test our detector on a handful of images and
get super-satisfying, graphical results.  It's also a good way to see how
fast a detector model will run on a particular machine.

See the "command-line driver" cell for example invocations.

If you would like to not use the GPU on the machine, set the environment variable CUDA_VISIBLE_DEVICES to "-1"

If no output directory is specified, writes detections for c:\foo\bar.jpg to
c:\foo\bar_detections.jpg .

This script will only consider detections with > 0.3 confidence at all times. The `threshold` you
provide is only for rendering the results. If you need to see lower-confidence detections, you can change
DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD.

Reference:
https://github.com/tensorflow/models/blob/master/research/object_detection/inference/detection_inference.py
"""


#%% Constants, imports, environment

import argparse
import glob
import math
import os
import statistics
import sys
import time
import warnings

import humanfriendly
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from tqdm import tqdm

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)

# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings('ignore', 'Metadata warning', UserWarning)

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf

print('TensorFlow version:', tf.__version__)
print('Is GPU available? tf.test.is_gpu_available:', tf.test.is_gpu_available())


#%% Classes

class ImagePathUtils:
    """A collection of utility functions supporting this stand-alone script"""

    # Stick this into filenames before the extension for the rendered result
    DETECTION_FILENAME_INSERT = '_detections'

    image_extensions = ['.jpg', '.jpeg', '.gif', '.png']

    @staticmethod
    def is_image_file(s):
        """
        Check a file's extension against a hard-coded set of image file extensions    '
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in ImagePathUtils.image_extensions

    @staticmethod
    def find_image_files(strings):
        """
        Given a list of strings that are potentially image file names, look for strings
        that actually look like image file names (based on extension).
        """
        return [s for s in strings if ImagePathUtils.is_image_file(s)]

    @staticmethod
    def find_images(dir_name, recursive=False):
        """
        Find all files in a directory that look like image file names
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        image_strings = ImagePathUtils.find_image_files(strings)

        return image_strings


class DetectorUtils:
    """
    A collection of utility functions to support TFDetector.
    The functions themselves live in ct_utils.py and visualization/visualization_utils.py
    and should be imported from there instead - this script needs to be stand-alone.
    """

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

    @staticmethod
    def truncate_float(x, precision=3):
        """
        Function for truncating a float scalar to the defined precision.
        For example: truncate_float(0.0003214884) --> 0.000321

        Args:
        x         (float) Scalar to truncate
        precision (int)   The number of significant digits to preserve, should be
                          greater or equal 1
        """

        assert precision > 0

        if np.isclose(x, 0):
            return 0
        else:
            # Determine the factor, which shifts the decimal point of x
            # just behind the last significant digit
            factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
            # Shift decimal point by multiplicatipon with factor, flooring, and
            # division by factor
            return math.floor(x * factor) / factor

    @staticmethod
    def round_and_make_float(d, precision=4):
        return DetectorUtils.truncate_float(float(d), precision=precision)

    @staticmethod
    def __open_image(input_file):
        """Opens an image in binary format using PIL.Image and convert to RGB mode.

        Args:
            input_file: an image in binary format read from the POST request's body or
                path to an image file (anything that PIL can open)

        Returns:
            an PIL image object in RGB mode
        """
        image = Image.open(input_file)
        if image.mode not in ('RGBA', 'RGB'):
            raise AttributeError('Input image not in RGBA or RGB mode and cannot be processed.')
        if image.mode == 'RGBA':
            # PIL.Image.convert() returns a converted copy of this image
            image = image.convert(mode='RGB')
        return image

    @staticmethod
    def load_image(input_file):
        """Loads the image at input_file as a PIL Image into memory; Image.open() is lazy and errors will
        occur downstream if not explicitly loaded

        Args:
            input_file: an image in binary format read from the POST request's body or
                path to an image file (anything that PIL can open)

        Returns:
            an PIL image object in RGB mode
        """
        image = DetectorUtils.__open_image(input_file)
        image.load()
        return image

    # The following functions are modified versions of those at:
    # https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

    @staticmethod
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
                    clss = TFDetector.NUM_DETECTOR_CATEGORIES + int(detection['classifications'][0][0])
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

        DetectorUtils.draw_bounding_boxes_on_image(image, display_boxes, classes,
                                                   display_strs=display_strs, thickness=thickness, expansion=expansion)

    @staticmethod
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
                DetectorUtils.draw_bounding_box_on_image(image,
                                                         boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3],
                                                         classes[i],
                                                         thickness=thickness, expansion=expansion,
                                                         display_str_list=display_str_list)

    @staticmethod
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
            color = DetectorUtils.COLORS[1]
        else:
            color = DetectorUtils.COLORS[int(clss) % len(DetectorUtils.COLORS)]

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


class TFDetector:
    """
    A detector model loaded at the time of initialization. It is intended to be used with
    the MegaDetector (TF). The inference batch size is set to 1; code needs to be modified
    to support larger batch sizes, including resizing appropriately.
    """

    # Number of decimal places to round to for confidence and bbox coordinates
    CONF_DIGITS = 3
    COORD_DIGITS = 4

    # MegaDetector was trained with batch size of 1, and the resizing function is a part
    # of the inference graph
    BATCH_SIZE = 1

    # An enumeration of failure reasons
    FAILURE_TF_INFER = 'Failure TF inference'
    FAILURE_IMAGE_OPEN = 'Failure image access'

    DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = 0.85  # to render bounding boxes
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.3  # to include in the output json file

    DEFAULT_DETECTOR_LABEL_MAP = {
        '1': 'animal',
        '2': 'person',
        '4': 'vehicle'  # will be available in megadetector v4
    }

    NUM_DETECTOR_CATEGORIES = 4  # animal, person, group, vehicle - for color assignment

    def __init__(self, model_path):
        """Loads the model at model_path and start a tf.Session with this graph. The necessary
        input and output tensor handles are obtained also."""
        detection_graph = TFDetector.__load_model(model_path)
        self.tf_session = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def __convert_coords(np_array):
        """ Two functionalities: convert the numpy floats to Python floats, and also change the coordinates from
        [y1, x1, y2, x2] to [x1, y1, width_box, height_box] (in relative coordinates still).

        Args:
            np_array: array of predicted bounding box coordinates from the TF detector

        Returns: array of predicted bounding box coordinates as Python floats and in [x1, y1, width_box, height_box]

        """
        # change from [y1, x1, y2, x2] to [x1, y1, width_box, height_box]
        width_box = np_array[3] - np_array[1]
        height_box = np_array[2] - np_array[0]

        new = [np_array[1], np_array[0], width_box, height_box]  # cannot be a numpy array; needs to be a list

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = DetectorUtils.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new

    @staticmethod
    def __load_model(model_path):
        """Loads a detection model (i.e., create a graph) from a .pb file.

        Args:
            model_path: .pb file of the model.

        Returns: the loaded graph.

        """
        print('TFDetector: Loading graph...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('TFDetector: Detection graph loaded.')

        return detection_graph

    def _generate_detections_one_image(self, image):
        np_im = np.asarray(image, np.uint8)
        im_w_batch_dim = np.expand_dims(np_im, axis=0)

        # need to change the above line to the following if supporting a batch size > 1 and resizing to the same size
        # np_images = [np.asarray(image, np.uint8) for image in images]
        # images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)

        # performs inference
        (box_tensor_out, score_tensor_out, class_tensor_out) = self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor],
            feed_dict={self.image_tensor: im_w_batch_dim})

        return box_tensor_out, score_tensor_out, class_tensor_out

    def generate_detections_one_image(self, image, image_id,
                                      detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD):
        """Apply the detector to an image.

        Args:
            image: the PIL Image object
            image_id: a path to identify the image; will be in the `file` field of the output object
            detection_threshold: confidence above which to include the detection proposal

        Returns:
        A dict with the following fields, see https://github.com/microsoft/CameraTraps/tree/siyu/inference_refactor/api/batch_processing#batch-processing-api-output-format
            - image_id (always present)
            - max_detection_conf
            - detections, which is a list of detection objects containing `category`, `conf` and `bbox`
            - failure
        """
        result = {
            'file': image_id
        }
        try:
            b_box, b_score, b_class = self._generate_detections_one_image(image)

            # our batch size is 1; need to loop the batch dim if supporting batch size > 1
            boxes, scores, classes = b_box[0], b_score[0], b_class[0]

            detections_cur_image = []  # will be empty for an image with no confident detections
            max_detection_conf = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_entry = {
                        'category': str(int(c)),  # use string type for the numerical class label, not int
                        'conf': DetectorUtils.truncate_float(float(s),  # cast to float for json serialization
                                                             precision=TFDetector.CONF_DIGITS),
                        'bbox': TFDetector.__convert_coords(b)
                    }
                    detections_cur_image.append(detection_entry)
                    if s > max_detection_conf:
                        max_detection_conf = s

            result['max_detection_conf'] = DetectorUtils.truncate_float(float(max_detection_conf),
                                                                        precision=TFDetector.CONF_DIGITS)
            result['detections'] = detections_cur_image

        except Exception as e:
            result['failure'] = TFDetector.FAILURE_TF_INFER
            print('TFDetector: image {} failed during inference: {}'.format(image_id, str(e)))

        return result


#%% Main function

def load_and_run_detector(model_file, image_file_names, output_dir,
                          render_confidence_threshold=TFDetector.DEFAULT_RENDERING_CONFIDENCE_THRESHOLD):
    if len(image_file_names) == 0:
        print('Warning: no files available')
        return

    # load and run detector on target images, and visualize the results
    start_time = time.time()
    tf_detector = TFDetector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))

    detection_results = []
    time_load = []
    time_infer = []

    # since we'll be writing a bunch of files to the same folder, rename
    # as necessary to avoid collisions
    output_file_names = {}

    for im_file in tqdm(image_file_names):
        try:
            start_time = time.time()

            image = DetectorUtils.load_image(im_file)

            elapsed = time.time() - start_time
            time_load.append(elapsed)
        except Exception as e:
            print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': TFDetector.FAILURE_IMAGE_OPEN
            }
            detection_results.append(result)
            continue

        try:
            start_time = time.time()

            result = tf_detector.generate_detections_one_image(image, im_file)
            detection_results.append(result)

            elapsed = time.time() - start_time
            time_infer.append(elapsed)
        except Exception as e:
            print('An error occurred while running the detector on image {}. Exception: {}'.format(im_file, e))
            # the error code and message is written by generate_detections_one_image,
            # which is wrapped in a big try catch
            continue

        try:
            # image is modified in place
            DetectorUtils.render_detection_bounding_boxes(result['detections'], image,
                                                          label_map=TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
                                                          confidence_threshold=render_confidence_threshold)
            fn = os.path.basename(im_file).lower()
            name, ext = os.path.splitext(fn)
            fn = '{}{}{}'.format(name, ImagePathUtils.DETECTION_FILENAME_INSERT, '.jpg')  # save all as JPG
            if fn in output_file_names:
                n_collisions = output_file_names[fn]  # if there were a collision, the count is at least 1
                fn = str(n_collisions) + '_' + fn
                output_file_names[fn] = n_collisions + 1
            else:
                output_file_names[fn] = 0

            output_full_path = os.path.join(output_dir, fn)
            image.save(output_full_path)

        except Exception as e:
            print('Visualizing results on the image {} failed. Exception: {}'.format(im_file, e))
            continue

    ave_time_load = statistics.mean(time_load)
    ave_time_infer = statistics.mean(time_infer)
    if len(time_load) > 1 and len(time_infer) > 1:
        std_dev_time_load = humanfriendly.format_timespan(statistics.stdev(time_load))
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_load = 'not available'
        std_dev_time_infer = 'not available'
    print('On average, for each image,')
    print('- loading took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_load),
                                                    std_dev_time_load))
    print('- inference took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_infer),
                                                      std_dev_time_infer))


#%% Command-line driver

def main():
    parser = argparse.ArgumentParser(
        description='Module to run a TF animal detection model on images'
    )
    parser.add_argument(
        'detector_file',
        help='Path to .pb TensorFlow detector model file'
    )
    group = parser.add_mutually_exclusive_group(required=True)  # must specify either an image file or a directory
    group.add_argument(
        '--image_file',
        help='Single file to process, mutually exclusive with --image_dir')
    group.add_argument(
        '--image_dir',
        help='Directory to search for images, with optional recursion by adding --recursive'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if using --image_dir'
    )
    parser.add_argument(
        '--output_dir',
        help='Directory for output images (defaults to same as input)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=TFDetector.DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
        help='Confidence threshold between 0 and 1.0; only render boxes above this confidence (but only boxes above 0.3 confidence will be considered at all)'
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), 'detector_file specified does not exist'
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'  # Python chained comparison

    if args.image_file:
        image_file_names = [args.image_file]
    else:
        image_file_names = ImagePathUtils.find_images(args.image_dir, args.recursive)

    print('Running detector on {} images...'.format(len(image_file_names)))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.image_dir:
            args.output_dir = args.image_dir
        else:
            # but for a single image, args.image_dir is also None
            args.output_dir = os.path.dirname(args.image_file)

    load_and_run_detector(model_file=args.detector_file,
                          image_file_names=image_file_names,
                          output_dir=args.output_dir,
                          render_confidence_threshold=args.threshold)


if __name__ == '__main__':
    main()

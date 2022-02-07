import io
import json
import math
import os
import sys
from datetime import datetime
from io import BytesIO
from typing import Union

from PIL import Image
import numpy as np
import requests
import tensorflow as tf
from azure.storage.blob import ContainerClient

print('score.py, tensorflow version:', tf.__version__)
print('score.py, tf.test.is_gpu_available:', tf.test.is_gpu_available())

PRINT_EVERY = 500


#%% Helper functions *copied* from ct_utils.py and visualization/visualization_utils.py
IMAGE_ROTATIONS = {
    3: 180,
    6: 270,
    8: 90
}

def truncate_float(x, precision=3):
    """
    Function for truncating a float scalar to the defined precision.
    For example: truncate_float(0.0003214884) --> 0.000321
    This function is primarily used to achieve a certain float representation
    before exporting to JSON
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
        return math.floor(x * factor)/factor


def open_image(input_file: Union[str, BytesIO]) -> Image:
    """Opens an image in binary format using PIL.Image and converts to RGB mode.

    This operation is lazy; image will not be actually loaded until the first
    operation that needs to load it (for example, resizing), so file opening
    errors can show up later.

    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes

    Returns:
        an PIL image object in RGB mode
    """
    if (isinstance(input_file, str)
            and input_file.startswith(('http://', 'https://'))):
        response = requests.get(input_file)
        image = Image.open(BytesIO(response.content))
        try:
            response = requests.get(input_file)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f'Error opening image {input_file}: {e}')
            raise
    else:
        image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L'):
        raise AttributeError(f'Image {input_file} uses unsupported mode {image.mode}')
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')

    # alter orientation as needed according to EXIF tag 0x112 (274) for Orientation
    # https://gist.github.com/dangtrinhnt/a577ece4cbe5364aad28
    # https://www.media.mit.edu/pia/Research/deepview/exif.html
    try:
        exif = image._getexif()
        orientation: int = exif.get(274, None)  # 274 is the key for the Orientation field
        if orientation is not None and orientation in IMAGE_ROTATIONS:
            image = image.rotate(IMAGE_ROTATIONS[orientation], expand=True)  # returns a rotated copy
    except Exception:
        pass

    return image


def load_image(input_file: Union[str, BytesIO]) -> Image.Image:
    """Loads the image at input_file as a PIL Image into memory.
    Image.open() used in open_image() is lazy and errors will occur downstream
    if not explicitly loaded.
    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes
    Returns: PIL.Image.Image, in RGB mode
    """
    image = open_image(input_file)
    image.load()
    return image


#%% TFDetector class, an unmodified *copy* of the class in detection/tf_detector.py,
# so we do not have to import the packages required by run_detector.py

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
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.1  # to include in the output json file

    DEFAULT_DETECTOR_LABEL_MAP = {
        '1': 'animal',
        '2': 'person',
        '3': 'vehicle'  # available in megadetector v4+
    }

    NUM_DETECTOR_CATEGORIES = 4  # animal, person, group, vehicle - for color assignment

    def __init__(self, model_path):
        """Loads model from model_path and starts a tf.Session with this graph. Obtains
        input and output tensor handles."""
        detection_graph = TFDetector.__load_model(model_path)
        self.tf_session = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def round_and_make_float(d, precision=4):
        return truncate_float(float(d), precision=precision)

    @staticmethod
    def __convert_coords(tf_coords):
        """Converts coordinates from the model's output format [y1, x1, y2, x2] to the
        format used by our API and MegaDB: [x1, y1, width, height]. All coordinates
        (including model outputs) are normalized in the range [0, 1].
        Args:
            tf_coords: np.array of predicted bounding box coordinates from the TF detector,
                has format [y1, x1, y2, x2]
        Returns: list of Python float, predicted bounding box coordinates [x1, y1, width, height]
        """
        # change from [y1, x1, y2, x2] to [x1, y1, width, height]
        width = tf_coords[3] - tf_coords[1]
        height = tf_coords[2] - tf_coords[0]

        new = [tf_coords[1], tf_coords[0], width, height]  # must be a list instead of np.array

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new

    @staticmethod
    def convert_to_tf_coords(array):
        """From [x1, y1, width, height] to [y1, x1, y2, x2], where x1 is x_min, x2 is x_max
        This is an extraneous step as the model outputs [y1, x1, y2, x2] but were converted to the API
        output format - only to keep the interface of the sync API.
        """
        x1 = array[0]
        y1 = array[1]
        width = array[2]
        height = array[3]
        x2 = x1 + width
        y2 = y1 + height
        return [y1, x1, y2, x2]

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
            image_id: a path to identify the image; will be in the "file" field of the output object
            detection_threshold: confidence above which to include the detection proposal
        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of detection objects containing keys 'category', 'conf' and 'bbox'
            - 'failure'
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
                        'conf': truncate_float(float(s),  # cast to float for json serialization
                                               precision=TFDetector.CONF_DIGITS),
                        'bbox': TFDetector.__convert_coords(b)
                    }
                    detections_cur_image.append(detection_entry)
                    if s > max_detection_conf:
                        max_detection_conf = s

            result['max_detection_conf'] = truncate_float(float(max_detection_conf),
                                                          precision=TFDetector.CONF_DIGITS)
            result['detections'] = detections_cur_image

        except Exception as e:
            result['failure'] = TFDetector.FAILURE_TF_INFER
            print('TFDetector: image {} failed during inference: {}'.format(image_id, str(e)))

        return result


#%% Scoring script

class BatchScorer:
    """
    Coordinates scoring images in this Task.

    1. have a synchronized queue that download tasks enqueue and scoring function dequeues - but need to be able to
    limit the size of the queue. We do not want to write the image to disk and then load it in the scoring func.
    """
    def __init__(self, **kwargs):
        print('score.py BatchScorer, __init__()')

        detector_path = kwargs.get('detector_path')
        self.detector = TFDetector(detector_path)

        self.use_url = kwargs.get('use_url')
        if not self.use_url:
            input_container_sas = kwargs.get('input_container_sas')
            self.input_container_client = ContainerClient.from_container_url(input_container_sas)

        self.detection_threshold = kwargs.get('detection_threshold')

        self.image_ids_to_score = kwargs.get('image_ids_to_score')

        # determine if there is metadata attached to each image_id
        self.metadata_available = True if isinstance(self.image_ids_to_score[0], list) else False

    def _download_image(self, image_file) -> Image:
        """
        Args:
            image_file: Public URL if use_url, else the full path from container root

        Returns:
            PIL image loaded
        """
        if not self.use_url:
            downloader = self.input_container_client.download_blob(image_file)
            image_file = io.BytesIO()
            blob_props = downloader.download_to_stream(image_file)

        image = open_image(image_file)
        return image

    def score_images(self) -> list:
        detections = []

        for i in self.image_ids_to_score:

            if self.metadata_available:
                image_id = i[0]
                image_metadata = i[1]
            else:
                image_id = i

            try:
                image = self._download_image(image_id)
            except Exception as e:
                print(f'score.py BatchScorer, score_images, download_image exception: {e}')
                result = {
                    'file': image_id,
                    'failure': TFDetector.FAILURE_IMAGE_OPEN
                }
            else:
                result = self.detector.generate_detections_one_image(
                    image, image_id, detection_threshold=self.detection_threshold)

            if self.metadata_available:
                result['meta'] = image_metadata

            detections.append(result)
            if len(detections) % PRINT_EVERY == 0:
                print(f'scored {len(detections)} images')

        return detections


def main():
    print('score.py, main()')

    # information to determine input and output locations
    api_instance_name = os.environ['API_INSTANCE_NAME']
    job_id = os.environ['AZ_BATCH_JOB_ID']
    task_id = os.environ['AZ_BATCH_TASK_ID']
    mount_point = os.environ['AZ_BATCH_NODE_MOUNTS_DIR']

    # other parameters for the task
    begin_index = int(os.environ['TASK_BEGIN_INDEX'])
    end_index = int(os.environ['TASK_END_INDEX'])

    input_container_sas = os.environ.get('JOB_CONTAINER_SAS', None)  # could be None if use_url
    use_url = os.environ.get('JOB_USE_URL', None)

    if use_url and use_url.lower() == 'true':  # bool of any non-empty string is True
        use_url = True
    else:
        use_url = False

    detection_threshold = float(os.environ['DETECTION_CONF_THRESHOLD'])

    print(f'score.py, main(), api_instance_name: {api_instance_name}, job_id: {job_id}, task_id: {task_id}, '
          f'mount_point: {mount_point}, begin_index: {begin_index}, end_index: {end_index}, '
          f'input_container_sas: {input_container_sas}, use_url (parsed): {use_url}'
          f'detection_threshold: {detection_threshold}')

    job_folder_mounted = os.path.join(mount_point, 'batch-api', f'api_{api_instance_name}', f'job_{job_id}')
    task_out_dir = os.path.join(job_folder_mounted, 'task_outputs')
    os.makedirs(task_out_dir, exist_ok=True)
    task_output_path = os.path.join(task_out_dir, f'job_{job_id}_task_{task_id}.json')

    # test that we can write to output path; also in case there is no image to process
    with open(task_output_path, 'w') as f:
        json.dump([], f)

    # list images to process
    list_images_path = os.path.join(job_folder_mounted, f'{job_id}_images.json')
    with open(list_images_path) as f:
        list_images = json.load(f)
    print(f'score.py, main(), length of list_images: {len(list_images)}')

    if (not isinstance(list_images, list)) or len(list_images) == 0:
        print('score.py, main(), zero images in specified overall list, exiting...')
        sys.exit(0)

    # items in this list can be strings or [image_id, metadata]
    list_images = list_images[begin_index: end_index]
    if len(list_images) == 0:
        print('score.py, main(), zero images in the shard, exiting')
        sys.exit(0)

    print(f'score.py, main(), processing {len(list_images)} images in this Task')

    # model path
    # Path to .pb TensorFlow detector model file, relative to the
    # models/megadetector_copies folder in mounted container
    detector_model_rel_path = os.environ['DETECTOR_REL_PATH']
    detector_path = os.path.join(mount_point, 'models', 'megadetector_copies', detector_model_rel_path)
    assert os.path.exists(detector_path), f'detector is not found at the specified path: {detector_path}'

    # score the images
    scorer = BatchScorer(
        detector_path=detector_path,
        use_url=use_url,
        input_container_sas=input_container_sas,
        detection_threshold=detection_threshold,
        image_ids_to_score=list_images
    )

    try:
        tick = datetime.now()
        detections = scorer.score_images()
        duration = datetime.now() - tick
        print(f'score.py, main(), score_images() duration: {duration}')
    except Exception as e:
        raise RuntimeError(f'score.py, main(), exception in score_images(): {e}')

    with open(task_output_path, 'w', encoding='utf-8') as f:
        json.dump(detections, f, ensure_ascii=False)


if __name__ == '__main__':
    main()

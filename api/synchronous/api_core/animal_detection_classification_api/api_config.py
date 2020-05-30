# general API configurations

# upper limit on total content length (all images and parameters)
MAX_CONTENT_LENGTH_IN_MB = 5 * 8  # 5MB per image * number of images allowed

MAX_IMAGES_ACCEPTED = 8

IMAGE_CONTENT_TYPES = ['image/png', 'application/octet-stream', 'image/jpeg']

GPU_BATCH_SIZE = 8

# Camera trap images are usually 4:3 width to height
# The config of the model in use (model/pipeline.config) has min_dimension
# 600 and max_dimension 1024 for the keep_aspect_ratio_resizer, which first resize an image so
# that the smaller edge is 600 pixels; if the longer edge is now more than 1024, it resizes such
# that the longer edge is 1024 pixels
# (https://github.com/tensorflow/models/issues/1794)
MIN_DIM = 600
MAX_DIM = 1024


# classification configurations

CLASSIFICATION_MODEL_PATHS = {
    'serengeti': '/app/animal_detection_classification_api/model/snapshot_serengeti_cropped_inceptionV4_2019_05_28_w_preprocessing.pb',
    'caltech': '/app/animal_detection_classification_api/model/cct_cropped_inceptionV4_2019_04_11_w_preprocessing.pb'
}

CLASSIFICATION_CLASS_NAMES = {
    'serengeti': '/app/animal_detection_classification_api/classnames/snapshot_serengeti_cropped_inceptionV4_2019_05_28_classlist.txt',
    'caltech': '/app/animal_detection_classification_api/classnames/cct_cropped_inceptionV4_2019_04_11_classlist.txt'
}

DETECTION_CATEGORY_WHITELIST = ['1']

# padding factor used for padding the detected animal 
PADDING_FACTOR = 1.6 

# Minimum detection confidence for showing a bounding box on the output image
DEFAULT_CONFIDENCE_THRESHOLD = 0.85 

# Number of top-scoring classes to show at each bounding box
NUM_ANNOTATED_CLASSES = 3

# Number of significant float digits in JSON output
NUM_SIGNIFICANT_DIGITS = 3



# detection configurations

DETECTOR_MODEL_PATH = '/app/animal_detection_classification_api/model/megadetector_v3_tf19.pb'

DETECTOR_MODEL_VERSION = 'megadetector_v3_tf19'

DEFAULT_DETECTION_CONFIDENCE = 0.9

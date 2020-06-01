# sync API configurations


# upper limit on total content length (all images and parameters)
MAX_CONTENT_LENGTH_IN_MB = 5 * 8  # 5MB per image * number of images allowed

MAX_IMAGES_ACCEPTED = 8

IMAGE_CONTENT_TYPES = ['image/png', 'application/octet-stream', 'image/jpeg']


# classification configurations
# TODO

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

DETECTOR_MODEL_PATH = '/app/animal_detection_classification_api/model/md_v4.1.0.pb'

DETECTOR_MODEL_VERSION = 'v4.1.0'

DEFAULT_DETECTION_CONFIDENCE = 0.9

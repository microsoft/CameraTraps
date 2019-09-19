# upper limit on total content length (all images and parameters)
MAX_CONTENT_LENGTH_IN_MB = 5 * 8  # 5MB per image * number of images allowed

MAX_IMAGES_ACCEPTED = 8

IMAGE_CONTENT_TYPES = ['image/png', 'application/octet-stream', 'image/jpeg']

GPU_BATCH_SIZE = 8

MODEL_PATH = '/app/animal_detection_api/model/megadetector_v3_tf19.pb'

CLASSIFICATION_MODEL_PATHS = {
    'serengeti': '/app/animal_detection_api/model/serengeti.pb',
    'caltech': '/app/animal_detection_api/model/caltech.pb'
}

CLASSIFICATION_CLASS_NAMES = {
    'serengeti': '/app/animal_detection_api/classnames/serengeti.txt',
    'caltech': '/app/animal_detection_api/classnames/caltech.txt'
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





MODEL_VERSION = 'megadetector_v3_tf19'

# Camera trap images are usually 4:3 width to height
# The config of the model in use (model/pipeline.config) has min_dimension
# 600 and max_dimension 1024 for the keep_aspect_ratio_resizer, which first resize an image so
# that the smaller edge is 600 pixels; if the longer edge is now more than 1024, it resizes such
# that the longer edge is 1024 pixels
# (https://github.com/tensorflow/models/issues/1794)
MIN_DIM = 600
MAX_DIM = 1024

DEFAULT_DETECTION_CONFIDENCE = 0.9

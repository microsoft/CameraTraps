## Camera trap real-time API configuration

REDIS_HOST = "localhost"

REDIS_PORT = 6379

TEMP_FOLDER = "temp"

REDIS_QUEUE = "camera-trap-queue"

# Upper limit on total content length (all images and parameters)
MAX_CONTENT_LENGTH_IN_MB = 5 * 8  # 5MB per image * number of images allowed

MAX_IMAGES_ACCEPTED = 8

IMAGE_CONTENT_TYPES = ['image/png', 'application/octet-stream', 'image/jpeg']

DETECTION_CATEGORY_WHITELIST = ['1']

# Padding factor used for padding the detected animal 
PADDING_FACTOR = 1.6 

# Minimum detection confidence for showing a bounding box on the output image
DEFAULT_CONFIDENCE_THRESHOLD = 0.85 

# Number of top-scoring classes to show at each bounding box
NUM_ANNOTATED_CLASSES = 3

# Number of significant float digits in JSON output
NUM_SIGNIFICANT_DIGITS = 3

DETECTOR_MODEL_PATH = '/app/animal_detection_api/model/md_v4.1.0.pb'

# Use this when testing without docker
#DETECTOR_MODEL_PATH = 'model/md_v4.1.0.pb'

DETECTOR_MODEL_VERSION = 'v4.1.0'

DEFAULT_DETECTION_CONFIDENCE = 0.8

API_PREFIX='/v1/camera-trap/sync'

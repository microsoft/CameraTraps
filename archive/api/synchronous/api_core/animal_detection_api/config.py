## Camera trap real-time API configuration

REDIS_HOST = 'localhost'

REDIS_PORT = 6379

# Full path to the temporary folder for image storage, only meaningful 
# within the Docker container
TEMP_FOLDER = '/app/temp'

REDIS_QUEUE_NAME = 'camera-trap-queue'

# Upper limit on total content length (all images and parameters)
MAX_CONTENT_LENGTH_IN_MB = 5 * 8  # 5MB per image * number of images allowed

MAX_IMAGES_ACCEPTED = 8

IMAGE_CONTENT_TYPES = ['image/png', 'application/octet-stream', 'image/jpeg']

DETECTOR_MODEL_PATH = '/app/animal_detection_api/model/md_v5a.0.0.pt'

DETECTOR_MODEL_VERSION = 'v5a.0.0'

# Minimum confidence threshold for detections
DEFAULT_CONFIDENCE_THRESHOLD = 0.01

# Minimum confidence threshold for showing a bounding box on the output image
DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = 0.2

API_PREFIX = '/v1/camera-trap/sync'

API_KEYS_FILE = 'allowed_keys.txt'

# Use this when testing without Docker
DETECTOR_MODEL_PATH_DEBUG = 'model/md_v5a.0.0.pt'

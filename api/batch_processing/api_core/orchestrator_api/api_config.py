# name of the container in the internal storage account to store user facing files:
# image list, detection results and failed images list.
INTERNAL_CONTAINER = 'async-api'

# name of the container in the internal storage account to store outputs of each AML job
AML_CONTAINER = 'aml-out'

MONITOR_PERIOD_MINUTES = 15

# if this number of times the thread wakes up to check is exceeded, stop the monitoring thread
MAX_MONITOR_CYCLES = 4 * 7 * int((60 * 24) / MONITOR_PERIOD_MINUTES)  # 4 weeks

# number of retries in the monitoring thread for getting job status and aggregating results (each counted separately)
NUM_RETRIES = 2

# lower case; must be tuple for endswith to take as arg
ACCEPTED_IMAGE_FILE_ENDINGS = ('.jpeg', '.jpg')

# max number of images in a container to accept for processing
MAX_NUMBER_IMAGES_ACCEPTED = 2 * 1000 * 1000  # accept up to 2 million images

# how many images are processed by each call to the scoring API
NUM_IMAGES_PER_JOB = 2000

# update API task manager after submitting x jobs to AML Compute
JOB_SUBMISSION_UPDATE_INTERVAL = 2

# AML Compute
AML_CONFIG = {
    'subscription_id': '',
    'workspace_region': 'southcentralus',
    'resource_group': 'camera_trap_api_rg',
    'workspace_name': 'camera_trap_aml_ws_sc',
    'aml_compute_name': 'camera-trap-com',

    'default_model_version': '3',
    'models': {
        '3': 'megadetector_v3_tf19',  # user input model_version : name of model registered with AML
        '2': 'megadetector_v2',
        'benchmark_190904_cct20_tf19': 'benchmark_190904_cct20_tf19',
        'benchmark_190904_ss1_tf19': 'benchmark_190904_ss1_tf19'
    },

    'source_dir': '/app/orchestrator_api/aml_scripts',
    'script_name': 'score.py',

    'param_batch_size': 8,
    'param_detection_threshold': 0.3,  # megadetector v3 tends to have more very low confident detections and issues with NMS

    'completed_status': ['Finished', 'Failed', 'Completed', 'Canceled'],

    # service principle for authenticating to AML
    'tenant-id': '',  # fill these out before building the container
    'application-id': ''
}

WHITELIST = []

# version of the detector model in use
SUPPORTED_MODEL_VERSIONS = sorted([k for k in AML_CONFIG['models']])

# max number of blobs to list in the output blob container
MAX_BLOBS_IN_OUTPUT_CONTAINER = 1000 * 1000

# URLs to the 3 output files expires after this many days
EXPIRATION_DAYS = 90

DETECTION_CATEGORIES = {
    '1': 'animal',
    '2': 'person'
}

OUTPUT_FORMAT_VERSION = '1.0'

# version of the detector model in use
MODEL_VERSION = 'models/object_detection/faster_rcnn_inception_resnet_v2_atrous/megadetector'

# name of the container in the internal storage account to store user facing files:
# image list, detection results and failed images list.
INTERNAL_CONTAINER = 'async-api-v2'

# name of the container in the internal storage account to store outputs of each AML job
AML_CONTAINER = 'aml-out'

# how often does the checking thread wake up to check if all jobs are done
MONITOR_PERIOD_MINUTES = 30

# if this number of times the thread wakes up to check is exceeded, stop the monitoring thread
MAX_MONITOR_CYCLES = 14 * 48  # 2 weeks, 30-minute interval

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
    'subscription_id': '74d91980-e5b4-4fd9-adb6-263b8f90ec5b',
    'workspace_region': 'eastus',
    'resource_group': 'camera_trap_api_rg',
    'workspace_name': 'camera_trap_aml_workspace',
    'aml_compute_name': 'camera-trap-com',

    'model_name': 'megadetector',

    'source_dir': '/app/orchestrator_api/aml_config_scripts',
    'script_name': 'score.py',

    'param_batch_size': 8,
    'param_detection_threshold': 0.05,

    'completed_status': ['Finished', 'Failed', 'Completed'],

    # service principle for authenticating to AML
    'tenant-id': 'my-tenant-id',  # place holders
    'application-id': 'my-application-id'
}

# max number of blobs to list in the output blob container
MAX_BLOBS_IN_OUTPUT_CONTAINER = 1000 * 1000

# URLs to the 3 output files expires after this many days
EXPIRATION_DAYS = 14

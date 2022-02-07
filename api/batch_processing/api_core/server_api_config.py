# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
A module to hold the configurations specific to an instance of the API.
"""

import os


#%% instance-specific API settings
# you likely need to modify these when deploying a new instance of the API

API_INSTANCE_NAME = 'cm'  # 'internal', 'cm', 'camelot', 'zooniverse'
POOL_ID = 'cm_1'  # name of the Batch pool created for this API instance

MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB = 4 * 1000 * 1000  # inclusive

# Azure Batch for batch processing
BATCH_ACCOUNT_NAME = 'cameratrapssc'
BATCH_ACCOUNT_URL = 'https://cameratrapssc.southcentralus.batch.azure.com'


#%% general API settings
API_PREFIX = '/v4/camera-trap/detection-batch'  # URL to root is http://127.0.0.1:5000/v4/camera-trap/detection-batch/

MONITOR_PERIOD_MINUTES = 10

# if this number of times the thread wakes up to check is exceeded, stop the monitoring thread
MAX_MONITOR_CYCLES = 4 * 7 * int((60 * 24) / MONITOR_PERIOD_MINUTES)  # 4 weeks

IMAGE_SUFFIXES_ACCEPTED = ('.jpg', '.jpeg', '.png')  # case-insensitive
assert isinstance(IMAGE_SUFFIXES_ACCEPTED, tuple)

OUTPUT_FORMAT_VERSION = '1.1'

NUM_IMAGES_PER_TASK = 2000

OUTPUT_SAS_EXPIRATION_DAYS = 180

# quota of active Jobs in our Batch account, which all node pools i.e. API instances share;
# cannot accept job submissions if there are this many active Jobs already
MAX_BATCH_ACCOUNT_ACTIVE_JOBS = 300


#%% MegaDetector info
DETECTION_CONF_THRESHOLD = 0.1

# relative to the `megadetector_copies` folder in the container `models`
# TODO add MD versions info to AppConfig
MD_VERSIONS_TO_REL_PATH = {
    '4.1': 'megadetector_v4_1/md_v4.1.0.pb',
    '3': 'megadetector_v3/megadetector_v3_tf19.pb',
    '2': 'megadetector_v2/frozen_inference_graph.pb'
}
DEFAULT_MD_VERSION = '4.1'
assert DEFAULT_MD_VERSION in MD_VERSIONS_TO_REL_PATH

# copied from TFDetector class in detection/run_detector.py
DETECTOR_LABEL_MAP = {
    '1': 'animal',
    '2': 'person',
    '3': 'vehicle'
}


#%% Azure Batch settings
NUM_TASKS_PER_SUBMISSION = 20  # max for the Python SDK without extension is 100

NUM_TASKS_PER_RESUBMISSION = 5


#%% env variables for service credentials, and info related to these services

# Cosmos DB `batch-api-jobs` table for job status
COSMOS_ENDPOINT = os.environ['COSMOS_ENDPOINT']
COSMOS_WRITE_KEY = os.environ['COSMOS_WRITE_KEY']

# Service principal of this "application", authorized to use Azure Batch
APP_TENANT_ID = os.environ['APP_TENANT_ID']
APP_CLIENT_ID = os.environ['APP_CLIENT_ID']
APP_CLIENT_SECRET = os.environ['APP_CLIENT_SECRET']

# Blob storage account for storing Batch tasks' outputs and scoring script
STORAGE_ACCOUNT_NAME = os.environ['STORAGE_ACCOUNT_NAME']
STORAGE_ACCOUNT_KEY = os.environ['STORAGE_ACCOUNT_KEY']

# STORAGE_CONTAINER_MODELS = 'models'  # names of the two containers supporting Batch
STORAGE_CONTAINER_API = 'batch-api'

# Azure Container Registry for Docker image used by our Batch node pools
REGISTRY_SERVER = os.environ['REGISTRY_SERVER']
REGISTRY_PASSWORD = os.environ['REGISTRY_PASSWORD']
CONTAINER_IMAGE_NAME = 'cameratracrsppftkje.azurecr.io/tensorflow:1.14.0-gpu-py3'

# Azure App Configuration instance to get configurations specific to
# this instance of the API
APP_CONFIG_CONNECTION_STR = os.environ['APP_CONFIG_CONNECTION_STR']

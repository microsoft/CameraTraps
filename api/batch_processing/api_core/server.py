# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import math
import os
import time
from datetime import datetime
from random import shuffle
import string
import urllib.parse
import uuid
from datetime import datetime

from flask import Flask, request, jsonify


#%% constants

# URL to root is http://127.0.0.1:5000/v4/camera-trap/detection-batch/
API_PREFIX = '/v4/camera-trap/detection-batch'

# configurations
API_INSTANCE_NAME = 'api_internal'
POOL_ID = 'internal_1'

OUTPUT_FORMAT_VERSION = '1.1'
STORAGE_CONTAINER_MODELS = 'models'
STORAGE_CONTAINER_API = 'batch-api'
NUM_IMAGES_PER_JOB = 2000
DETECTION_CONF_THRESHOLD = 0.1

# relative to the `megadetector_copies` folder in the container `models`
MD_VERSIONS_TO_REL_PATH = {
    '4.1': 'megadetector_v4_1/md_v4.1.0.pb',
    '3': 'megadetector_v3/megadetector_v3_tf19.pb',
    '2': 'megadetector_v2/frozen_inference_graph.pb'
}
DEFAULT_MD_VERSION = '4.1'
assert DEFAULT_MD_VERSION in MD_VERSIONS_TO_REL_PATH

# env variables

# Cosmos DB `batch-api-jobs` table for job status
# used by job_status_table but check that they are set properly here
COSMOS_ENDPOINT = os.environ['COSMOS_ENDPOINT']
COSMOS_WRITE_KEY = os.environ['COSMOS_WRITE_KEY']

# Service principal of this "application", authorized to use Azure Batch
APP_TENANT_ID = os.environ['APP_TENANT_ID']
APP_CLIENT_ID = os.environ['APP_CLIENT_ID']
APP_CLIENT_SECRET = os.environ['APP_CLIENT_SECRET']

# Blob storage account for storing Batch tasks' outputs and scoring script
STORAGE_ACCOUNT_NAME = os.environ['STORAGE_ACCOUNT_NAME']
STORAGE_ACCOUNT_KEY = os.environ['STORAGE_ACCOUNT_KEY']

# Azure Batch for batch processing
BATCH_ACCOUNT_NAME = os.environ['BATCH_ACCOUNT_NAME']
BATCH_ACCOUNT_URL = os.environ['BATCH_ACCOUNT_URL']

# Azure Container Registry for Docker image used by our Batch node pools
REGISTRY_SERVER = os.environ['REGISTRY_SERVER']
REGISTRY_PASSWORD = os.environ['REGISTRY_PASSWORD']

# Azure App Configuration instance to get configurations specific to
# this instance of the API
APP_CONFIG_CONNECTION_STR = os.environ['APP_CONFIG_CONNECTION_STR']


#%% Flask endpoints

print('server.py, creating Flask application...')

app = Flask(__name__)


@app.route(f'{API_PREFIX}/')
def hello():
    return 'Camera traps batch processing API'


@app.route(f'{API_PREFIX}/request_detections', methods=['POST'])
def request_detections():
    pass


@app.route(f'{API_PREFIX}/task')
def get_job_status():
    """
    Retains the /task endpoint name.
    """
    pass


@app.route(f'{API_PREFIX}/cancel_request', methods=['POST'])
def cancel_request():
    pass


@app.route(f'{API_PREFIX}/default_model_version')
def get_default_model_version():
    return DEFAULT_MD_VERSION


@app.route(f'{API_PREFIX}/supported_model_versions')
def get_supported_model_versions():
    return jsonify(sorted(list(MD_VERSIONS_TO_REL_PATH.keys())))

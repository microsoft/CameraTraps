# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import string
import uuid
from threading import Thread

import sas_blob_utils  # from ai4eutils
from flask import Flask, request, jsonify

import server_api_config as api_config
from server_app_config import AppConfig
from server_batch import BatchJobManager
from server_job import create_batch_job
from server_job_status_table import JobStatusTable
from server_utils import *

#%% Helper classes
app_config = AppConfig()
job_status_table = JobStatusTable()
batch_job_manager = BatchJobManager()
print('server.py, finished instantiating helper classes')


#%% Flask app and endpoints

print('server.py, creating Flask application...')

app = Flask(__name__)
API_PREFIX = api_config.API_PREFIX


@app.route(f'{API_PREFIX}/')
def hello():
    return 'Camera traps batch processing API'


@app.route(f'{API_PREFIX}/request_detections', methods=['POST'])
def request_detections():
    """
    Checks that the input parameters to this endpoint are valid, starts a thread
    to launch the batch processing job, and return the job_id/request_id to the user.
    """
    if not request.is_json:
        msg = 'Body needs to have a JSON mimetype (e.g., application/json).'
        return make_error(415, msg)

    try:
        post_body = request.get_json()
    except Exception as e:
        return make_error(415, f'Error occurred reading POST request body: {e}.')

    # required params

    caller_id = post_body.get('caller', None)
    if caller_id is None or caller_id not in app_config.get_allowlist():
        msg = ('Parameter caller is not supplied or is not on our allowlist. '
               'Please email cameratraps@microsoft.com to request access.')
        return make_error(401, msg)

    use_url = post_body.get('use_url', False)
    if use_url and isinstance(use_url, str):  # in case it is included but is intended to be False
        if use_url.lower() in ['false', 'f', 'no', 'n']:
            use_url = False

    input_container_sas = post_body.get('input_container_sas', None)
    if not input_container_sas and not use_url:
        msg = ('input_container_sas with read and list access is a required '
               'field when not using image URLs.')
        return make_error(400, msg)

    if input_container_sas is not None:
        if not sas_blob_utils.is_container_uri(input_container_sas):
            return make_error(400, 'input_container_sas provided is not for a container.')

        result = check_data_container_sas(input_container_sas)
        if result is not None:
            return make_error(result[0], result[1])

    images_requested_json_sas = post_body.get('images_requested_json_sas', None)
    if images_requested_json_sas is not None:
        exists = sas_blob_utils.check_blob_exists(images_requested_json_sas)
        if not exists:
            return make_error(400, 'images_requested_json_sas does not point to a valid file.')

    # if use_url, then images_requested_json_sas is required
    if use_url:
        if images_requested_json_sas is None:
            msg = 'images_requested_json_sas is required since use_url is true.'
            return make_error(400, msg)

    # optional params

    # check model_version is among the available model versions
    model_version = post_body.get('model_version', '')
    if model_version != '':
        model_version = str(model_version)  # in case user used an int
        if model_version not in api_config.MD_VERSIONS_TO_REL_PATH:  # TODO check AppConfig
            return make_error(400, f'model_version {model_version} is not supported.')

    # check request_name has only allowed characters
    request_name = post_body.get('request_name', '')
    if request_name != '':
        if len(request_name) > 92:
            return make_error(400, 'request_name is longer than 92 characters.')
        allowed = set(string.ascii_letters + string.digits + '_' + '-')
        if not set(request_name) <= allowed:
            msg = ('request_name contains invalid characters (only letters, '
                   'digits, - and _ are allowed).')
            return make_error(400, msg)

    # optional params for telemetry collection
    country = post_body.get('country', None)
    organization_name = post_body.get('organization_name', None)
    # TODO log this request to Insights

    try:
        job_id = uuid.uuid4().hex
        job_status_table.create_job_status(
            job_id=job_id,
            status='created',
            call_params=post_body
        )
    except Exception as e:
        return make_error(500, f'Error creating a job status entry: {e}')

    try:
        thread = Thread(target=create_batch_job, kwargs={'job_id': job_id, 'body': post_body})
        thread.start()
    except Exception as e:
        return make_error(500, f'Error creating or starting the batch processing thread: {e}')

    return {'request_id': job_id}


@app.route(f'{API_PREFIX}/cancel_request', methods=['POST'])
def cancel_request():
    """
    Cancels a request/job given the job_id and caller_id
    """
    if not request.is_json:
        msg = 'Body needs to have a JSON mimetype (e.g., application/json).'
        return make_error(415, msg)
    try:
        post_body = request.get_json()
    except Exception as e:
        return make_error(415, f'Error occurred reading POST request body: {e}.')

    # required fields
    job_id = post_body.get('task_id', None)
    if job_id is None:
        return make_error(400, 'task_id is a required field.')

    caller_id = post_body.get('caller', None)
    if caller_id is None or caller_id not in app_config.get_allowlist():
        return make_error(401, 'Parameter caller is not supplied or is not on our allowlist.')

    item_read = job_status_table.read_job_status(job_id)
    if item_read is None:
        return make_error(404, 'Task is not found.')
    if 'status' not in item_read:
        return make_error(404, 'Something went wrong. This task does not have a status field.')

    request_status = item_read['status']['request_status']
    if request_status not in ['running', 'problem']:
        # request_status is either completed or failed
        return make_error(400, f'Task has {request_status} and cannot be canceled')

    try:
        batch_job_manager.cancel_batch_job(job_id)
        # the create_batch_job thread will stop when it wakes up the next time
    except Exception as e:
        return make_error(500, f'Error when canceling the request: {e}')
    else:
        job_status_table.update_job_status(job_id, {
            'request_status': 'canceled',
            'message': 'Request has been canceled by the user.'
        })
    return 200, 'Canceling signal has been sent. You can verify the status at the /task endpoint'


@app.route(f'{API_PREFIX}/task/<job_id>')
def get_job_status(job_id):
    """
    Does not require the "caller" field to avoid checking the allowlist in App Configurations.
    Retains the /task endpoint name to be compatible with previous versions.
    """
    item_read = job_status_table.read_job_status(job_id)
    if item_read is None:
        return make_error(404, 'Task is not found.')
    if 'status' not in item_read or 'last_updated' not in item_read:
        return make_error(404, 'Something went wrong. This task does not have a valid status.')

    # conform to previous schemes
    item_to_return = {
        'Status': item_read['status'],
        'Endpoint': f'{API_PREFIX}/request_detections',
        'TaskId': job_id,
        'Timestamp': item_read['last_updated']
    }
    return item_to_return


@app.route(f'{API_PREFIX}/default_model_version')
def get_default_model_version() -> str:
    return api_config.DEFAULT_MD_VERSION


@app.route(f'{API_PREFIX}/supported_model_versions')
def get_supported_model_versions() -> str:
    return jsonify(sorted(list(api_config.MD_VERSIONS_TO_REL_PATH.keys())))

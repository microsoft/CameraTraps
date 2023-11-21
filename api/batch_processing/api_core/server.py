# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import string
import uuid
import threading
from datetime import timedelta

import sas_blob_utils  # from ai4eutils
from flask import Flask, request, jsonify

import server_api_config as api_config
from server_app_config import AppConfig
from server_batch_job_manager import BatchJobManager
from server_orchestration import create_batch_job, monitor_batch_job
from server_job_status_table import JobStatusTable
from server_utils import *

# %% Flask app
app = Flask(__name__)

# reference: https://trstringer.com/logging-flask-gunicorn-the-manageable-way/
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)


API_PREFIX = api_config.API_PREFIX
app.logger.info('server, created Flask application...')

# %% Helper classes

app_config = AppConfig()
job_status_table = JobStatusTable()
batch_job_manager = BatchJobManager()
app.logger.info('server, finished instantiating helper classes')


# %% Flask endpoints

@app.route(f'{API_PREFIX}/')
def hello():
    return f'Camera traps batch processing API. Instance: {api_config.API_INSTANCE_NAME}'


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

    app.logger.info(f'server, request_detections, post_body: {post_body}')

    # required params

    caller_id = post_body.get('caller', None)
    if caller_id is None or caller_id not in app_config.get_allowlist():
        msg = ('Parameter caller is not supplied or is not on our allowlist. '
               'Please email cameratraps@lila.science to request access.')
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

    # can be an URL to a file not hosted in an Azure blob storage container
    images_requested_json_sas = post_body.get('images_requested_json_sas', None)

    if images_requested_json_sas is not None:
        if not images_requested_json_sas.startswith(('http://', 'https://')):
            return make_error(400, 'images_requested_json_sas needs to be an URL.')

    # if use_url, then images_requested_json_sas is required
    if use_url and images_requested_json_sas is None:
            return make_error(400, 'images_requested_json_sas is required since use_url is true.')

    # optional params

    # check model_version is among the available model versions
    model_version = post_body.get('model_version', '')
    if model_version != '':
        model_version = str(model_version)  # in case user used an int
        if model_version not in api_config.MD_VERSIONS_TO_REL_PATH:  # TODO use AppConfig to store model version info
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

    # optional params for telemetry collection - logged to status table for now as part of call_params
    country = post_body.get('country', None)
    organization_name = post_body.get('organization_name', None)

    # All API instances / node pools share a quota on total number of active Jobs;
    # we cannot accept new Job submissions if we are at the quota
    try:
        num_active_jobs = batch_job_manager.get_num_active_jobs()
        if num_active_jobs >= api_config.MAX_BATCH_ACCOUNT_ACTIVE_JOBS:
            return make_error(503, f'Too many active jobs, please try again later')
    except Exception as e:
        return make_error(500, f'Error checking number of active jobs: {e}')

    try:
        job_id = uuid.uuid4().hex
        job_status_table.create_job_status(
            job_id=job_id,
            status= get_job_status('created', 'Request received. Listing images next...'),
            call_params=post_body
        )
    except Exception as e:
        return make_error(500, f'Error creating a job status entry: {e}')

    try:
        thread = threading.Thread(
            target=create_batch_job,
            name=f'job_{job_id}',
            kwargs={'job_id': job_id, 'body': post_body}
        )
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

    app.logger.info(f'server, cancel_request received, body: {post_body}')

    # required fields
    job_id = post_body.get('request_id', None)
    if job_id is None:
        return make_error(400, 'request_id is a required field.')

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
    return 'Canceling signal has been sent. You can verify the status at the /task endpoint'


@app.route(f'{API_PREFIX}/task/<job_id>')
def retrieve_job_status(job_id: str):
    """
    Does not require the "caller" field to avoid checking the allowlist in App Configurations.
    Retains the /task endpoint name to be compatible with previous versions.
    """
    # Fix for Zooniverse - deleting any "-" characters in the job_id
    job_id = job_id.replace('-', '')

    item_read = job_status_table.read_job_status(job_id)  # just what the monitoring thread wrote to the DB
    if item_read is None:
        return make_error(404, 'Task is not found.')
    if 'status' not in item_read or 'last_updated' not in item_read or 'call_params' not in item_read:
        return make_error(404, 'Something went wrong. This task does not have a valid status.')

    # If the status is running, it could be a Job submitted before the last restart of this
    # API instance. If that is the case, we should start to monitor its progress again.
    status = item_read['status']

    last_updated = datetime.fromisoformat(item_read['last_updated'][:-1])  # get rid of "Z" (required by Cosmos DB)
    time_passed = datetime.utcnow() - last_updated
    job_is_unmonitored = True if time_passed > timedelta(minutes=(api_config.MONITOR_PERIOD_MINUTES + 1)) else False

    if isinstance(status, dict) and \
            'request_status' in status and \
            status['request_status'] in ['running', 'problem'] and \
            'num_tasks' in status and \
            job_id not in get_thread_names() and \
            job_is_unmonitored:
        # WARNING model_version could be wrong (a newer version number gets written to the output file) around
        # the time that  the model is updated, if this request was submitted before the model update
        # and the API restart; this should be quite rare
        model_version = item_read['call_params'].get('model_version', api_config.DEFAULT_MD_VERSION)

        num_tasks = status['num_tasks']
        job_name = item_read['call_params'].get('request_name', '')
        job_submission_timestamp = item_read.get('job_submission_time', '')

        thread = threading.Thread(
            target=monitor_batch_job,
            name=f'job_{job_id}',
            kwargs={
                'job_id': job_id,
                'num_tasks': num_tasks,
                'model_version': model_version,
                'job_name': job_name,
                'job_submission_timestamp': job_submission_timestamp
            }
        )
        thread.start()
        app.logger.info(f'server, started a new thread to monitor job {job_id}')

    # conform to previous schemes
    if 'num_tasks' in status:
        del status['num_tasks']
    item_to_return = {
        'Status': status,
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


# %% undocumented endpoints

def get_thread_names() -> list:
    thread_names = []
    for thread in threading.enumerate():
        if thread.name.startswith('job_'):
            thread_names.append(thread.name.split('_')[1])
    return sorted(thread_names)


@app.route(f'{API_PREFIX}/all_jobs')
def get_all_jobs():
    """List all Jobs being monitored since this API instance started"""
    thread_names = get_thread_names()
    return jsonify(thread_names)

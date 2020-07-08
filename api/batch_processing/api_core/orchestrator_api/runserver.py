# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Python version is 3.5.2

import os
import string
import sys
from time import sleep

from flask import Flask, request, make_response, jsonify
from azure.storage.blob import BlockBlobService
# /ai4e_api_tools has been added to the PYTHONPATH, so we can reference those libraries directly.
from ai4e_app_insights_wrapper import AI4EAppInsights
from ai4e_service import APIService

import api_config
import orchestrator
from orchestrator import get_task_status
from sas_blob_utils import SasBlob  # file in this directory, not the ai4eutil repo one


print('Creating application')
app = Flask(__name__)

# Use the AI4EAppInsights library to send log messages. NOT REQUIRED
log = AI4EAppInsights()

# Use the APIService to executes your functions within a logging trace, supports long-running/async functions,
# handles SIGTERM signals from AKS, etc., and handles concurrent requests.
with app.app_context():
    ai4e_service = APIService(app, log)


# Instantiate blob storage service to the internal container to put intermediate results and files
storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
storage_account_key = os.getenv('STORAGE_ACCOUNT_KEY')
internal_storage_service = BlockBlobService(account_name=storage_account_name, account_key=storage_account_key)
internal_container = api_config.INTERNAL_CONTAINER
internal_datastore = {
    'account_name': storage_account_name,
    'account_key': storage_account_key,
    'container_name': internal_container
}
print('Internal storage container {} in account {} is set up.'.format(internal_container, storage_account_name))


def _abort(error_code, error_message):
    return make_response(jsonify({'error': error_message}), error_code)


def _request_detections_validate_params(request):
    """Returns an error straight-away without issuing the user a request_id if the parameters
    are not acceptable.

    Args:
        request: request body of the POST call

    Returns:
        dict of parameters to use in request_detections()
    """
    if not request.is_json:
        return _abort(415, 'Body needs to have a mimetype for JSON (e.g. application/json).')

    try:
        post_body = request.get_json()
    except Exception as e:
        return _abort(415, 'Error occurred while reading the POST request body: {}.'.format(str(e)))

    # required params
    caller_id = post_body.get('caller', None)
    if caller_id is None or caller_id not in api_config.ALLOWLIST:
        return _abort(401, ('Parameter caller is not supplied or is not on our allowlist. '
        'Please email cameratraps@microsoft.com to request access.'))

    use_url = post_body.get('use_url', False)

    input_container_sas = post_body.get('input_container_sas', None)
    if not input_container_sas and not use_url:
        return _abort(400, ('input_container_sas with read and list access is a required field when '
                            'not using image URLs.'))

    if input_container_sas is not None:
        result = orchestrator.check_data_container_sas(input_container_sas)
        if result is not None:
            return _abort(result[0], result[1])

    images_requested_json_sas = post_body.get('images_requested_json_sas', None)

    # if use_url, then images_requested_json_sas is required
    if use_url and images_requested_json_sas is None:
        return _abort(400, 'Since use_url is true, images_requested_json_sas is required.')

    # check model_version and request_name params are valid
    model_version = post_body.get('model_version', '')
    if model_version != '':
        model_version = str(model_version)  # in case an int is specified
        if model_version not in api_config.SUPPORTED_MODEL_VERSIONS:
            return _abort(400, 'model_version {} is not supported.'.format(model_version))

    # check request_name has only allowed characters
    request_name = post_body.get('request_name', '')
    if request_name != '':
        if len(request_name) > 92:
            return _abort(400, 'request_name is longer than 92 characters.')
        allowed = set(string.ascii_letters + string.digits + '_' + '-')
        if not set(request_name) <= allowed:
            return _abort(400, 'request_name contains unallowed characters (only letters, digits, - and _ are allowed).')

    first_n = post_body.get('first_n', None)

    sample_n = post_body.get('sample_n', None)

    model_version = post_body.get('model_version', '')
    if model_version == '':
        model_version = api_config.AML_CONFIG['default_model_version']
    model_name = api_config.AML_CONFIG['models'][model_version]

    # TODO check that the expiry date of input_container_sas is at least a few days into the future

    # TODO check images_requested_json_sas is a blob not a container

    return {
        'input_container_sas': input_container_sas,
        'images_requested_json_sas': images_requested_json_sas,
        'image_path_prefix': post_body.get('image_path_prefix', None),
        'first_n': int(first_n) if first_n else None,
        'sample_n': int(sample_n) if sample_n else None,
        'model_version': model_version,
        'model_name': model_name,
        'request_name': request_name,
        # request_name and request_submission_timestamp are for appending to output file names
        'request_submission_timestamp': orchestrator.get_utc_timestamp(),
        'use_url': use_url
    }


@ai4e_service.api_async_func(
    api_path = '/request_detections',
    methods = ['POST'],
    request_processing_function = _request_detections_validate_params, # This is the data process function that you created above.
    maximum_concurrent_requests = 1, # If the number of requests exceed this limit, a 503 is returned to the caller.
    content_types = ['application/json'],
    content_max_length = 10000, # In bytes
    trace_name = 'post:request_detections')
def request_detections(*args, **kwargs):
    # Since this is an async function, we need to keep the task updated.
    request_id = kwargs.get('taskId')
    input_container_sas = kwargs.get('input_container_sas')
    images_requested_json_sas = kwargs.get('images_requested_json_sas')
    image_path_prefix = kwargs.get('image_path_prefix')
    first_n = kwargs.get('first_n')
    sample_n = kwargs.get('sample_n')
    model_version = kwargs.get('model_version')
    model_name = kwargs.get('model_name')
    request_name = kwargs.get('request_name')
    request_submission_timestamp = kwargs.get('request_submission_timestamp')
    use_url = kwargs.get('use_url')

    log.log_debug('Started task', request_id)  # Log to Application Insights TODO - where does this log to?

    print(('runserver.py, request_id {}, model_version {}, model_name {}, request_name {}, '
           'request_submission_timestamp is {}').format(
        request_id, model_version, model_name, request_name, request_submission_timestamp))

    ai4e_service.api_task_manager.UpdateTaskStatus(request_id, get_task_status('running', 'Request received.'))

    sleep(10)  # TODO

    ai4e_service.api_task_manager.UpdateTaskStatus(request_id,
                                                   get_task_status('running',
                                                                   'Images submitted to cluster for processing.'))



# Define a function for processing request data, if applicable.  This function loads data or files into
# a dictionary for access in your API function.  We pass this function as a parameter to your API setup.
def process_request_data(request):
    return_values = {'data': None}
    try:
        # Attempt to load the body
        return_values['data'] = request.get_json()
    except:
        log.log_error('Unable to load the request data')   # Log to Application Insights
    return return_values

# Define a function that runs your model.  This could be in a library.
def run_model(taskId, body):
    # Update the task status, so the caller knows it has been accepted and is running.
    ai4e_service.api_task_manager.UpdateTaskStatus(taskId, 'running model')

    log.log_debug('Running model', taskId) # Log to Application Insights
    #INSERT_YOUR_MODEL_CALL_HERE
    sleep(10)  # replace with real code

# POST, long-running/async API endpoint example
@ai4e_service.api_async_func(
    api_path = '/example', 
    methods = ['POST'], 
    request_processing_function = process_request_data, # This is the data process function that you created above.
    maximum_concurrent_requests = 3, # If the number of requests exceed this limit, a 503 is returned to the caller.
    content_types = ['application/json'],
    content_max_length = 1000, # In bytes
    trace_name = 'post:my_long_running_funct')
def default_post(*args, **kwargs):
    # Since this is an async function, we need to keep the task updated.
    taskId = kwargs.get('taskId')
    log.log_debug('Started task', taskId) # Log to Application Insights

    # Get the data from the dictionary key that you assigned in your process_request_data function.
    request_json = kwargs.get('data')

    if not request_json:
        ai4e_service.api_task_manager.FailTask(taskId, 'Task failed - Body was empty or could not be parsed.')
        return -1

    # Run your model function
    run_model(taskId, request_json)

    # Once complete, ensure the status is updated.
    log.log_debug('Completed task', taskId) # Log to Application Insights
    # Update the task with a completion event.
    ai4e_service.api_task_manager.CompleteTask(taskId, 'completed')

# GET, sync API endpoint example
@ai4e_service.api_sync_func(api_path = '/echo/<string:text>', methods = ['GET'], maximum_concurrent_requests = 1000, trace_name = 'get:echo', kwargs = {'text'})
def echo(*args, **kwargs):
    return 'Echo: ' + kwargs['text']

if __name__ == '__main__':
    app.run()
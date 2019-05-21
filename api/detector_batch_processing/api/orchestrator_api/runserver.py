# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# # /ai4e_api_tools has been added to the PYTHONPATH, so we can reference those
# libraries directly.
import json
import math
import os
import time
from datetime import datetime
from random import shuffle
import string

from ai4e_app_insights import AppInsights
from ai4e_app_insights_wrapper import AI4EAppInsights
from ai4e_service import AI4EWrapper
from azure.storage.blob import BlockBlobService
from flask import Flask, request, make_response, jsonify
from flask_restful import Api
from task_management.api_task import ApiTaskManager

import api_config
import orchestrator
from orchestrator import get_task_status
from sas_blob_utils import SasBlob


print('Creating application')

api_prefix = os.getenv('API_PREFIX')
print('API prefix: ', api_prefix)
app = Flask(__name__)
api = Api(app)

# Log requests, traces and exceptions to the Application Insights service
appinsights = AppInsights(app)

# Use the AI4EAppInsights library to send log messages.
log = AI4EAppInsights()

# Use the internal-container AI for Earth Task Manager (not for production use!).
api_task_manager = ApiTaskManager(flask_api=api, resource_prefix=api_prefix)

# Use the AI4EWrapper to executes your functions within a logging trace.
# Also, helps support long-running/async functions.
ai4e_wrapper = AI4EWrapper(app)

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

@app.route('/', methods=['GET'])
def health_check():
    return 'Health check OK'


def _stop_aync(request_id, error_message):
    api_task_manager.UpdateTaskStatus(request_id, get_task_status('failed', 'Error: {}'.format(error_message)))
    # plus any telemetry to be collected


def _abort(error_code, error_message):
    return make_response(jsonify({'error': error_message}), error_code)


@app.route(api_prefix + '/request_detections', methods=['POST'])
def request_detections():
    if not request.is_json:
        return _abort(415, 'Body needs to have a mimetype for JSON (e.g. application/json).')

    try:
        post_body = request.get_json()
    except Exception as e:
        return _abort(415, 'Error occurred while reading the POST request body: {}.'.format(str(e)))

    # required param
    input_container_sas = post_body.get('input_container_sas', None)
    if not input_container_sas:
        return _abort(400, 'input_container_sas with read and list access is a required field.')

    result = orchestrator.check_data_container_sas(input_container_sas)
    if result is not None:
        return _abort(result[0], result[1])

    # check model_version and request_name params are valid
    model_version = post_body.get('model_version', None)
    if model_version is not None and model_version != '':
        model_version = str(model_version)  # in case an int is specified
        if model_version not in api_config.SUPPORTED_MODEL_VERSIONS:
            return _abort(400, 'model_version {} is not supported.'.format(model_version))

    # check request_name has only allowed characters
    request_name = post_body.get('request_name', '')
    if request_name != '':
        if len(request_name) > 32:
            return _abort(400, 'request_name is longer than 32 characters.')
        allowed = set(string.ascii_letters + string.digits + '_' + '-')
        if not set(request_name) <= allowed:
            return _abort(400, 'request_name contains unallowed characters (only letters, digits, - and _ are allowed).')

    # TODO check that the expiry date of input_container_sas is at least a month into the future

    # TODO check images_requested_json_sas is a blob not a container

    task_info = api_task_manager.AddTask('queued')
    request_id = str(task_info['uuid'])

    # wrap_async_endpoint executes the function in a new thread and wraps it within a logging trace
    ai4e_wrapper.wrap_async_endpoint(_request_detections, 'post:request_detections',
                                     request_id=request_id, post_body=post_body)

    return jsonify(request_id=request_id)


def _request_detections(**kwargs):
    try:
        body = kwargs.get('post_body')

        input_container_sas = body['input_container_sas']

        images_requested_json_sas = body.get('images_requested_json_sas', None)

        image_path_prefix = body.get('image_path_prefix', None)

        first_n = body.get('first_n', None)
        first_n = int(first_n) if first_n else None

        sample_n = body.get('sample_n', None)
        sample_n = int(sample_n) if sample_n else None

        model_version = body.get('model_version', None)
        if model_version is None or model_version == '':
            model_version = api_config.AML_CONFIG['default_model_version']
        print('runserver.py, model_version to use is {}'.format(model_version))

        # request_name and request_submission_timestamp are for appending to output file names
        request_name = body.get('request_name', '')
        request_submission_timestamp = orchestrator.get_utc_timestamp()

        request_id = kwargs['request_id']
        api_task_manager.UpdateTaskStatus(request_id, get_task_status('running', 'Request received.'))

        if images_requested_json_sas is None:
            api_task_manager.UpdateTaskStatus(request_id, get_task_status('running', 'Listing all images to process.'))
            print('runserver.py, running - listing all images to process.')

            # list all images to process
            blob_prefix = None if image_path_prefix is None else image_path_prefix
            image_paths = SasBlob.list_blobs_in_container(api_config.MAX_NUMBER_IMAGES_ACCEPTED,
                                                          sas_uri=input_container_sas,
                                                          blob_prefix=blob_prefix, blob_suffix='.jpg')
        else:
            print('runserver.py, running - using provided list of images.')
            image_paths_text = SasBlob.download_blob_to_text(images_requested_json_sas)
            image_paths = json.loads(image_paths_text)
            print('runserver.py, length of image_paths provided by the user: {}'.format(len(image_paths)))

            image_paths = [i for i in image_paths if str(i).lower().endswith(api_config.ACCEPTED_IMAGE_FILE_ENDINGS)]
            print('runserver.py, length of image_paths provided by the user, after filtering to jpg: {}'.format(
                len(image_paths)))

            if image_path_prefix is not None:
                image_paths = [i for i in image_paths if str(i).startswith(image_path_prefix)]
                print(
                    'runserver.py, length of image_paths provided by the user, after filtering for image_path_prefix: {}'.format(
                        len(image_paths)))

            res = orchestrator.spot_check_blob_paths_exist(image_paths, input_container_sas)
            if res is not None:
                raise LookupError(
                    'path {} provided in list of images to process does not exist in the container pointed to by data_container_sas.'.format(
                        res))

        # apply the first_n and sample_n filters
        if first_n is not None:
            assert first_n > 0, 'parameter first_n is 0.'
            image_paths = image_paths[:first_n]  # will not error if first_n > total number of images

        if sample_n is not None:
            assert sample_n > 0, 'parameter sample_n is 0.'
            if sample_n > len(image_paths):
                raise ValueError(
                    'parameter sample_n specifies more images than available (after filtering by other provided params).')

            # we sample by just shuffling the image paths and take the first sample_n images
            print('First path before shuffling:', image_paths[0])
            shuffle(image_paths)
            print('First path after shuffling:', image_paths[0])
            image_paths = image_paths[:sample_n]
            image_paths = sorted(image_paths)

        num_images = len(image_paths)
        print('runserver.py, num_images after applying all filters: {}'.format(num_images))
        if num_images < 1:
            api_task_manager.UpdateTaskStatus(request_id,
                                              get_task_status('completed',
                                                              'Zero images found in container or in provided list of images after filtering with the provided parameters.'))
            return
        if num_images > api_config.MAX_NUMBER_IMAGES_ACCEPTED:
            api_task_manager.UpdateTaskStatus(request_id,
                                              get_task_status('failed',
                                                              'The number of images ({}) requested for processing exceeds the maximum accepted ({}) in one call.'.format(
                                                                  num_images, api_config.MAX_NUMBER_IMAGES_ACCEPTED)))
            return

        image_paths_string = json.dumps(image_paths, indent=2)
        internal_storage_service.create_blob_from_text(internal_container,
                                                       '{}/{}_images_{}_{}.json'.format(request_id, request_id,
                                                                                        request_name,
                                                                                        request_submission_timestamp),
                                                       image_paths_string)
        api_task_manager.UpdateTaskStatus(request_id,
                                          get_task_status('running',
                                                          'Images listed; processing {} images.'.format(num_images)))
        print('runserver.py, running - images listed; processing {} images.'.format(num_images))

        # set up connection to AML Compute and data stores
        # do this for each request since pipeline step is associated with the data stores
        aml_compute = orchestrator.AMLCompute(request_id, input_container_sas, internal_datastore)
        print('AMLCompute resource connected successfully.')

        num_images_per_job = api_config.NUM_IMAGES_PER_JOB
        num_jobs = math.ceil(num_images / num_images_per_job)

        list_jobs = {}
        for job_index in range(num_jobs):
            begin, end = job_index * num_images_per_job, (job_index + 1) * num_images_per_job
            job_id = 'request{}_jobindex{}_total{}'.format(request_id, job_index, num_jobs)
            list_jobs[job_id] = {'begin': begin, 'end': end}

        list_jobs_submitted = aml_compute.submit_jobs(request_id, list_jobs, api_task_manager, num_images)
        api_task_manager.UpdateTaskStatus(request_id,
                                          get_task_status('running',
                                                          'All {} images submitted to cluster for processing.'.format(
                                                              num_images)))

    except Exception as e:
        api_task_manager.UpdateTaskStatus(request_id,
                                          get_task_status('failed',
                                                          'An error occurred while processing the request: {}'.format(
                                                              e)))
        print('runserver.py, exception in _request_detections: {}'.format(str(e)))
        return  # do not initiate _monitor_detections_request

    try:
        aml_monitor = orchestrator.AMLMonitor(request_id, list_jobs_submitted,
                                              request_name, request_submission_timestamp)

        # start another thread to monitor the jobs and consolidate the results when they finish
        ai4e_wrapper.wrap_async_endpoint(_monitor_detections_request, 'post:_monitor_detections_request',
                                         request_id=request_id,
                                         aml_monitor=aml_monitor)
    except Exception as e:
        api_task_manager.UpdateTaskStatus(request_id,
                                          get_task_status('problem', (
                                              'An error occurred when starting the status monitoring process. '
                                              'The images should be submitted for processing though - please contact us to retrieve your results. '
                                              'Error: {}'.format(e))))
        print('runserver.py, exception when starting orchestrator.AMLMonitor: ', str(e))


def _monitor_detections_request(**kwargs):
    try:
        request_id = kwargs['request_id']
        aml_monitor = kwargs['aml_monitor']

        max_num_checks = api_config.MAX_MONITOR_CYCLES
        num_checks = 0
        num_errors_job_status = 0  # number of errors encountered during aml_monitor.check_job_status()
        num_errors_aggregation = 0  # number of errors encountered during aml_monitor.aggregate_results()

        print('Monitoring thread with _monitor_detections_request started.')

        # time.sleep() blocks the current thread only
        while True:
            time.sleep(api_config.MONITOR_PERIOD_MINUTES * 60)

            print('runserver.py, _monitor_detections_request, woke up at {} for check number {}.'.format(
                datetime.now(), num_checks))

            # check the status of the jobs, with retries
            try:
                all_jobs_finished, status_tally = aml_monitor.check_job_status()
            except Exception as e:
                num_errors_job_status += 1
                print(
                    'runserver.py, _monitor_detections_request, exception in aml_monitor.check_job_status(): {}'.format(
                        e))

                if num_errors_job_status <= api_config.NUM_RETRIES:
                    print('Will retry in the next monitoring cycle. Number of errors so far: {}'.format(
                        num_errors_job_status))
                    continue
                else:
                    print('Number of retries reached for aml_monitor.check_job_status().')
                    raise e

            print('all jobs finished? {}'.format(all_jobs_finished))
            for status, count in status_tally.items():
                print('status {}, number of jobs = {}'.format(status, count))

            num_failed = status_tally['Failed']
            # need to periodically check the enumerations are what AML returns - not the same as in their doc
            num_finished = status_tally['Finished'] + num_failed

            # if all jobs finished, aggregate the results and return the URLs to the output files
            if all_jobs_finished:
                api_task_manager.UpdateTaskStatus(request_id,
                                                  get_task_status('almost completed',
                                                                  'Model inference finished; now aggregating results.'))

                # retrieve and join the output CSVs from each job, with retries
                try:
                    output_file_urls = aml_monitor.aggregate_results()
                except Exception as e:
                    num_errors_aggregation += 1
                    print(('runserver.py, _monitor_detections_request, exception in '
                           'aml_monitor.aggregate_results(): {}').format(e))

                    if num_errors_aggregation <= api_config.NUM_RETRIES:
                        print('Will retry during the next monitoring wake-up cycle. Number of errors so far: {}'.format(
                            num_errors_aggregation))
                        api_task_manager.UpdateTaskStatus(request_id,
                                                          get_task_status('almost completed',
                                                                          'All shards finished but results aggregation failed. Will retry in {} minutes.'.format(
                                                                              api_config.MONITOR_PERIOD_MINUTES)))
                        continue
                    else:
                        print('Number of retries reached for aml_monitor.aggregate_results().')
                        raise e

                output_file_urls_str = json.dumps(output_file_urls)
                api_task_manager.UpdateTaskStatus(request_id, get_task_status('completed',
                                                                              'Completed at {}. Number of failed shards: {}. URLs to output files: {}'.format(
                                                                                  orchestrator.get_utc_time(),
                                                                                  num_failed, output_file_urls_str)))
                break

            # not all jobs are finished, update the status with number of shards finished
            api_task_manager.UpdateTaskStatus(request_id, get_task_status('running',
                                                                          'Last status check at {}, {} out of {} shards finished processing, {} failed.'.format(
                                                                              orchestrator.get_utc_time(), num_finished,
                                                                              aml_monitor.get_total_jobs(),
                                                                              num_failed)))

            # not all jobs are finished but the maximum number of checking cycle is reached, stop this thread
            num_checks += 1
            if num_checks >= max_num_checks:
                api_task_manager.UpdateTaskStatus(request_id, get_task_status('problem',
                                                                              'Request unfinished after {} x {} minutes; abandoning the monitoring thread. Please contact us to retrieve any results.'.format(
                                                                                  api_config.MAX_MONITOR_CYCLES,
                                                                                  api_config.MONITOR_PERIOD_MINUTES
                                                                              )))
                print('runserver.py, _monitor_detections_request, ending!')

                break
    except Exception as e:
        api_task_manager.UpdateTaskStatus(request_id, get_task_status('problem', (
            'An error occurred while monitoring the status of this request. '
            'The images should be processing though - please contact us to retrieve your results. Error: {}'.format(
                e))))
        print('runserver.py, exception in _monitor_detections_request(): ', str(e))


@app.route(api_prefix + '/default_model_version', methods=['GET'])
def default_model_version():
    # wrap_sync_endpoint wraps your function within a logging trace.
    return ai4e_wrapper.wrap_sync_endpoint(_default_model_version, 'get:default_model_version')

def _default_model_version():
    return api_config.AML_CONFIG['default_model_version']


@app.route(api_prefix + '/supported_model_versions', methods=['GET'])
def supported_model_versions():
    # wrap_sync_endpoint wraps your function within a logging trace.
    return ai4e_wrapper.wrap_sync_endpoint(_supported_model_versions, 'get:supported_model_versions')

def _supported_model_versions():
    return json.dumps(api_config.SUPPORTED_MODEL_VERSIONS)


if __name__ == '__main__':
    app.run()

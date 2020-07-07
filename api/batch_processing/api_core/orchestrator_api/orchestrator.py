import copy
import io
import os
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
import json

import azureml.core
from azure.storage.blob import BlockBlobService, BlobPermissions
from azureml.core import Workspace, Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.datastore import Datastore
from azureml.core.runconfig import RunConfiguration, DEFAULT_GPU_IMAGE
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

import api_config
from sas_blob_utils import SasBlob

print('Version of AML: {}'.format(azureml.core.__version__))

# Service principle authentication for AML
svc_pr_password = os.environ.get('AZUREML_PASSWORD')
svc_pr = ServicePrincipalAuthentication(
    api_config.AML_CONFIG['tenant-id'],  # 'my-tenant-id'
    api_config.AML_CONFIG['application-id'],  # 'my-application-id'
    svc_pr_password)


# %% Utility functions

def get_utc_time():
    # return the current UTC time in string format '2019-05-19 08:57:43'
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


def get_utc_timestamp():
    # return current UTC time in succinct format as a string, e.g. '20190519085759'
    return datetime.utcnow().strftime("%Y%m%d%H%M%S")


def get_task_status(request_status, message):
    return {
        'request_status': request_status,
        'time': get_utc_time(),
        'message': message
    }

def check_data_container_sas(sas_uri):
    permissions = SasBlob.get_permissions_from_uri(sas_uri)
    if 'read' not in permissions or 'list' not in permissions:
        return (400, 'input_container_sas provided does not have both read and list permissions.')
    return None


def spot_check_blob_paths_exist(paths, container_sas, metadata_available):
    ''' Check that the first blob in paths exists in the container specified in container_sas.

    Args:
        paths: A list of blob paths in the container
        container_sas: Azure blob storage SAS token to a container where the paths are supposed to be in
        metadata_available: paths will contain items that are [image_id, metadata] instead of just image_id
    Returns:
        None if no problem found. Return the image_path if it does not exist in the container specified
    '''
    if len(paths) == 0:  # redundant check...
        return None

    path = paths[0][0] if metadata_available else paths[0]
    if SasBlob.check_blob_exists_in_container(path, container_sas_uri=container_sas):
        return None
    return path


def validate_provided_image_paths(image_paths):
    ''' Given a list of image_paths (list length at least 1), validate them and determine if metadata is available.

    Args:
        image_paths: a list of string (image_id) or a list of 2-item lists ([image_id, image_metadata])

    Returns:
        error: None if checks passed; a string message indicating the problem otherwise
        metadata_available: bool, True if available
    '''
    # image_paths will have length at least 1, otherwise would have ended before this step
    first_item = image_paths[0]
    metadata_available = False
    if isinstance(first_item, str):
        for i in image_paths:
            if not isinstance(i, str):
                return 'Not all items in image_paths supplied is of type string.', metadata_available
        return None, metadata_available
    elif isinstance(first_item, list):
        metadata_available = True
        for i in image_paths:
            if len(i) != 2:  # i should be [image_id, metadata_string]
                return 'Items in image_paths are lists, but not all lists are of length 2 [image locator, metadata].', metadata_available
        return None, metadata_available
    else:
        return 'image_paths contain items that are not strings nor lists.', metadata_available


def sort_image_paths(image_paths, metadata_available):
    if len(image_paths) == 0:  # redundant check...
        return image_paths

    if metadata_available:
        return sorted(image_paths, key=lambda x: x[0])
    else:
        return sorted(image_paths)


# %% AML Compute

class AMLCompute:
    def __init__(self, request_id, use_url, input_container_sas, internal_datastore, model_name):
        try:
            self.request_id = request_id

            aml_config = api_config.AML_CONFIG

            self.ws = Workspace(
                subscription_id=aml_config['subscription_id'],
                resource_group=aml_config['resource_group'],
                workspace_name=aml_config['workspace_name'],
                auth=svc_pr
            )
            print('AMLCompute constructor, AML workspace obtained.')

            internal_dir, output_dir = self._get_data_references(request_id, internal_datastore)

            compute_target = self.ws.compute_targets[aml_config['aml_compute_name']]

            dependencies = CondaDependencies.create(pip_packages=['tensorflow-gpu==1.12.0',
                                                                  'pillow',
                                                                  'numpy',
                                                                  'azure-storage-blob==2.1.0',
                                                                  'azureml-defaults==1.0.41'])

            amlcompute_run_config = RunConfiguration(conda_dependencies=dependencies)
            amlcompute_run_config.environment.docker.enabled = True
            amlcompute_run_config.environment.docker.gpu_support = True
            amlcompute_run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
            amlcompute_run_config.environment.spark.precache_packages = False

            # default values are required and need to be literal values or data references as JSON
            param_job_id = PipelineParameter(name='param_job_id', default_value='default_job_id')

            param_begin_index = PipelineParameter(name='param_begin_index', default_value=0)
            param_end_index = PipelineParameter(name='param_end_index', default_value=0)

            param_detection_threshold = PipelineParameter(name='param_detection_threshold', default_value=0.05)

            batch_score_step = PythonScriptStep(aml_config['script_name'],
                                                source_directory=aml_config['source_dir'],
                                                hash_paths=['.'],  # include all contents of source_directory
                                                name='batch_scoring',
                                                arguments=['--job_id', param_job_id,
                                                           '--model_name', model_name,
                                                           '--input_container_sas', input_container_sas,  # can be None
                                                           '--use_url', use_url,
                                                           '--internal_dir', internal_dir,
                                                           '--begin_index', param_begin_index,  # inclusive
                                                           '--end_index', param_end_index,  # exclusive
                                                           '--output_dir', output_dir,
                                                           '--detection_threshold', param_detection_threshold],
                                                compute_target=compute_target,
                                                inputs=[internal_dir],
                                                outputs=[output_dir],
                                                runconfig=amlcompute_run_config
                                                )
            self.pipeline = Pipeline(workspace=self.ws, steps=[batch_score_step])
            self.aml_config = aml_config
            print('AMLCompute constructor all good.')
        except Exception as e:
            raise RuntimeError('Error in setting up AML Compute resource: {}.'.format(str(e)))

    def _get_data_references(self, request_id, internal_datastore):
        print('AMLCompute, _get_data_references() called. Request ID: {}'.format(request_id))
        try:
            # setting the overwrite flag to True overwrites any datastore that was created previously with that name

            # internal_datastore stores all user-facing files: list of images, detection results, list of failed images
            # and it so happens that each job also needs the list of images as an input
            internal_datastore_name = 'internal_datastore_{}'.format(request_id)
            internal_account_name = internal_datastore['account_name']
            internal_account_key = internal_datastore['account_key']
            internal_container_name = internal_datastore['container_name']
            internal_datastore = Datastore.register_azure_blob_container(self.ws, internal_datastore_name,
                                                                         internal_container_name,
                                                                         internal_account_name,
                                                                         account_key=internal_account_key)
            print('internal_datastore done')

            # output_datastore stores the output from score.py in each job, which is another container
            # in the same storage account as interl_datastore
            output_datastore_name = 'output_datastore_{}'.format(request_id)
            output_container_name = api_config.AML_CONTAINER
            output_datastore = Datastore.register_azure_blob_container(self.ws, output_datastore_name,
                                                                       output_container_name,
                                                                       internal_account_name,
                                                                       account_key=internal_account_key)
            print('output_datastore done')

        except Exception as e:
            raise RuntimeError('Error in connecting to the datastores for AML Compute: {}'.format(str(e)))

        try:
            internal_dir = DataReference(datastore=internal_datastore,
                                         data_reference_name='internal_dir',
                                         mode='mount')

            output_dir = PipelineData('output_{}'.format(request_id),
                                      datastore=output_datastore,
                                      output_mode='mount')
            print('Finished setting up the Data References.')
        except Exception as e:
            raise RuntimeError('Error in creating data references for AML Compute: {}.'.format(str(e)))

        return internal_dir, output_dir

    def submit_jobs(self, list_jobs, api_task_manager, num_images):
        try:
            print('AMLCompute, submit_jobs() called.')
            list_jobs_active = copy.deepcopy(list_jobs)

            for i, (job_id, job) in enumerate(list_jobs.items()):
                pipeline_run = Experiment(self.ws, job_id).submit(self.pipeline, pipeline_params={
                    'param_job_id': job_id,
                    'param_begin_index': job['begin'],
                    'param_end_index': job['end'],
                    'param_detection_threshold': self.aml_config['param_detection_threshold'],
                })
                list_jobs_active[job_id]['pipeline_run'] = pipeline_run

                # various attempts at getting the child_run's ID
                # child_run_id = None
                # print('pipeline_run:', pipeline_run)
                # for child_run in pipeline_run.get_children():
                #     child_run_id = child_run.id  # we can do this because there's only one step in the pipeline - not working
                #
                # print('=' * 20)
                # exp = Experiment(self.ws, job_id)
                # run = Run(exp, pipeline_run.id)
                # print('run:', run)
                # for c in run.get_children():
                #     print('found run.get_children:')
                #     print(c.id)
                #     child_run_id = c.id
                # print('=' * 20)

                # list_jobs_active[job_id]['step_run_id']  = child_run_id  # this is the ID we can identify the output folder with

                print('Submitted job {}.'.format(job_id))

                if i % api_config.JOB_SUBMISSION_UPDATE_INTERVAL == 0:
                    api_task_manager.UpdateTaskStatus(self.request_id, get_task_status('running',
                                                                                  '{} images out of total {} submitted for processing.'.format(
                                                                                      i * api_config.NUM_IMAGES_PER_JOB,
                                                                                      num_images)))
            print('AMLCompute, submit_jobs() finished.')
            return list_jobs_active
        except Exception as e:
            raise RuntimeError('Error in submitting jobs to AML Compute cluster: {}.'.format(str(e)))


# %% AML Monitor

class AMLMonitor:
    def __init__(self, request_id, list_jobs_submitted, request_name, request_submission_timestamp, model_version):
        self.request_id = request_id
        self.jobs_submitted = list_jobs_submitted
        self.request_name = request_name  # None if not provided by the user
        self.request_submission_timestamp = request_submission_timestamp  # str
        self.model_version = model_version  # str

        storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
        storage_account_key = os.getenv('STORAGE_ACCOUNT_KEY')
        self.internal_storage_service = BlockBlobService(account_name=storage_account_name,
                                                         account_key=storage_account_key)
        self.internal_datastore = {
            'account_name': storage_account_name,
            'account_key': storage_account_key,
            'container_name': api_config.INTERNAL_CONTAINER
        }
        self.aml_output_container = api_config.AML_CONTAINER
        self.internal_container = api_config.INTERNAL_CONTAINER

    def get_total_jobs(self):
        return len(self.jobs_submitted)

    def check_job_status(self):
        print('AMLMonitor, check_job_status() called.')
        all_jobs_finished = True
        status_tally = defaultdict(int)

        for job_id, job in self.jobs_submitted.items():
            pipeline_run = job['pipeline_run']
            status = pipeline_run.get_status()  # common values returned include Running, Completed, and Failed - March 19 apparently Finished is the enumeration

            print('request_id {}, job_id {}, status is {}'.format(self.request_id, job_id, status))
            status_tally[status] += 1

            if status not in api_config.AML_CONFIG['completed_status']:  # else all_job_finished will not be flipped
                all_jobs_finished = False

        return all_jobs_finished, status_tally

    def _download_read_json(self, blob_path):
        blob = self.internal_storage_service.get_blob_to_text(self.aml_output_container, blob_path)
        stream = io.StringIO(blob.content)
        result = json.load(stream)
        return result

    def _generate_urls_for_outputs(self):
        try:
            request_id = self.request_id
            request_name, request_submission_timestamp = self.request_name, self.request_submission_timestamp

            blob_paths = {
                'detections': '{}/{}_detections_{}_{}.json'.format(request_id, request_id,
                                                                   request_name, request_submission_timestamp),
                'failed_images': '{}/{}_failed_images_{}_{}.json'.format(request_id, request_id,
                                                                         request_name, request_submission_timestamp),
                # list of images do not have request_name and timestamp in the file name so score.py can locate it easily
                'images': '{}/{}_images.json'.format(request_id, request_id)
            }

            expiry = datetime.utcnow() + timedelta(days=api_config.EXPIRATION_DAYS)

            urls = {}
            for output, blob_path in blob_paths.items():
                sas = self.internal_storage_service.generate_blob_shared_access_signature(
                    self.internal_container, blob_path, permission=BlobPermissions.READ, expiry=expiry
                )
                url = self.internal_storage_service.make_blob_url(self.internal_container, blob_path, sas_token=sas)
                urls[output] = url
            return urls
        except Exception as e:
            raise RuntimeError('An error occurred while generating URLs for the output files. ' +
                               'Please contact us to retrieve your results. ' +
                               'Error: {}'.format(str(e)))

    def aggregate_results(self):
        print('AMLMonitor, aggregate_results() called')

        # The more efficient method is to know the run_id which is the folder name that the result is written to.
        # Since we can't reliably get the run_id after submitting the run, resort to listing all blobs in the output
        # container and match by the request_id

        # listing all (up to a large limit) because don't want to worry about generator next_marker
        datastore_aml_container = copy.deepcopy(self.internal_datastore)
        datastore_aml_container['container_name'] = self.aml_output_container
        list_blobs = SasBlob.list_blobs_in_container(api_config.MAX_BLOBS_IN_OUTPUT_CONTAINER,
                                                     datastore=datastore_aml_container,
                                                     blob_suffix='.json')
        all_detections = []
        failures = []
        num_aggregated = 0
        for blob_path in list_blobs:
            if blob_path.endswith('.json'):
                # blob_path is azureml/run_id/output_requestID/out_file_name.json
                out_file_name = blob_path.split('/')[-1]
                # "request" is part of the AML job_id
                if out_file_name.startswith('detections_request{}_'.format(self.request_id)):
                    all_detections.extend(self._download_read_json(blob_path))
                    num_aggregated += 1
                    print('Number of results aggregated: ', num_aggregated)
                elif out_file_name.startswith('failures_request{}_'.format(self.request_id)):
                    failures.extend(self._download_read_json(blob_path))

        print('aggregate_results(), length of all_detections: {}'.format(len(all_detections)))

        detection_output_content = {
            'info': {
                'detector': 'megadetector_v{}'.format(self.model_version),
                'detection_completion_time': get_utc_time(),
                'format_version': api_config.OUTPUT_FORMAT_VERSION
            },
            'detection_categories': api_config.DETECTION_CATEGORIES,
            'images': all_detections
        }
        # order the json output keys
        detection_output_content = OrderedDict([
                                                ('info', detection_output_content['info']),
                                                ('detection_categories', detection_output_content['detection_categories']),
                                                ('images', detection_output_content['images'])])

        detection_output_str = json.dumps(detection_output_content, indent=1)

        # upload aggregated results to output_store
        self.internal_storage_service.create_blob_from_text(self.internal_container,
                                                            '{}/{}_detections_{}_{}.json'.format(
                                                                self.request_id, self.request_id,
                                                                self.request_name, self.request_submission_timestamp),
                                                            detection_output_str, max_connections=4)
        print('aggregate_results(), detections uploaded')

        print('aggregate_results(), number of failed images: {}'.format(len(failures)))
        failures_str = json.dumps(failures, indent=1)
        self.internal_storage_service.create_blob_from_text(self.internal_container,
                                                            '{}/{}_failed_images_{}_{}.json'.format(
                                                                self.request_id, self.request_id,
                                                                self.request_name, self.request_submission_timestamp),
                                                            failures_str)
        print('aggregate_results(), failures uploaded')

        output_file_urls = self._generate_urls_for_outputs()
        return output_file_urls

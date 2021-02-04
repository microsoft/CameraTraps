"""
Functions to submit images to the Azure Batch node pool for processing, monitor
the Job and fetch results when completed.
"""

import io
import json
import time
import urllib.parse
from datetime import timedelta
from random import shuffle

import sas_blob_utils  # from ai4eutils
from azure.storage.blob import ContainerClient, BlobSasPermissions, generate_blob_sas
from tqdm import tqdm

from server_utils import *
import server_api_config as api_config
from server_batch import BatchJobManager
from server_job_status_table import JobStatusTable


def create_batch_job(job_id: str, body: dict):
    """
    This is the target to be run in a thread to submit a batch processing job and monitor progress
    """
    job_status_table = JobStatusTable()
    try:
        print(f'server_utils, create_batch_job, job_id {job_id}, {body}')

        input_container_sas = body.get('input_container_sas', None)

        use_url = body.get('use_url', False)

        images_requested_json_sas = body.get('images_requested_json_sas', None)

        image_path_prefix = body.get('image_path_prefix', None)

        first_n = body.get('first_n', None)
        first_n = int(first_n) if first_n else None

        sample_n = body.get('sample_n', None)
        sample_n = int(sample_n) if sample_n else None

        model_version = body.get('model_version', '')
        if model_version == '':
            model_version = api_config.DEFAULT_MD_VERSION

        # request_name and request_submission_timestamp are for appending to
        # output file names
        job_name = body.get('request_name', '')  # in earlier versions we used "request" to mean a "job"
        job_submission_timestamp = get_utc_time()

        # image_paths can be a list of strings (Azure blob names or public URLs)
        # or a list of length-2 lists where each is a [image_id, metadata] pair
        job_status = get_job_status('running', 'Listing all images to process.')
        job_status_table.update_job_status(job_id, job_status)

        # Case 1: listing all images in the container
        # - not possible to have attached metadata if listing images in a blob
        if images_requested_json_sas is None:
            print('server_utils, create_batch_job, running - listing all images to process.')

            # list all images to process
            image_paths = sas_blob_utils.list_blobs_in_container(
                container_uri=input_container_sas,
                blob_prefix=image_path_prefix,
                blob_suffix=api_config.IMAGE_SUFFIXES_ACCEPTED,
                limit=api_config.MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB + 1
                # + 1 so if the number of images listed > MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB
                # we will know and not proceed
            )

        # Case 2: user supplied a list of images to process; can include metadata
        else:
            print('server_utils, create_batch_job, running - using provided list of images.')
            output_stream, blob_properties = sas_blob_utils.download_blob_to_stream(images_requested_json_sas)
            image_paths = json.load(output_stream)
            print('server_utils, create_batch_job, length of image_paths provided by the user: {}'.format(
                len(image_paths)))
            if len(image_paths) == 0:
                job_status = get_job_status(
                    'completed', '0 images found in provided list of images.')
                job_status_table.update_job_status(job_id, job_status)
                return

            error, metadata_available = validate_provided_image_paths(image_paths)
            if error is not None:
                msg = 'image paths provided in the json are not valid: {}'.format(error)
                raise ValueError(msg)

            # filter down to those conforming to the provided prefix and accepted suffixes (image file types)
            valid_image_paths = []
            for p in image_paths:
                locator = p[0] if metadata_available else p

                # prefix is case-sensitive; suffix is not
                if image_path_prefix is not None and not locator.startswith(image_path_prefix):
                    continue

                # Although urlparse(p).path preserves the extension on local paths, it will not work for
                # blob file names that contains "#", which will be treated as indication of a query.
                # If the URL is generated via Azure Blob Storage, the "#" char will be properly encoded
                path = urllib.parse.urlparse(locator).path.lower() if use_url else locator

                if path.lower().endswith(api_config.IMAGE_SUFFIXES_ACCEPTED):
                    valid_image_paths.append(p)
            image_paths = valid_image_paths
            print_job(job_id, ('server_utils, create_batch_job, length of image_paths provided by user, '
                               f'after filtering to jpg: {len(image_paths)}'))

        # apply the first_n and sample_n filters
        if first_n:
            assert first_n > 0, 'parameter first_n is 0.'
            # OK if first_n > total number of images
            image_paths = image_paths[:first_n]

        if sample_n:
            assert sample_n > 0, 'parameter sample_n is 0.'
            if sample_n > len(image_paths):
                msg = ('parameter sample_n specifies more images than '
                       'available (after filtering by other provided params).')
                raise ValueError(msg)

            # sample by shuffling image paths and take the first sample_n images
            print('First path before shuffling:', image_paths[0])
            shuffle(image_paths)
            print('First path after shuffling:', image_paths[0])
            image_paths = image_paths[:sample_n]

        num_images = len(image_paths)
        print(f'server_utils, create_batch_job, num_images after applying all filters: {num_images}')

        if num_images < 1:
            job_status = get_job_status('completed', (
                'Zero images found in container or in provided list of images '
                'after filtering with the provided parameters.'))
            job_status_table.update_job_status(job_id, job_status)
            return
        if num_images > api_config.MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB:
            job_status = get_job_status(
                'failed',
                (f'The number of images ({num_images}) requested for processing exceeds the maximum '
                 f'accepted {api_config.MAX_NUMBER_IMAGES_ACCEPTED_PER_JOB} in one call'))
            job_status_table.update_job_status(job_id, job_status)
            return

        # upload the image list to the container, which is also mounted on all nodes
        # all sharding and scoring use the uploaded list
        images_list_str_as_bytes = bytes(json.dumps(image_paths), 'utf-8')

        container_url = sas_blob_utils.build_azure_storage_uri(account=api_config.STORAGE_ACCOUNT_NAME,
                                                               container=api_config.STORAGE_CONTAINER_API)
        with ContainerClient.from_container_url(container_url,
                                                credential=api_config.STORAGE_ACCOUNT_KEY) as api_container_client:
            _ = api_container_client.upload_blob(
                name=f'api_{api_config.API_INSTANCE_NAME}/job_{job_id}/{job_id}_images.json',
                data=images_list_str_as_bytes)

        job_status = get_job_status('running', f'{num_images} images listed; submitting the job...')
        job_status_table.update_job_status(job_id, job_status)

    except Exception as e:
        job_status = get_job_status('failed', f'Error occurred while preparing the Batch job: {e}')
        job_status_table.update_job_status(job_id, job_status)
        print(f'server_utils, create_batch_job, Error occurred while preparing the Batch job: {e}')
        return  # do not start monitoring

    try:
        batch_job_manager = BatchJobManager()

        model_rel_path = api_config.MD_VERSIONS_TO_REL_PATH[model_version]
        batch_job_manager.create_job(job_id,
                                     model_rel_path,
                                     input_container_sas,
                                     use_url)

        num_tasks, task_ids_failed_to_submit = batch_job_manager.submit_tasks(job_id, num_images)

        job_status = get_job_status('running',
                                    (f'Submitted {num_images} images to cluster in {num_tasks} shards. '
                                     f'Number of shards failed to be submitted: {len(task_ids_failed_to_submit)}'))
        job_status_table.update_job_status(job_id, job_status)
    except Exception as e:
        job_status = get_job_status('problem', f'Error occurred while submitting the Batch job: {e}')
        job_status_table.update_job_status(job_id, job_status)
        print(f'server_utils, create_batch_job, Error occurred while submitting the Batch job: {e}')
        return

    try:
        num_checks = 0

        while True:
            time.sleep(api_config.MONITOR_PERIOD_MINUTES * 60)
            num_checks += 1

            # a completed Task could have a non-zero error code TODO check how many failed
            num_tasks_completed = batch_job_manager.get_num_completed_tasks(job_id)
            job_status = get_job_status('running',
                                        (f'Check number {num_checks}, '
                                         f'{num_tasks_completed} out of {num_tasks} shards have completed'))
            job_status_table.update_job_status(job_id, job_status)
            print(f'Check number {num_checks}, {num_tasks_completed} out of {num_tasks} shards have completed')

            if num_tasks_completed >= num_tasks:
                break

            if num_checks > api_config.MAX_MONITOR_CYCLES:
                job_status = get_job_status('problem',
                    (
                        f'Job unfinished after {num_checks} x {api_config.MONITOR_PERIOD_MINUTES} minutes, '
                        f'please contact us to retrieve the results. Number of completed tasks: {num_tasks_completed}')
                    )
                job_status_table.update_job_status(job_id, job_status)
                print(f'server_utils, create_batch_job, MAX_MONITOR_CYCLES reached, ending thread')
                break  # still aggregate the Tasks' outputs

    except Exception as e:
        job_status = get_job_status('problem', f'Error occurred while monitoring the Batch job: {e}')
        job_status_table.update_job_status(job_id, job_status)
        print(f'server_utils, create_batch_job, Error occurred while monitoring the Batch job: {e}')
        return

    try:
        output_sas_url = aggregate_results(job_id, model_version, job_name, job_submission_timestamp)
        # preserving format from before, but SAS URL to 'failed_images' and 'images' are no longer provided
        # failures should be contained in the output entries, indicated by an 'error' field
        msg = {
            'output_file_urls': {
                'detections': output_sas_url
            }
        }
        job_status = get_job_status('completed', msg)
        job_status_table.update_job_status(job_id, job_status)

    except Exception as e:
        job_status = get_job_status('problem',
                        f'Please contact us to retrieve the results. Error occurred while aggregating results: {e}')
        job_status_table.update_job_status(job_id, job_status)
        print(f'server_utils, create_batch_job, Error occurred while aggregating results: {e}')
        return


def aggregate_results(job_id, model_version, job_name, job_submission_timestamp):
    task_outputs_dir = f'api_{api_config.API_INSTANCE_NAME}/job_{job_id}/task_outputs/'

    container_url = sas_blob_utils.build_azure_storage_uri(account=api_config.STORAGE_ACCOUNT_NAME,
                                                           container=api_config.STORAGE_CONTAINER_API)

    all_results = []

    with ContainerClient.from_container_url(container_url,
                                            credential=api_config.STORAGE_ACCOUNT_KEY) as container_client:
        generator = container_client.list_blobs(name_starts_with=task_outputs_dir)

        blobs = [i for i in generator if i.name.endswith('.json')]

        for blob_props in tqdm(blobs):
            with container_client.get_blob_client(blob_props) as blob_client:
                stream = io.BytesIO()
                blob_client.download_blob().readinto(stream)
                stream.seek(0)
                task_results = json.load(stream)
                all_results.extend(task_results)

        api_output = {
            'info': {
                'detector': f'megadetector_v{model_version}',
                'detection_completion_time': get_utc_time(),
                'format_version': api_config.OUTPUT_FORMAT_VERSION
            },
            'detection_categories': api_config.DETECTOR_LABEL_MAP,
            'images': all_results
        }

        # upload the output JSON to the Job folder
        api_output_as_bytes = bytes(json.dumps(api_output, indent=1), 'utf-8')
        output_file_path = f'api_{api_config.API_INSTANCE_NAME}/job_{job_id}/{job_id}_detections_{job_name}_{job_submission_timestamp}.json'
        _ = container_client.upload_blob(name=output_file_path, data=api_output_as_bytes)

    output_sas = generate_blob_sas(
        account_name=api_config.STORAGE_ACCOUNT_NAME,
        container_name=api_config.STORAGE_CONTAINER_API,
        blob_name=output_file_path,
        account_key=api_config.STORAGE_ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True, write=False),
        expiry=datetime.utcnow() + timedelta(days=api_config.OUTPUT_SAS_EXPIRATION_DAYS)
    )
    output_sas_url = sas_blob_utils.build_azure_storage_uri(
        account=api_config.STORAGE_ACCOUNT_NAME,
        container=api_config.STORAGE_CONTAINER_API,
        blob=output_file_path,
        sas_token=output_sas
    )
    print(f'aggregate_results done, job_id: {job_id}')
    print(f'output_sas_url: {output_sas_url}')
    return output_sas_url

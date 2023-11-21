# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
A class wrapping the Azure Batch client.
"""

import logging
import os
import math
from typing import Tuple
from datetime import datetime, timedelta

import sas_blob_utils  # from ai4eutils
from azure.storage.blob import ContainerClient, ContainerSasPermissions, generate_container_sas
from azure.batch import BatchServiceClient
from azure.batch.models import *
from azure.common.credentials import ServicePrincipalCredentials

import server_api_config as api_config


# Gunicorn logger handler will get attached if needed in server.py
log = logging.getLogger(os.environ['FLASK_APP'])


class BatchJobManager:
    """Wrapper around the Azure App Configuration client"""

    def __init__(self):
        credentials = ServicePrincipalCredentials(
            client_id=api_config.APP_CLIENT_ID,
            secret=api_config.APP_CLIENT_SECRET,
            tenant=api_config.APP_TENANT_ID,
            resource='https://batch.core.windows.net/'
        )
        self.batch_client = BatchServiceClient(credentials=credentials,
                                               batch_url=api_config.BATCH_ACCOUNT_URL)

    def create_job(self, job_id: str, detector_model_rel_path: str,
                   input_container_sas: str, use_url: bool):
        log.info(f'BatchJobManager, create_job, job_id: {job_id}')
        job = JobAddParameter(
            id=job_id,
            pool_info=PoolInformation(pool_id=api_config.POOL_ID),

            # set for all tasks in the job
            common_environment_settings=[
                EnvironmentSetting(name='DETECTOR_REL_PATH', value=detector_model_rel_path),
                EnvironmentSetting(name='API_INSTANCE_NAME', value=api_config.API_INSTANCE_NAME),
                EnvironmentSetting(name='JOB_CONTAINER_SAS', value=input_container_sas),
                EnvironmentSetting(name='JOB_USE_URL', value=str(use_url)),
                EnvironmentSetting(name='DETECTION_CONF_THRESHOLD', value=api_config.DETECTION_CONF_THRESHOLD)
            ]
        )
        self.batch_client.job.add(job)

    def submit_tasks(self, job_id: str, num_images: int) -> Tuple[int, list]:
        """
        Shard the images and submit each shard as a Task under the Job pointed to by this job_id
        Args:
            job_id: ID of the Batch Job to submit the tasks to
            num_images: total number of images to be processed in this Job

        Returns:
            num_task: total number of Tasks that should be in this Job
            task_ids_failed_to_submit: which Tasks from the above failed to be submitted
        """
        log.info('BatchJobManager, submit_tasks')

        # cannot execute the scoring script that is in the mounted directory; has to be copied to cwd
        # not luck giving the commandline arguments via formatted string - set as env vars instead
        score_command = '/bin/bash -c \"cp $AZ_BATCH_NODE_MOUNTS_DIR/batch-api/scripts/score.py . && python score.py\" '

        num_images_per_task = api_config.NUM_IMAGES_PER_TASK

        # form shards of images and assign each shard to a Task
        num_tasks = math.ceil(num_images / num_images_per_task)

        # for persisting stdout and stderr
        permissions = ContainerSasPermissions(read=True, write=True, list=True)
        access_duration_hrs = api_config.MONITOR_PERIOD_MINUTES * api_config.MAX_MONITOR_CYCLES / 60
        container_sas_token = generate_container_sas(
            account_name=api_config.STORAGE_ACCOUNT_NAME,
            container_name=api_config.STORAGE_CONTAINER_API,
            account_key=api_config.STORAGE_ACCOUNT_KEY,
            permission=permissions,
            expiry=datetime.utcnow() + timedelta(hours=access_duration_hrs))
        container_sas_url = sas_blob_utils.build_azure_storage_uri(
            account=api_config.STORAGE_ACCOUNT_NAME,
            container=api_config.STORAGE_CONTAINER_API,
            sas_token=container_sas_token)

        tasks = []
        for task_id in range(num_tasks):
            begin_index = task_id * num_images_per_task
            end_index = begin_index + num_images_per_task

            # persist stdout and stderr (will be removed when node removed)
            # paths are relative to the Task working directory
            stderr_destination = OutputFileDestination(
                container=OutputFileBlobContainerDestination(
                    container_url=container_sas_url,
                    path=f'api_{api_config.API_INSTANCE_NAME}/job_{job_id}/task_logs/job_{job_id}_task_{task_id}_stderr.txt'
                )
            )
            stdout_destination = OutputFileDestination(
                container=OutputFileBlobContainerDestination(
                    container_url=container_sas_url,
                    path=f'api_{api_config.API_INSTANCE_NAME}/job_{job_id}/task_logs/job_{job_id}_task_{task_id}_stdout.txt'
                )
            )
            std_err_and_out = [
                OutputFile(
                    file_pattern='../stderr.txt',  # stderr.txt is at the same level as wd
                    destination=stderr_destination,
                    upload_options=OutputFileUploadOptions(upload_condition=OutputFileUploadCondition.task_completion)
                    # can also just upload on failure
                ),
                OutputFile(
                    file_pattern='../stdout.txt',
                    destination=stdout_destination,
                    upload_options=OutputFileUploadOptions(upload_condition=OutputFileUploadCondition.task_completion)
                )
            ]

            task = TaskAddParameter(
                id=str(task_id),
                command_line=score_command,
                container_settings=TaskContainerSettings(
                    image_name=api_config.CONTAINER_IMAGE_NAME,
                    working_directory='taskWorkingDirectory'
                ),
                environment_settings=[
                    EnvironmentSetting(name='TASK_BEGIN_INDEX', value=begin_index),
                    EnvironmentSetting(name='TASK_END_INDEX', value=end_index),
                ],
                output_files=std_err_and_out
            )
            tasks.append(task)

        # first try submitting Tasks
        task_ids_failed_to_submit = self._create_tasks(job_id, tasks, api_config.NUM_TASKS_PER_SUBMISSION, 1)

        # retry submitting Tasks
        if len(task_ids_failed_to_submit) > 0:
            task_ids_failed_to_submit_set = set(task_ids_failed_to_submit)
            tasks_to_retry = [t for t in tasks if t.id in task_ids_failed_to_submit_set]
            task_ids_failed_to_submit = self._create_tasks(job_id,
                                                           tasks_to_retry,
                                                           api_config.NUM_TASKS_PER_RESUBMISSION,
                                                           2)

            if len(task_ids_failed_to_submit) > 0:
                log.info('BatchJobManager, submit_tasks, after retry, '
                      f'len of task_ids_failed_to_submit: {len(task_ids_failed_to_submit)}')
            else:
                log.info('BatchJobManager, submit_tasks, after retry, all Tasks submitted')
        else:
            log.info('BatchJobManager, submit_tasks, all Tasks submitted after first try')

        # Change the Job's on_all_tasks_complete option to 'terminateJob' so the Job's status changes automatically
        # after all submitted tasks are done
        # This is so that we do not take up the quota for active Jobs in the Batch account.
        job_patch_params = JobPatchParameter(
            on_all_tasks_complete=OnAllTasksComplete.terminate_job
        )
        self.batch_client.job.patch(job_id, job_patch_params)

        return num_tasks, task_ids_failed_to_submit

    def _create_tasks(self, job_id, tasks, num_tasks_per_submission, n_th_try) -> list:
        task_ids_failed_to_submit = []

        for i in range(0, len(tasks), num_tasks_per_submission):
            tasks_to_submit = tasks[i: i + num_tasks_per_submission]

            # return type: TaskAddCollectionResult
            collection_results = self.batch_client.task.add_collection(job_id, tasks_to_submit, threads=10)

            for task_result in collection_results.value:
                if task_result.status is not TaskAddStatus.success:
                    # actually we should probably only re-submit if it's a server_error
                    task_ids_failed_to_submit.append(task_result.task_id)
                    log.info(f'task {task_result.task_id} failed to submitted after {n_th_try} try/tries, '
                          f'status: {task_result.status}, error: {task_result.error}')

        return task_ids_failed_to_submit

    def get_num_completed_tasks(self, job_id: str) -> Tuple[int, int]:
        """
        Returns the number of completed tasks for the job of job_id, as a tuple:
        (number of succeeded jobs, number of failed jobs) - both are considered "completed".=
        """
        # docs: # https://docs.microsoft.com/en-us/rest/api/batchservice/odata-filters-in-batch#list-tasks
        tasks = self.batch_client.task.list(job_id,
                                            task_list_options=TaskListOptions(
                                                filter='state eq \'completed\'',
                                                select='id, executionInfo'  # only the id field will be non-empty
                                            ))
        num_succeeded, num_failed = 0, 0
        for task in tasks:
            exit_code: int = task.execution_info.exit_code
            if exit_code == 0:
                num_succeeded += 1
            else:
                num_failed += 1
        return num_succeeded, num_failed

    def cancel_batch_job(self, job_id: str):
        self.batch_client.job.terminate(job_id, terminate_reason='APIUserCanceled')

    def get_num_active_jobs(self) -> int:
        jobs_generator = self.batch_client.job.list(
            job_list_options=JobListOptions(
                filter='state eq \'active\'',
                select='id'
            ))
        jobs_list = [j for j in jobs_generator]
        return len(jobs_list)

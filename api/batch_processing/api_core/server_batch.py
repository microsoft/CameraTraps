# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
A class wrapping the Azure Batch client.
"""

import math
from typing import Tuple

from azure.batch import BatchServiceClient
from azure.batch.models import *
from azure.common.credentials import ServicePrincipalCredentials

import server_api_config as api_config


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
        print(f'BatchJobManager, create_job, job_id: {job_id}')
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
        print('BatchJobManager, submit_tasks')

        # cannot execute the scoring script that is in the mounted directory; has to be copied to cwd
        # not luck giving the commandline arguments via formatted string - set as env vars instead
        score_command = '/bin/bash -c \"cp $AZ_BATCH_NODE_MOUNTS_DIR/batch-api/scripts/score.py . && python score.py\" '

        num_images_per_task = api_config.NUM_IMAGES_PER_TASK

        # form shards of images and assign each shard to a Task
        num_tasks = math.ceil(num_images / num_images_per_task)

        tasks = []

        for task_id in range(num_tasks):
            begin_index = task_id * num_images_per_task
            end_index = begin_index + num_images_per_task

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
                ]
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
                print('BatchJobManager, submit_tasks, after retry, '
                      f'len of task_ids_failed_to_submit: {len(task_ids_failed_to_submit)}')
            else:
                print('BatchJobManager, submit_tasks, after retry, all Tasks submitted')
        else:
            print('BatchJobManager, submit_tasks, all Tasks submitted after first try')

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
                    print(f'task {task_result.task_id} failed to submitted after {n_th_try} try/tries, '
                          f'status: {task_result.status}, error: {task_result.error}')

        return task_ids_failed_to_submit

    def get_num_completed_tasks(self, job_id):
        tasks = self.batch_client.task.list(job_id,
                                            task_list_options=TaskListOptions(
                                                filter='state eq \'completed\'',
                                                select='id'  # only the id field will be non-empty
                                            ))
        completed_task_ids = [task.id for task in tasks]
        return len(completed_task_ids)

    def cancel_batch_job(self, job_id):
        self.batch_client.job.terminate(job_id, terminate_reason='APIUserCanceled')

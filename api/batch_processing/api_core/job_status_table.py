# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
A class to manage updating the status of an API request / Azure Batch Job.
"""
import os
import unittest
import uuid
from datetime import datetime

from azure.cosmos.cosmos_client import CosmosClient


class JobStatusTable:

    def __init__(self, api_instance):
        self.api_instance = api_instance
        COSMOS_ENDPOINT = os.environ['COSMOS_ENDPOINT']
        COSMOS_WRITE_KEY = os.environ['COSMOS_WRITE_KEY']

        cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_WRITE_KEY)
        db_client = cosmos_client.get_database_client('camera-trap')
        self.db_jobs_client = db_client.get_container_client('batch_api_jobs')


    def create_job_status(self, job_id: str, status: dict, call_params: dict):
        assert 'request_status' in status
        assert status['request_status'] in ['running', 'failed', 'problem', 'completed']
        assert 'message' in status

        # job_id should be unique across all instances, and is also the partition key
        item = {
            'id': job_id,
            'api_instance': self.api_instance,
            'status': status,
            'last_updated': str(datetime.now()),
            'call_params': call_params
        }
        created_item = self.db_jobs_client.create_item(item)
        return created_item


    def update_job_status(self, job_id: str, status: dict):
        assert 'request_status' in status
        assert status['request_status'] in ['running', 'failed', 'problem', 'completed']
        assert 'message' in status

        item = {
            'id': job_id,
            'api_instance': self.api_instance,
            'status': status,
            'last_updated': str(datetime.now()),
        }
        replaced_item = self.db_jobs_client.replace_item(job_id, item)
        return replaced_item


    def read_job_status(self, job_id):
        # job_id is also the partition key
        read_item = self.db_jobs_client.read_item(job_id, partition_key=job_id)
        item = {k: v for k, v in read_item.items() if not k.startswith('_')}
        return item


class TestJobStatusTable(unittest.TestCase):

    def test_insert(self):
        api_instance = 'api_test'
        table = JobStatusTable(api_instance)
        status = {
            'request_status': 'running',
            'message': 'this is a test'
        }
        job_id = uuid.uuid4().hex
        item = table.create_job_status(job_id, status, {'container_sas': 'random_string'})
        self.assertTrue(job_id == item['id'], 'Expect job_id to be the id of the item')
        self.assertTrue(item['status']['request_status'] == 'running', 'Expect fields to be inserted correctly')


    def test_update_and_read(self):
        api_instance = 'api_test'
        table = JobStatusTable(api_instance)
        status = {
            'request_status': 'running',
            'message': 'this is a test'
        }
        job_id = uuid.uuid4().hex
        res = table.create_job_status(job_id, status, {'container_sas': 'random_string'})

        status = {
            'request_status': 'completed',
            'message': 'this is a test again'
        }
        res = table.update_job_status(job_id, status)
        print(f'JOB_ID IS {job_id}')
        item_read = table.read_job_status(job_id)
        self.assertTrue(item_read['status']['request_status'] == 'completed', 'Expect field to have updated')


if __name__ == '__main__':
    unittest.main()

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import requests
import datetime
import json
from os import getenv
from urllib.parse import urlparse, urlunparse
from urllib3.util import Retry

class DistributedApiTaskManager:
    def __init__(self):
        self.cache_connector_upsert_url = getenv('CACHE_CONNECTOR_UPSERT_URI')
        self.cache_connector_get_url = getenv('CACHE_CONNECTOR_GET_URI')

    def AddTask(self):
        r = requests.post(self.cache_connector_upsert_url)

        if r.status_code != 200:
            return json.loads('{"TaskId": "-1", "Status": "error"}')
        else:
            return r.json()

    def _UpdateTaskStatus(self, taskId, status, backendStatus):
        old_stat = self.GetTaskStatus(taskId)
        endpoint = 'http://localhost'
        if old_stat['Status'] == "not found":
            print("Cannot find task status. Creating")
        else:
            endpoint = old_stat['Endpoint']
        
        r = requests.post(self.cache_connector_upsert_url,
            data=json.dumps(
                {'TaskId': taskId, 
                'Timestamp': datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S"), 
                'Status': status, 
                'BackendStatus': backendStatus,
                'Endpoint': endpoint,
                'PublishToGrid': False
                })
            )

        if r.status_code != 200:
            print("status code: " + str(r.status_code))
            return json.loads('{"TaskId": "' + taskId + '", "Status": "not found"}')
        else:
            return r.json()

    def CompleteTask(self, taskId, status):
        return self._UpdateTaskStatus(taskId, status, 'completed')

    def UpdateTaskStatus(self, taskId, status):
        return self._UpdateTaskStatus(taskId, status, 'running')

    def FailTask(self, taskId, status):
        return self._UpdateTaskStatus(taskId, status, 'failed')

    def AddPipelineTask(self, taskId, organization_moniker, version, api_name, body):
        old_stat = self.GetTaskStatus(taskId)
        if old_stat['Status'] == "not found":
            print("Cannot find task status.")
            return json.loads('{"TaskId": "-1", "Status": "error"}')
        
        parsed_endpoint = urlparse(old_stat['Endpoint'])

        next_endpoint = urlunparse(('http', parsed_endpoint.netloc, version + '/' + organization_moniker + '/' + api_name))
        print("Sending to next endpoint: " + next_endpoint)

        r = requests.post(self.cache_connector_upsert_url,
            data=json.dumps(
                {'TaskId': taskId, 
                'Timestamp': datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S"), 
                'Status': 'created', 
                'BackendStatus': 'created',
                'Endpoint': next_endpoint,
                'Body': body,
                'PublishToGrid': True
                })
            )

        if r.status_code != 200:
            print("status code: " + str(r.status_code))
            return json.loads('{"TaskId": "' + taskId + '", "Status": "not found"}')
        else:
            return r.json()

    def GetTaskStatus(self, taskId):
        session = requests.Session()
        retry = Retry(read=0)
        adapter = requests.adapters.HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)

        r = session.get(self.cache_connector_get_url, params={'taskId': taskId})

        if r.status_code != 200:
            return json.loads('{"TaskId": "' + taskId + '", "Status": "not found"}')
        else:
            return r.json()


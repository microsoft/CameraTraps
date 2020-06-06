# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from threading import Thread
from os import getenv
from opencensus.trace.tracer import Tracer
from opencensus.ext.azure.trace_exporter import AzureExporter

from flask import Flask, abort, request, current_app, views
from flask_restful import Resource, Api
import signal
from ai4e_app_insights import AppInsights
from task_management.api_task import TaskManager
import sys
from functools import wraps
from werkzeug.exceptions import HTTPException

from opencensus.trace.tracer import Tracer
from opencensus.trace.samplers import ProbabilitySampler

import requests
import json

if not getenv('APPINSIGHTS_INSTRUMENTATIONKEY', None):
    tracer = Tracer()
else:
    sampling_rate = getenv('TRACE_SAMPLING_RATE', None)
    if not sampling_rate:
        sampling_rate = 1.0

    tracer = Tracer(
        exporter=AzureExporter(),
        sampler=ProbabilitySampler(float(sampling_rate)),
    )

disable_request_metric = getenv('DISABLE_CURRENT_REQUEST_METRIC', True)

current_processing_upsert_url = getenv('CURRENT_PROCESSING_UPSERT_URI')
service_cluster = getenv('SERVICE_CLUSTER', 'undefined')

MAX_REQUESTS_KEY_NAME = 'max_requests'
CONTENT_TYPE_KEY_NAME = 'content_type'
CONTENT_MAX_KEY_NAME = 'content_max_length'

APP_INSIGHTS_REQUESTS_KEY_NAME = 'CURRENT_REQUESTS'

class Task(Resource):
    def __init__(self, **kwargs):
        self.task_mgr = kwargs['task_manager']

    def get(self, id):
        st = self.task_mgr.GetTaskStatus(str(id))
        ret = {}
        ret['TaskId'] = id
        ret['Status'] = st[0]
        ret['Timestamp'] = st[1]
        ret['Endpoint'] = "uri"
        return(ret)

class APIService():
    def __init__(self, flask_app, logger):
        self.app = flask_app
        self.log = logger
        self.api = Api(self.app)
        self.appinsights = AppInsights(self.app)
        self.is_terminating = False
        self.func_properties = {}
        self.func_request_counts = {}
        self.api_prefix = getenv('API_PREFIX')
        
        self.api_task_manager = TaskManager()
        signal.signal(signal.SIGINT, self.initialize_term)

        # Add health check endpoint
        self.app.add_url_rule(self.api_prefix + '/', view_func = self.health_check, methods=['GET'])
        print("Adding url rule: " + self.api_prefix + '/')
        # Add task endpoint
        self.api.add_resource(Task, self.api_prefix + '/task/<int:id>', resource_class_kwargs={ 'task_manager': self.api_task_manager })
        print("Adding url rule: " + self.api_prefix + '/task/<int:taskId>')

        self.app.before_request(self.before_request)

    def health_check(self):
        print("Health check call successful.")
        return 'Health check OK'

    def api_func(self, is_async, api_path, methods, request_processing_function, maximum_concurrent_requests, content_types = None, content_max_length = None, trace_name = None, *args, **kwargs):
        def decorator_api_func(func):
            if not api_path in self.func_properties:
                self.func_properties[api_path] = {MAX_REQUESTS_KEY_NAME: maximum_concurrent_requests, CONTENT_TYPE_KEY_NAME: content_types, CONTENT_MAX_KEY_NAME: content_max_length}
                self.func_request_counts[api_path] = 0

            @wraps(func)
            def api(*args, **kwargs):
                internal_args = {"func": func, "api_path": api_path}

                if request_processing_function:
                    return_values = request_processing_function(request)
                    combined_kwargs = {**internal_args, **kwargs, **return_values}
                else:
                    combined_kwargs = {**internal_args, **kwargs}
                
                if is_async:
                    task_info = self.api_task_manager.AddTask(request)
                    taskId = str(task_info['TaskId'])
                    combined_kwargs["taskId"] = taskId

                    self.wrap_async_endpoint(trace_name, *args, **combined_kwargs)
                    return 'TaskId: ' + taskId
                else:
                    return self.wrap_sync_endpoint(trace_name, *args, **combined_kwargs)

            api.__name__ = 'api_' + api_path.replace('/', '')
            print("Adding url rule: " + self.api_prefix + api_path + ", " + api.__name__)
            self.app.add_url_rule(self.api_prefix + api_path, view_func = api, methods=methods, provide_automatic_options=True)
        return decorator_api_func

    def api_async_func(self, api_path, methods, request_processing_function = None, maximum_concurrent_requests = None, content_types = None, content_max_length = None, trace_name = None, *args, **kwargs):
        is_async = True
        return self.api_func(is_async, api_path, methods, request_processing_function, maximum_concurrent_requests, content_types, content_max_length, trace_name, *args, **kwargs)

    def api_sync_func(self, api_path, methods, request_processing_function = None, maximum_concurrent_requests = None, content_types = None, content_max_length = None, trace_name=None, *args, **kwargs):
        is_async = False
        return self.api_func(is_async, api_path, methods, request_processing_function, maximum_concurrent_requests, content_types, content_max_length, trace_name, *args, **kwargs)

    def initialize_term(self, signum, frame):
        print('Signal handler called with signal: ' + signum)
        print('SIGINT received, service is terminating and will no longer accept requests.')
        self.is_terminating = True

    def before_request(self):
        # Don't accept a request if SIGTERM has been called on this instance.
        if (self.is_terminating):
            print('Process is being terminated. Request has been denied.')
            abort(503, {'message': 'Service is busy, please try again later.'})

        if request.path in self.func_properties:
            if (self.func_request_counts[request.path] + 1 > self.func_properties[request.path][MAX_REQUESTS_KEY_NAME]):
                print('Service is busy. Request has been denied.')
                abort(503, {'message': 'Service is busy, please try again later.'})

            if (self.func_properties[request.path][CONTENT_TYPE_KEY_NAME] and not request.content_type in self.func_properties[request.path][CONTENT_TYPE_KEY_NAME]):
                print('Invalid content type. Request has been denied.')
                abort(401, {'message': 'Content-type must be ' + self.func_properties[request.path][CONTENT_TYPE_KEY_NAME]})

            if (self.func_properties[request.path][CONTENT_MAX_KEY_NAME] and request.content_length > self.func_properties[request.path][CONTENT_MAX_KEY_NAME]):
                print('Request is too large. Request has been denied.')
                abort(413, {'message': 'Request content too large (' + str(request.content_length) + "). Must be smaller than: " + str(self.func_properties[request.path][CONTENT_MAX_KEY_NAME])})

    def update_processing_count(self, api_path, increment_by, decrement_by):
        r = requests.post(current_processing_upsert_url,
            data=json.dumps(
                {'ApiPath': self.api_prefix + api_path, 
                'ServiceCluster': service_cluster, 
                'IncrementBy': increment_by, 
                'DecrementBy': decrement_by
                })
        )

        if r.status_code != 200:
            print("status code: " + str(r.status_code))

    def increment_requests(self, api_path):
        self.func_request_counts[api_path] += 1
        if (disable_request_metric == False):
            self.update_processing_count(api_path, increment_by=1, decrement_by=0)

    def decrement_requests(self, api_path):
        self.func_request_counts[api_path] -= 1
        if (disable_request_metric == False):
            self.update_processing_count(api_path, increment_by=0, decrement_by=1)

    def wrap_sync_endpoint(self, trace_name=None, *args, **kwargs):
        if (trace_name):
            with tracer.span(name=trace_name) as span:
                return self._execute_func_with_counter(False, *args, **kwargs)
        else:
            return self._execute_func_with_counter(False, *args, **kwargs)

    def wrap_async_endpoint(self, trace_name=None, *args, **kwargs):
        if (trace_name):
            with tracer.span(name=trace_name) as span:
                self._create_and_execute_thread(*args, **kwargs)
        else:
            self._create_and_execute_thread(*args, **kwargs)

    def _create_and_execute_thread(self, *args, **kwargs):
        kwargs['request'] = request
        thread = Thread(target = self._execute_func_with_counter, args=args, kwargs=kwargs)
        thread.start()

    def _log_and_fail_exeception(self, is_async, **kwargs):
        if ('taskId' in kwargs):
            taskId = kwargs['taskId']
            if taskId:
                self.log.log_exception(sys.exc_info()[0], taskId)
                if is_async:
                    self.api_task_manager.FailTask(taskId, 'Task failed - try again')
            else:
                self.log.log_exception(sys.exc_info()[0])
        else:
            self.log.log_exception(sys.exc_info()[0])

    def _execute_func_with_counter(self, is_async=True, *args, **kwargs):
        func = kwargs['func']
        api_path = kwargs['api_path']

        self.increment_requests(api_path)
        try:
            r = func(*args, **kwargs)
            return r
        except HTTPException as e:
            self._log_and_fail_exeception(is_async, **kwargs)
            return e
        except:
            print(sys.exc_info()[0])
            self._log_and_fail_exeception(is_async, **kwargs)
            abort(500)
        finally:
            self.decrement_requests(api_path)
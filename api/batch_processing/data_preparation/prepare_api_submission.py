"""

prepare_api_submission.py

This module defines the Task class and helper methods that are useful for
submitting tasks to the AI for Earth Camera Trap Batch Detection API.

Here's the stuff we usually do before submitting a task:

1) Upload images to Azure Blob Storage... we do this with azcopy, not addressed
    in this script.

2) List the files you want the API to process.
    ai4eutils.ai4e_azure_utils.enumerate_blobs_to_file()

3) Divide that list into chunks that will become individual API submissions.
    divide_files_into_tasks()

3) Put each .json file in a blob container and get a read-only SAS URL for it.
    Task.upload_images_list()

4) Generate the API query(ies) you'll submit to the API.
    Task.generate_api_request()

5) Submit the API query. This can be done manually with Postman as well.
    Task.submit()

6) Monitor task status
    Task.check_status()

7) Combine multiple API outputs

8) We're now into what we really call "postprocessing", rather than
    "data_preparation", but... possibly do some amount of partner-specific
    renaming, folder manipulation, etc. This is very partner-specific, but
    generally done via:

    find_repeat_detections.py
    subset_json_detector_output.py
    postprocess_batch_results.py
    
"""


#%% Imports

from enum import Enum
import json
import os
import posixpath
import string
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple
import urllib

import requests

import ai4e_azure_utils  # from ai4eutils
import path_utils  # from ai4eutils


#%% Constants

MAX_FILES_PER_API_TASK = 1_000_000
IMAGES_PER_SHARD = 2000

VALID_REQUEST_NAME_CHARS = f'-_{string.ascii_letters}{string.digits}'
REQUEST_NAME_CHAR_LIMIT = 92


#%% Classes

class BatchAPISubmissionError(Exception):
    pass


class BatchAPIResponseError(Exception):
    pass


class TaskStatus(str, Enum):
    RUNNING = 'running'
    FAILED = 'failed'
    PROBLEM = 'problem'
    COMPLETED = 'completed'    


class Task:
    """
    Represents a Batch Detection API task.

    Given the Batch Detection API URL, assumes that the endpoints are:
        /request_detections
            for submitting tasks
        /task/<task.id>
            for checking on task status
    """

    # class variables
    request_endpoint: ClassVar[str] = 'request_detections'  # submit tasks
    task_status_endpoint: ClassVar[str] = 'task'            # check task status

    # instance variables, in order of when they are typically set
    name: str
    api_url: str
    local_images_list_path: str
    remote_images_list_url: str  # includes SAS token if uploaded with one
    api_request: Dict[str, Any]  # request object before JSON serialization
    id: str
    response: Dict[str, Any]  # decoded response JSON
    status: TaskStatus
    bypass_status_check: bool # set when we manually complete a task

    def __init__(self, name: str, task_id: Optional[str] = None,
                 images_list_path: Optional[str] = None,
                 validate: bool = True, api_url: Optional[str] = None):
        """
        Initializes a Task.

        If desired, validates that the images list does not exceed the maximum
        length and that all files in the images list are actually images.

        Args:
            name: str, name of the request
            task_id: optional str, ID of submitted task
            images_list_path: str, path or URL to a JSON file containing a list
                of image paths, must start with 'http' if a URL
            local: bool, set to True if images_list_path is a local path,
                set to False if images_list_path is a URL
            validate: bool, whether to validate the given images list,
                only used if images_list_path is not None
            api_url: optional str, Batch Detection API URL,
                defaults to environment variable BATCH_DETECTION_API_URL

        Raises:
            requests.HTTPError: if images_list_path is a URL but an error
                occurred trying to fetch it
            ValueError: if images_list_path is given, but the file contains more
                than MAX_FILES_PER_API_TASK entries, or if one of the entries
                is not a supported image file type
        """
        self.bypass_status_check = False
        
        clean_name = clean_request_name(name)
        if name != clean_name:
            print('Warning: renamed {} to {}'.format(name,clean_name))
        self.name = clean_name

        if api_url is None:
            api_url = os.environ['BATCH_DETECTION_API_URL']
        assert api_url is not None and api_url != ''
        self.api_url = api_url

        if task_id is not None:
            self.id = task_id

        if images_list_path is not None:
            
            if images_list_path.startswith('http'):
                self.remote_images_list_url = images_list_path
            else:
                self.local_images_list_path = images_list_path

            if validate:
                
                if images_list_path.startswith('http'):
                    images_list = requests.get(images_list_path).json()
                else:
                    with open(images_list_path, 'r') as f:
                        images_list = json.load(f)

                if len(images_list) > MAX_FILES_PER_API_TASK:
                    raise ValueError('Images list has too many files')

                # Leaving this commented out to remind us that we don't want this check here; let
                # the API fail on these images.  It's a huge hassle to remove non-image
                # files.
                    #
                # for path_or_url in images_list:
                #     if not is_image_file_or_url(path_or_url):
                #         raise ValueError('{} is not an image'.format(path_or_url))


    def __repr__(self) -> str:
        return 'Task(name={name}, id={id})'.format(
            name=self.name,
            id=getattr(self, 'id', None))
            # Commented out as a reminder: don't check task status (which is a rest API call)
            # in __repr__; require the caller to explicitly request status     
            # status=getattr(self, 'status', None))


    def upload_images_list(self, account: str, container: str, sas_token: str,
                           blob_name: Optional[str] = None, overwrite: bool=False) -> None:
        """
        Uploads the local images list to an Azure Blob Storage container.

        Sets self.remote_images_list_url to the blob URL of the uploaded file.

        Args:
            account: str, Azure Storage account name
            container: str, Azure Blob Storage container name
            sas_token: str, Shared Access Signature (SAS) with write permission,
                does not start with '?'
            blob_name: optional str, defaults to basename of
                self.local_images_list_path if blob_name is not given
        """
        
        if blob_name is None:
            blob_name = os.path.basename(self.local_images_list_path)
        self.remote_images_list_url = ai4e_azure_utils.upload_file_to_blob(
            account_name=account, container_name=container,
            local_path=self.local_images_list_path, blob_name=blob_name,
            sas_token=sas_token, overwrite=overwrite)


    def generate_api_request(self,
                             caller: str,
                             input_container_url: Optional[str] = None,
                             image_path_prefix: Optional[str] = None,
                             **kwargs: Any
                             ) -> Dict[str, Any]:
        """
        Generate API request JSON.

        Sets self.api_request to the request JSON. For complete list of API
        input parameters, see:
        https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing#api-inputs

        Args:
            caller: str
            input_container_url: optional str, URL to Azure Blob Storage
                container where images are stored. URL must include SAS token
                with read and list permissions if the container is not public.
                Only provide this parameter when the image paths in
                self.remote_images_list_url are relative to a container.
            image_path_prefix: optional str, TODO
            kwargs: additional API input parameters

        Returns: dict, represents the JSON request to be submitted
        """
        
        request = kwargs
        request.update({
            'request_name': self.name,
            'caller': caller,
            'images_requested_json_sas': self.remote_images_list_url
        })
        if input_container_url is None:
            request['use_url'] = True
        else:
            request['input_container_sas'] = input_container_url
        if image_path_prefix is not None:
            request['image_path_prefix'] = image_path_prefix
        self.api_request = request
        return request


    def submit(self) -> str:
        """
        Submit this task to the Batch Detection API.

        Sets self.id to the returned request ID. Only run this method after
        generate_api_request().

        Returns: str, task ID

        Raises:
            requests.HTTPError, if an HTTP error occurred
            BatchAPISubmissionError, if request returns an error
        """
        
        request_endpoint = posixpath.join(self.api_url, self.request_endpoint)
        r = requests.post(request_endpoint, json=self.api_request)
        r.raise_for_status()
        assert r.status_code == requests.codes.ok

        response = r.json()
        if 'error' in response:
            raise BatchAPISubmissionError(response['error'])
        if 'request_id' not in response:
            raise BatchAPISubmissionError(
                '"request_id" not in API response: {}'.format(response))
        self.id = response['request_id']
        return self.id


    def check_status(self) -> Dict[str, Any]:
        """
        Checks the task status.

        Sets self.response and self.status.

        Returns: dict, contains fields ['Status', 'TaskId'] and possibly others

        Raises:
            requests.HTTPError, if an HTTP error occurred
            BatchAPIResponseError, if response task ID does not match self.id
        """
        
        if self.bypass_status_check:
            return self.response
        
        url = posixpath.join(self.api_url, self.task_status_endpoint, self.id)
        r = requests.get(url)

        r.raise_for_status()
        assert r.status_code == requests.codes.ok

        self.response = r.json()
        if self.response['TaskId'] != self.id:
            raise BatchAPIResponseError(
                f'Response task ID {self.response["TaskId"]} does not match '
                f'expected task ID {self.id}.')
        try:
            self.status = TaskStatus(self.response['Status']['request_status'])
        except Exception as e:
            self.status = 'Exception error: {}'.format(str(e))
        return self.response


    def force_completion(self,response) -> None:
        """
        Simulate completion of a task by passing a manually-created response
        string.
        """
        self.response = response
        self.status = TaskStatus(self.response['Status']['request_status'])
        self.bypass_status_check = True
        
        
    def get_output_file_urls(self, verbose: bool = False) -> Dict[str, str]:
        """
        Retrieves the dictionary of URLs for the three output files for this task
        """
        
        assert self.status == TaskStatus.COMPLETED
        message = self.response['Status']['message']
        output_file_urls = message['output_file_urls']
        return output_file_urls
        
        
    def get_missing_images(self, submitted_images, verbose: bool = False) -> List[str]:
        """
        Compares the submitted and processed images lists to find missing
        images.

        "missing": an image from the submitted list that was not processed,
            for whatever reason
        "failed": a missing image explicitly marked as 'failed' by the
            batch detection API

        Only run this method when task.status == TaskStatus.COMPLETED.

        Returns: list of str, sorted list of missing image paths
        
        Ignores non-image filenames.
        """
        
        assert self.status == TaskStatus.COMPLETED
        message = self.response['Status']['message']

        # estimate # of failed images from failed shards
        if 'num_failed_shards' in message:
            n_failed_shards = message['num_failed_shards']
        else:
            n_failed_shards = 0
        
        # Download all three JSON urls to memory
        output_file_urls = message['output_file_urls']
        for url in output_file_urls.values():
            if self.id not in url:
                raise BatchAPIResponseError(
                    'Task ID missing from output URL: {}'.format(url))
        detections = requests.get(output_file_urls['detections']).json()
        
        return get_missing_images_from_json(submitted_images,detections,n_failed_shards,verbose)
    
    
def create_response_message(n_failed_shards,detections_url,task_id):
    """
    Manually create a response message in the format of the batch API.  Used when tasks hang or fail
    and we need to simulate their completion by directly pulling the results from the AML output.
    """
    output_file_urls = {
        'detections':detections_url
        }
    message = {'num_failed_shards':n_failed_shards,'output_file_urls':output_file_urls}
    status = {'message':message,'request_status':str(TaskStatus.COMPLETED.value)}
    response = {}
    response['Status'] = status
    response['request_id'] = task_id
    return response
    

def get_missing_images_from_json(submitted_images,detections,n_failed_shards,verbose=False):
    """
    Given the json-encoded results for the lists of submitted images and detections,
    find and return the list of images missing in the list of detections.  Ignores
    non-image filenames.
    """
    
    # Remove files that were submitted but don't appear to be images
    # assert all(is_image_file_or_url(s) for s in submitted_images)
    non_image_files_submitted = [s for s in submitted_images if not is_image_file_or_url(s)]
    if len(non_image_files_submitted) > 0:
        print('Warning, {} non-image files submitted:\n'.format(len(non_image_files_submitted)))
        for k in range(0,min(10,len(non_image_files_submitted))):
            print(non_image_files_submitted[k])
        print('...\n')
        
    submitted_images = [s for s in submitted_images if is_image_file_or_url(s)]
            
    # Diff submitted and processed images
    processed_images = [d['file'] for d in detections['images']]
    missing_images = sorted(set(submitted_images) - set(processed_images))
    
    if verbose:
        estimated_failed_shard_images = n_failed_shards * IMAGES_PER_SHARD
        print('Submitted {} images'.format(len(submitted_images)))
        print('Received results for {} images'.format(len(processed_images)))
        print(f'{n_failed_shards} failed shards '
              f'(~approx {estimated_failed_shard_images} images)')
        print('{} images not in results'.format(len(missing_images)))

    # Confirm that the procesed images are a subset of the submitted images
    assert set(processed_images) <= set(submitted_images), (
        'Failed images should be a subset of missing images')
    
    return missing_images

    
def divide_chunks(l: Sequence[Any], n: int) -> List[Sequence[Any]]:
    """
    Divide list *l* into chunks of size *n*, with the last chunk containing
    <= n items.
    """
    
    # https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    chunks = [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n)]
    return chunks


def divide_list_into_tasks(file_list: Sequence[str],
                           save_path: str,
                           n_files_per_task: int = MAX_FILES_PER_API_TASK
                           ) -> Tuple[List[str], List[Sequence[Any]]]:
    """
    Divides a list of filenames into a set of JSON files, each containing a
    list of length *n_files_per_task* (the last file will contain <=
    *n_files_per_task* files).

    Output JSON files are saved to *save_path* except the extension is replaced
    with `*.chunkXXX.json`. For example, if *save_path* is `blah.json`, output
    files will be `blah.chunk000.json`, `blah.chunk001.json`, etc.

    Args:
        file_list: list of str, filenames to split across multiple JSON files
        save_path: str, base path to save the chunked lists
        n_files_per_task: int, max number of files to include in each API task

    Returns:
        output_files: list of str, output JSON file names
        chunks: list of list of str, chunks[i] is the content of output_files[i]
    """
    
    chunks = divide_chunks(file_list, n_files_per_task)
    output_files = []

    for i_chunk, chunk in enumerate(chunks):
        chunk_id = 'chunk{:0>3d}'.format(i_chunk)
        output_file = path_utils.insert_before_extension(
            save_path, chunk_id)
        output_files.append(output_file)
        with open(output_file, 'w') as f:
            json.dump(chunk, f, indent=1)
    return output_files, chunks


def divide_files_into_tasks(file_list_json: str,
                            n_files_per_task: int = MAX_FILES_PER_API_TASK
                            ) -> Tuple[List[str], List[Sequence[Any]]]:
    """
    Convenience wrapper around divide_list_into_tasks() when the file_list
    itself is already saved as a JSON file.
    """
    
    with open(file_list_json) as f:
        file_list = json.load(f)
    return divide_list_into_tasks(file_list, save_path=file_list_json,
                                  n_files_per_task=n_files_per_task)


def clean_request_name(request_name: str,
                       whitelist: str = VALID_REQUEST_NAME_CHARS,
                       char_limit: int = REQUEST_NAME_CHAR_LIMIT) -> str:
    """
    Removes invalid characters from an API request name.
    """
    return path_utils.clean_filename(
        filename=request_name, whitelist=whitelist, char_limit=char_limit).replace(':','_')


def download_url(url: str, save_path: str, verbose: bool = False) -> None:
    """
    Download a URL to a local file.
    """
    if verbose:
        print('Downloading {} to {}'.format(url,save_path))
    urllib.request.urlretrieve(url, save_path)
    assert os.path.isfile(save_path)


def is_image_file_or_url(path_or_url: str) -> bool:
    """
    Checks (via file extension) whether a file path or URL is an image.

    If path_or_url is a URL, strip away any query strings '?...'. This should
    have no adverse effect on local paths.
    """
    stripped_path_or_url = urllib.parse.urlparse(path_or_url).path
    return path_utils.is_image_file(stripped_path_or_url)


#%% Interactive driver

if False:

    #%%
    
    account_name = ''
    sas_token = 'st=...'
    container_name = ''
    rsearch = None # '^Y53'
    output_file = r'output.json'
    
    blobs = ai4e_azure_utils.enumerate_blobs_to_file(
        output_file=output_file,
        account_name=account_name,
        sas_token=sas_token,
        container_name=container_name,
        rsearch=rsearch)
    
    #%%
    
    file_list_json = r"D:\temp\idfg_20190801-hddrop_image_list.json"
    task_files = divide_files_into_tasks(file_list_json)
    
    #%%
    
    file_list_sas_urls = [
        '','',''
    ]
    
    input_container_sas_url = ''
    request_name_base = ''
    caller = 'blah@blah.com'
    
    request_strings,request_dicts = generate_api_queries(
        input_container_sas_url,
        file_list_sas_urls,
        request_name_base,
        caller)
    
    for s in request_strings:
        print(s)

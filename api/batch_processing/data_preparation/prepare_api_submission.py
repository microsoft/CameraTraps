#
# prepare_api_submission.py
#
# This module is somewhere between "documentation" and "code".  It is intended to 
# capture the steps the precede running a job via the AI for Earth Camera Trap
# Image Processing API, and it automates a couple of those steps.  We hope to 
# gradually automate all of these.
#
# Here's the stuff we usually do before submitting a job:
#
# 1) Upload data to Azure... we do this with azcopy, not addressed in this script
#
# 2) List the files you want the API to process... this module supports that via
#    enumerate_blobs_to_file.
#
# 3) Divide that list into chunks that will become individual API submissions...
#    this module supports that via divide_files_into_tasks.

# 3) Put each .json file in a blob container, and generate a read-only SAS 
#    URL for it.  Not automated right now.
#
# 4) Generate the API query(ies) you'll submit to the API... this module supports that
#    via generate_api_queries.
#
# 5) Submit the API query... I currently do this with Postman.
#
# 6) Monitor task status
#
# 7) Combine multiple API outputs
#
# 8) We're now into what we really call "postprocessing", rather than "data_preparation", 
#    but... possibly do some amount of partner-specific renaming, folder manipulation, etc.  
#    This is very partner-specific, but generally done via:
#
#    find_repeat_detections.py
#    subset_json_detector_output.py.
#    postprocess_batch_results.py
#

#%% Imports and constants

import json
import re
import string
import unicodedata

from azure.storage.blob import BlobServiceClient

# assumes ai4eutils is on the path
import path_utils

default_n_files_per_api_task = 1000000


#%% File enumeration

def concatenate_json_string_lists(input_files,output_file=None):
    """
    Given several files that contain json-formatted lists of strings (typically filenames),
    concatenate them into one new file.
    """
    output_list = []
    for fn in input_files:
        file_list = json.load(open(fn)) 
        output_list.extend(file_list)
    if output_file is not None:
        s = json.dumps(output_list,indent=1)
        with open(output_file,'w') as f:
            f.write(s)
    return output_list

        
def write_list_to_file(output_file,strings):
    """
    Writes a list of strings to file, either .json or text depending on extension
    """
    if output_file.endswith('.json'):
        s = json.dumps(strings,indent=1)
        with open(output_file,'w') as f:
            f.write(s)
    else:
        with open(output_file,'w') as f:
            for fn in strings:
                f.write(fn + '\n')
                
    # print('Finished writing list {}'.format(output_file))
    
   
def read_list_from_file(filename):
    """
    Reads a json-formatted list of strings from *filename*
    """
    assert filename.endswith('.json')
    file_list = json.load(open(filename))             
    assert isinstance(file_list,list)
    for s in file_list:
        assert isinstance(s,str)
    return file_list
    

def account_name_to_url(account_name):
    storage_account_url_blob = 'https://' + account_name + '.blob.core.windows.net'
    return storage_account_url_blob


def copy_file_to_blob(account_name,sas_token,container_name,
                      local_path,remote_path):
    """
    Copies a local file to blob storage
    """
    blob_service_client = BlobServiceClient(account_url=account_name_to_url(account_name), 
                                            credential=sas_token)
    
    container_client = blob_service_client.get_container_client(container_name)

    with open(local_path, 'rb') as data:
        container_client.upload_blob(remote_path, data)
    
    
def enumerate_blobs(account_name,sas_token,container_name,rmatch=None,prefix=None):
    """
    Enumerates blobs in a container, optionally filtering with a regex
    
    Using the prefix parameter is faster than using a regex starting with ^
    
    sas_token should start with st=
    """
    
    folder_string = '{}/{}'.format(account_name,container_name)
    if prefix is not None:
        folder_string += '/{}'.format(prefix)
    if rmatch is not None:
        folder_string += ' (matching {})'.format(rmatch)
    print('Enumerating blobs from {}'.format(folder_string))
        
    blob_service_client = BlobServiceClient(account_url=account_name_to_url(account_name), 
                                            credential=sas_token)
    
    container_client = blob_service_client.get_container_client(container_name)
    
    generator = container_client.list_blobs(name_starts_with=prefix)
    matched_blobs = []

    i_blob = 0
    for blob in generator:
        blob_name = blob.name
        if rmatch is not None:
            m = re.match(rmatch,blob_name)
            if m is None:
                continue
        matched_blobs.append(blob.name)
        i_blob += 1
        if (i_blob % 1000) == 0:
            print('.',end='')
        if (i_blob % 50000) == 0:
            print('{} blobs enumerated ({} matches)'.format(i_blob,len(matched_blobs)))
                
    print('Enumerated {} matching blobs (of {} total) from {}/{}'.format(len(matched_blobs),
          i_blob,account_name,container_name))

    return matched_blobs


def enumerate_blobs_to_file(output_file,account_name,sas_token,container_name,account_key=None,rmatch=None,prefix=None):
    """
    Enumerates to a .json string if output_file ends in ".json", otherwise enumerates to a 
    newline-delimited list.
    
    See enumerate_blobs for parameter information.
    """        
    
    matched_blobs = enumerate_blobs(account_name=account_name,
                                    sas_token=sas_token,
                                    container_name=container_name,
                                    rmatch=rmatch,
                                    prefix=prefix)
    
    write_list_to_file(output_file,matched_blobs)
    return matched_blobs


def enumerate_image_blobs(account_name,sas_token,container_name,
                                         account_key=None,rmatch=None,prefix=None):    
    """
    Enumerates files from a blob container, returning only files with image extensions
    
    See enumerate_blobs for parameter information.
    """        
    matched_blobs = enumerate_blobs(account_name,sas_token,container_name,account_key=None,rmatch=None,prefix=None)
    matched_blobs = path_utils.find_image_strings(matched_blobs)
    return matched_blobs
    

def enumerate_image_blobs_fo_file(output_file,account_name,sas_token,container_name,
                                         account_key=None,rmatch=None,prefix=None):    
    """
    Enumerates files from a blob container, returning only files with image extensions
    
    See enumerate_blobs for parameter information.
    """        
    matched_blobs = enumerate_blobs(account_name,sas_token,container_name,account_key=None,rmatch=None,prefix=None)
    matched_blobs = path_utils.find_image_strings(matched_blobs)
    write_list_to_file(output_file,matched_blobs)
    return matched_blobs
    

#%% Dividing files into multiple tasks

def divide_chunks(l, n): 
    """
    Divide list *l* into chunks of size *n*, with the last chunk containing <= n items.
    """
    # https://www.geeksforgeeks.org/break-list-chunks-size-n-python/    
    chunks = [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n )]
    return chunks
        

def divide_files_into_tasks(file_list_json,n_files_per_task=default_n_files_per_api_task):
    """
    Divides the file *file_list_json*, which should contain a single json-encoded list
    of strings, into a set of json files, each containing *n_files_per_task* files
    (the last file will contain <= *n_files_per_task* files).
    
    If the input .json is blah.json, output_files will be blah.chunk000.json,
    blah.chunk001.json, etc.
    
    Returns the list of .json filenames and the list of lists of files.
    
    return output_files,chunks
    """
    
    with open(file_list_json) as f:
        file_list = json.load(f)
    
    chunks = divide_chunks(file_list,n_files_per_task)
    
    output_files = []
    
    # i_chunk = 0; chunk = chunks[0]
    for i_chunk,chunk in enumerate(chunks):
        chunk_id = 'chunk{0:0>3d}'.format(i_chunk)
        output_file = path_utils.insert_before_extension(file_list_json,chunk_id)
        output_files.append(output_file)
        s = json.dumps(chunk,indent=1)
        with open(output_file,'w') as f:
            f.write(s)
    
    return output_files,chunks    

valid_request_name_chars = "-_%s%s" % (string.ascii_letters, string.digits)
request_name_char_limit = 100

def clean_request_name(request_name, whitelist=valid_request_name_chars):
    """
    Removes invalid characters from an API request name
    """    
    cleaned_name = unicodedata.normalize('NFKD', request_name).encode('ASCII', 'ignore').decode()
    
    # keep only whitelisted chars
    cleaned_name = ''.join([c for c in cleaned_name if c in whitelist])
    return cleaned_name[:request_name_char_limit]  


def generate_api_queries(input_container_sas_url,file_list_sas_urls,request_name_base,
                         caller,additional_args={},image_path_prefixes=None):
    """
    Generate .json-formatted API input from input parameters.  file_list_sas_urls is
    a list of SAS URLs to individual file lists (all relative to the same container).
    
    request_name_base is a request name for the set; if the base name is 'blah', individual
    requests will get request names of 'blah_chunk000', 'blah_chunk001', etc.
    
    additional_args is a dictionary of custom arguments to be added to each query (to specify
    different custom args per query, use multiple calls to generate_api_query())
    
    image_path_prefixes, if supplied, can be a single string or a list of strings 
    (one per request)
    
    Returns both strings and Python dicts
    
    return request_strings,request_dicts
    """
    
    assert isinstance(file_list_sas_urls,list)        

    request_name_original = request_name_base
    request_name_base = clean_request_name(request_name_base)
    if request_name_base != request_name_original:
        print('Warning: renamed {} to {}'.format(request_name_original,request_name_base))
        
    request_dicts = []
    request_strings = []
    # i_url = 0; file_list_sas_url = file_list_sas_urls[0]
    for i_url,file_list_sas_url in enumerate(file_list_sas_urls):
        
        d = {}
        d['input_container_sas'] = input_container_sas_url
        d['images_requested_json_sas'] = file_list_sas_url
        if len(file_list_sas_urls) > 1:
            chunk_id = '_chunk{0:0>3d}'.format(i_url)
            request_name = request_name_base + chunk_id
        else:
            request_name = request_name_base
        d['request_name'] = request_name
        d['caller'] = caller
        
        for k in additional_args.keys():
            d[k] = additional_args[k]
            
        if image_path_prefixes is not None:
            if not isinstance(image_path_prefixes,list):
                d['image_path_prefix'] = image_path_prefixes
            else:
                d['image_path_prefix'] = image_path_prefixes[i_url]
        request_dicts.append(d)
        request_strings.append(json.dumps(d,indent=1))
    
    return request_strings,request_dicts


def generate_api_query(input_container_sas_url,file_list_sas_url,request_name,caller,
                       additional_args={},image_path_prefix=None):
    """
    Convenience function to call generate_api_queries for a single batch.
    
    See generate_api_queries, and s/lists/single items.
    """
    
    file_list_sas_urls = [file_list_sas_url]
    image_path_prefixes = [image_path_prefix]
    request_strings,request_dicts = generate_api_queries(input_container_sas_url,
                                                         file_list_sas_urls,
                                                         request_name,
                                                         caller,
                                                         additional_args,
                                                         image_path_prefixes)
    return request_strings[0],request_dicts[0]


#%% Tools for working with API output

# I suspect this whole section will move to a separate file at some point,
# so leaving these imports and constants here for now.
from posixpath import join as urljoin

import urllib
import tempfile    
import os
import requests
    
ct_api_temp_dir = os.path.join(tempfile.gettempdir(),'camera_trap_api')
IMAGES_PER_SHARD = 2000

def fetch_task_status(endpoint_url,task_id):
    """
    Currently a very thin wrapper to fetch the .json content from the task URL
    
    Returns status dictionary,status code
    """
    response = requests.get(urljoin(endpoint_url,str(task_id)))
    return response.json(),response.status_code


def get_output_file_urls(response):
    """
    Given the dictionary returned by fetch_task_status, get the set of
    URLs returned at the end of the task, or None if they're not available.'    
    """
    try:
        output_file_urls = response['status']['message']['output_file_urls']
    except:
        return None
    assert 'detections' in output_file_urls
    assert 'failed_images' in output_file_urls
    assert 'images' in output_file_urls
    return output_file_urls
    

def download_url(url, destination_filename, verbose=False):
    """
    Download a URL to a local file
    """
    if verbose:
        print('Downloading {} to {}'.format(url,destination_filename))
    urllib.request.urlretrieve(url, destination_filename)  
    assert(os.path.isfile(destination_filename))
    return destination_filename


def get_temporary_filename():
    os.makedirs(ct_api_temp_dir,exist_ok=True)
    fn = os.path.join(ct_api_temp_dir,next(tempfile._get_candidate_names()))
    return fn

        
def download_to_temporary_file(url):
    return download_url(url,get_temporary_filename())


def get_missing_images(response,verbose=False):
    """
    Downloads and parses the list of submitted and processed images for a task,
    and compares them to find missing images.  Double-checks that 'failed_images'
    is a subset of the missing images.
    """
    output_file_urls = get_output_file_urls(response)
    if output_file_urls is None:
        return None
    
    # Download all three urls to temporary files
    #
    # detections, failed_images, images
    temporary_files = {}
    for s in output_file_urls.keys():
        temporary_files[s] = download_to_temporary_file(output_file_urls[s])
        
    # Load all three files
    results = {}
    for s in temporary_files.keys():
        with open(temporary_files[s]) as f:
            results[s] = json.load(f)
    
    # Diff submitted and processed images
    submitted_images = results['images']
    if verbose:
        print('Submitted {} images'.format(len(submitted_images)))
    
    detections = results['detections']
    processed_images = [detection['file'] for detection in detections['images']]
    if verbose:
        print('Received results for {} images'.format(len(processed_images)))
    
    failed_images = results['failed_images']
    if verbose:
        print('{} failed images'.format(len(failed_images)))
    
    n_failed_shards = int(response['status']['message']['num_failed_shards'])
    estimated_failed_shard_images = n_failed_shards * IMAGES_PER_SHARD
    if verbose:
        print('{} failed shards (approimately {} images)'.format(n_failed_shards,estimated_failed_shard_images))
            
    missing_images = list(set(submitted_images) - set(processed_images))
    if verbose:
        print('{} images not in results'.format(len(missing_images)))
    
    # Confirm that the failed images are a subset of the missing images
    assert len(set(failed_images) - set(missing_images)) == 0, 'Failed images should be a subset of missing images'
        
    for fn in temporary_files.values():
        os.remove(fn)
        
    return missing_images
        

def download_detection_results(endpoint_url,task_id,output_file):
    """
    Download the detection results .json file for a task
    """
    response,_ = fetch_task_status(endpoint_url,task_id)
    output_file_urls = get_output_file_urls(response)
    if output_file_urls is None:
        return None
    detection_url = output_file_urls['detections']
    download_url(detection_url,output_file)
    return response


def generate_resubmission_list(endpoint_url,task_id,resubmission_file_list_name):
    """
    Finds all the image files that failed to process in a job and writes them to a file.
    """
    response,_ = fetch_task_status(endpoint_url,task_id)
    missing_files = get_missing_images(response)
    missing_images = path_utils.find_image_strings(missing_files)
    non_images = list(set(missing_files) - set(missing_images))
    write_list_to_file(resubmission_file_list_name,missing_images)
    return missing_images,non_images


#%% Interactive driver
        
if False:

    #%%
    from api.batch_processing.data_preparation import prepare_api_submission
    
    #%%
    account_name = ''
    sas_token = 'st=...'
    container_name = ''
    rmatch = None # '^Y53'
    output_file = r'output.json'
    
    blobs = prepare_api_submission.enumerate_blobs_to_file(output_file=output_file,
                                                account_name=account_name,
                                                sas_token=sas_token,
                                                container_name=container_name,
                                                rmatch=rmatch)

    #%%
    
    file_list_json = r"D:\temp\idfg_20190801-hddrop_image_list.json"
    task_files = prepare_api_submission.divide_files_into_tasks(file_list_json)

    #%%
    
    file_list_sas_urls = [
        '','',''
    ]    
    
    input_container_sas_url = ''
    request_name_base = ''
    caller = 'blah@blah.com'
    
    request_strings,request_dicts = \
        generate_api_queries(input_container_sas_url,file_list_sas_urls,request_name_base,caller)
       
    for s in request_strings:
        print(s)
        
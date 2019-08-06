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

from azure.storage.blob import BlockBlobService

# assumes ai4eutils is on the path
from path_utils import insert_before_extension

default_n_files_per_api_task = 1000000


#%% File enumeration

def enumerate_blobs(account_name,sas_token,container_name,rmatch=None):
    """
    Enumerates blobs in a container, optionally filtering with a regex
    """
    
    print('Enumerating blobs from {}/{}'.format(account_name,container_name))
        
    block_blob_service = BlockBlobService(account_name=account_name, sas_token=sas_token)
    
    generator = block_blob_service.list_blobs(container_name)
    matched_blobs = []
    i_blob = 0
    for blob in generator:
        i_blob += 1
        if (i_blob % 1000) == 0:
            print('.',end='')
        if (i_blob % 50000) == 0:
            print('{} blobs enumerated ({} matches)'.format(i_blob,len(matched_blobs)))
        blob_name = blob.name
        if rmatch is not None:
            m = re.match(rmatch,blob_name)
            if m is None:
                continue
        matched_blobs.append(blob.name)
                
    print('Enumerated {} matching blobs (of {} total) from {}/{}'.format(len(matched_blobs),
          i_blob,account_name,container_name))

    return matched_blobs


def enumerate_blobs_to_file(output_file,account_name,sas_token,container_name,account_key=None,rmatch=None):
    """
    Enumerates to a .json string if output_file ends in ".json", otherwise enumerates to a 
    newline-delimited list.
    """        
    
    matched_blobs = enumerate_blobs(account_name=account_name,
                                    sas_token=sas_token,
                                    container_name=container_name,
                                    rmatch=rmatch)
    
    if output_file.endswith('.json'):
        s = json.dumps(matched_blobs,indent=1)
        with open(output_file,'w') as f:
            f.write(s)
    else:
        with open(output_file,'w') as f:
            for fn in matched_blobs:
                f.write(fn + '\n')
                
    print('Finished writing results for {}/{} to {}'.format(account_name,container_name,output_file))
    
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
    of strings, into a set of json files, each containing *n_files_per_task* files (the last 
    file will contain <= *n_files_per_task* files).
    
    If the input .json is blah.json, output_files will be blah.chunk001.json, blah.002.json, etc.
    
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
        output_file = insert_before_extension(file_list_json,chunk_id)
        output_files.append(output_file)
        s = json.dumps(chunk,indent=1)
        with open(output_file,'w') as f:
            f.write(s)
    
    return output_files,chunks    


def generate_api_queries(input_container_sas_url,file_list_sas_urls,request_name_base,caller):
    """
    Generate .json-formatted API input from input parameters.  file_list_sas_urls is
    a list of SAS URLs to individual file lists (all relative to the same container).
    
    request_name_base is a request name for the set; if the base name is 'blah', individual
    requests will get request names of 'blah_chunk000', 'blah_chunk001', etc.
    
    Returns both strings and Python dicts
    
    return request_strings,request_dicts
    """
    
    assert isinstance(file_list_sas_urls,list)        

    request_dicts = []
    request_strings = []
    # i_url = 0; file_list_sas_url = file_list_sas_urls[0]
    for i_url,file_list_sas_url in enumerate(file_list_sas_urls):
        
        d = {}
        d['input_container_sas'] = input_container_sas_url
        d['images_requested_json_sas'] = file_list_sas_url
        if len(file_list_sas_urls) > 0:
            chunk_id = '_chunk{0:0>3d}'.format(i_url)
            request_name = request_name_base + chunk_id
        d['request_name'] = request_name
        d['caller'] = caller
        request_dicts.append(d)
        request_strings.append(json.dumps(d,indent=1))
    
    return request_strings,request_dicts


def generate_api_query(input_container_sas_url,file_list_sas_url,request_name,caller):
    """
    Convenience function to call generate_api_queries for a single batch.
    
    See generate_api_queries, and s/lists/single items.
    """
    
    file_list_sas_urls = [file_list_sas_url]
    request_strings,request_dicts = generate_api_queries(input_container_sas_url,
                                                         file_list_sas_urls,
                                                         request_name,
                                                         caller)
    return request_strings[0],request_dicts[0]


#%% Interactive driver
        
if False:

    #%%
    from api.batch_processing.data_preparation import prepare_api_submission
    
    #%%
    account_name = ''
    sas_token = ''
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
        
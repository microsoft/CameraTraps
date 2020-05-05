#
# manage_api_submission.py
#
# Semi-automated process for submitting and managing camera trap
# API jobs.
#

#%% Imports

import os
import ntpath
import posixpath
import json
import humanfriendly
import itertools
import clipboard

from urllib.parse import urlsplit, unquote

import path_utils

from api.batch_processing.data_preparation import prepare_api_submission
from api.batch_processing.postprocessing import combine_api_outputs

from api.batch_processing.postprocessing.postprocess_batch_results import PostProcessingOptions
from api.batch_processing.postprocessing.postprocess_batch_results import process_batch_results


#%% Constants I set per job

### Required

storage_account_name = 'blah'
container_name = 'blah'
job_set_name = 'institution-20191215'
base_output_folder_name = r'f:\institution'

# These point to the same container; the read-only token is used for
# accessing images; the write-enabled token is used for writing file lists
read_only_sas_token = '?st=2019-12...'
read_write_sas_token = '?st=2019-12...'

caller = 'caller'
endpoint_base = 'http://blah.endpoint.com:6022/v2/camera-trap/detection-batch/'

### Typically left as default

container_prefix = ''

# This is how we break the container up into multiple jobs, e.g. for separate
# surveys.  The typical case is to do the whole container as a single job.
folder_names = [''] # ['folder1','folder2','folder3']

# This is only necessary if you will be performing postprocessing steps that
# don't yet support SAS URLs, specifically the "subsetting" step, or in some cases
# the splitting of files into multiple output directories for empty/animal/vehicle/people.
#
# For those applications, you will need to mount the container to a local drive.  For
# this case I recommend using rclone whether you are on Windows or Linux; rclone is much easier
# than blobfuse for transient mounting.
#
# But most of the time, you can ignore this.
image_base = 'x:\\'

additional_job_args = {}

# Supported model_versions: '4', '3', '4_prelim'
#
# Also available at the /supported_model_versions and /default_model_version endpoints
#
# Unless you have any specific reason to set this to a non-default value, leave 
# it at the default, which as of 2020.04.28 is MegaDetector 4.1
#
# additional_job_args = {"model_version":"4_prelim"}
#


#%% Derived variables, path setup

assert not (len(folder_names) == 0)

task_status_endpoint_url = endpoint_base + 'task'
submission_endpoint_url = endpoint_base + 'request_detections'

container_base_url = 'https://' + storage_account_name + '.blob.core.windows.net/' + container_name
read_only_sas_url = container_base_url + read_only_sas_token
write_sas_url = container_base_url + read_write_sas_token

filename_base = os.path.join(base_output_folder_name,job_set_name)
os.makedirs(filename_base,exist_ok=True)

raw_api_output_folder = os.path.join(filename_base,'raw_api_outputs')
os.makedirs(raw_api_output_folder,exist_ok=True)

combined_api_output_folder = os.path.join(filename_base,'combined_api_outputs')
os.makedirs(combined_api_output_folder,exist_ok=True)

postprocessing_output_folder = os.path.join(filename_base,'postprocessing')
os.makedirs(postprocessing_output_folder,exist_ok=True)

# Turn warnings into errors if more than this many images are missing
max_tolerable_missing_images = 20

# import clipboard; clipboard.copy(read_only_sas_url)
# configure mount point with rclone config
# rclone mount mountname: z:

# Not yet automated:
#
# Mounting the image source (see comment above)
#
# Submitting the jobs (code written below, but it doesn't really work)
#
# Handling failed jobs/shards/images (though most of the code exists in generate_resubmission_list)
#
# Pushing the final results to shared storage and generating a SAS URL to share with the collaborator
#
# Pushing the previews to shared storage


#%% Support functions

# https://gist.github.com/zed/c2168b9c52b032b5fb7d
def url_to_filename(url):
    
    # scheme, netloc, path, query, fragment
    urlpath = urlsplit(url).path
    
    basename = posixpath.basename(unquote(urlpath))
    if (os.path.basename(basename) != basename or
        unquote(posixpath.basename(urlpath)) != basename):
        raise ValueError  # reject '%2f' or 'dir%5Cbasename.ext' on Windows
        
    return basename


#%% Enumerate blobs to files

file_lists_by_folder = []

# folder_name = folder_names[0]
for folder_name in folder_names:
    list_file = os.path.join(filename_base,job_set_name + '_' + path_utils.clean_filename(folder_name) + '_all.json')
    
    # If this is intended to be a folder, it needs to end in '/', otherwise files that start
    # with the same string will match too
    folder_name_suffix = folder_name
    folder_name_suffix = folder_name_suffix.replace('\\','/')
    if (not len(folder_name) == 0) and (not folder_name_suffix.endswith('/')):
        folder_name_suffix = folder_name_suffix + '/'
    prefix = container_prefix + folder_name_suffix
    file_list = prepare_api_submission.enumerate_blobs_to_file(output_file=list_file,
                                    account_name=storage_account_name,sas_token=read_only_sas_token,
                                    container_name=container_name,
                                    account_key=None,
                                    rmatch=None,prefix=prefix)
    file_lists_by_folder.append(list_file)

assert len(file_lists_by_folder) == len(folder_names)


#%% Divide images into chunks for each folder

# This will be a list of lists
folder_chunks = []

# list_file = file_lists_by_folder[0]
for list_file in file_lists_by_folder:
    
    chunked_files,chunks = prepare_api_submission.divide_files_into_tasks(list_file)
    print('Divided images into files:')
    for i_fn,fn in enumerate(chunked_files):
        new_fn = chunked_files[i_fn].replace('__','_').replace('_all','')
        os.rename(fn, new_fn)
        chunked_files[i_fn] = new_fn
        print(fn,len(chunks[i_fn]))
    folder_chunks.append(chunked_files)

assert len(folder_chunks) == len(folder_names)


#%% Copy image lists to blob storage for each job

# Maps job name to a remote path
job_name_to_list_url = {}
job_names_by_task_group = []

# chunked_folder_files = folder_chunks[0]; chunk_file = chunked_folder_files[0]
for chunked_folder_files in folder_chunks:
    
    job_names_this_task_group = []
    
    for chunk_file in chunked_folder_files:
        
        job_name = job_set_name + '_' + os.path.splitext(ntpath.basename(chunk_file))[0]
        
        # periods not allowed in job names
        job_name = job_name.replace('.','_')
        
        remote_path = 'api_inputs/' + job_set_name + '/' + ntpath.basename(chunk_file)
        print('Job {}: uploading {} to {}'.format(
            job_name,chunk_file,remote_path))
        prepare_api_submission.copy_file_to_blob(storage_account_name,read_write_sas_token,
                                                     container_name,chunk_file,
                                                     remote_path)
        assert job_name not in job_name_to_list_url
        list_url = read_only_sas_url.replace('?','/' + remote_path + '?')
        job_name_to_list_url[job_name] = list_url
        job_names_this_task_group.append(job_name)    
        
    job_names_by_task_group.append(job_names_this_task_group)

    # ...for each task within this task group

# ...for each folder


#%% Generate API calls for each job

request_strings_by_task_group = []

# job_name = list(job_name_to_list_url.keys())[0]
for task_group_job_names in job_names_by_task_group:
    
    request_strings_this_task_group = []
    
    for job_name in task_group_job_names:
        list_url = job_name_to_list_url[job_name]
        s,d = prepare_api_submission.generate_api_query(read_only_sas_url,
                                                  list_url,
                                                  job_name,
                                                  caller,
                                                  additional_args=additional_job_args,
                                                  image_path_prefix=None)
        request_strings_this_task_group.append(s)
        
    request_strings_by_task_group.append(request_strings_this_task_group)

request_strings = list(itertools.chain.from_iterable(request_strings_by_task_group))

for s in request_strings:
    print(s)

clipboard.copy('\n\n'.join(request_strings))


#%% Run the jobs (still in progress, doesn't actually work yet)

# Not working yet, something is wrong with my post call

task_ids_by_task_group = []

# task_group_request_strings = request_strings_by_task_group[0]; request_string = task_group_request_strings[0]
for task_group_request_strings in request_strings_by_task_group:
    
    task_ids_this_task_group = []
    for request_string in task_group_request_strings:
                
        request_string = request_string.replace('\n','')
        # response = requests.post(submission_endpoint_url,json=request_string)
        # print(response.json())
        task_id = 0
        task_ids_this_task_group.append(task_id)
    
    task_ids_by_task_group.append(task_ids_this_task_group)
    
  # for each string in this task group

# for each task group
    
# List of task IDs, grouped by logical job
task_groups = task_ids_by_task_group


#%% Manually define task groups if we ran the jobs manually

# The nested lists will make sense below, I promise.

# For just one job...
task_groups = [[9999]]

# For multiple jobs...
task_groups = [[1111],[2222],[3333]]


#%% Estimate total time

n_images = 0
for fn in file_lists_by_folder:
    images = json.load(open(fn))
    n_images += len(images)
    
print('Processing a total of {} images'.format(n_images))

# Around 0.8s/image on 16 GPUs
expected_seconds = (0.8 / 16) * n_images
print('Expected time: {}'.format(humanfriendly.format_timespan(expected_seconds)))


#%% Status check

for task_group in task_groups:
    for task_id in task_group:
        response,status = prepare_api_submission.fetch_task_status(task_status_endpoint_url,task_id)
        assert status == 200
        print(response)


#%% Look for failed shards or missing images, start new jobs if necessary

n_resubmissions = 0

# i_task_group = 0; task_group = task_groups[i_task_group]; task_id = task_group[0]
for i_task_group,task_group in enumerate(task_groups):
    
    for task_id in task_group:
        
        response,status = prepare_api_submission.fetch_task_status(task_status_endpoint_url,task_id)
        assert status == 200
        n_failed_shards = int(response['status']['message']['num_failed_shards'])
        
        # assert n_failed_shards == 0
        
        if n_failed_shards != 0:
            print('Warning: {} failed shards for task {}'.format(n_failed_shards,task_id))
        
        output_file_urls = prepare_api_submission.get_output_file_urls(response)
        detections_url = output_file_urls['detections']
        fn = url_to_filename(detections_url)
        
        # Each task group corresponds to one of our folders
        assert (folder_names[i_task_group] in fn) or \
            (prepare_api_submission.clean_request_name(folder_names[i_task_group]) in fn)
        assert 'chunk' in fn
        
        missing_images_fn = fn.replace('.json','_missing.json')
        missing_images_fn = os.path.join(raw_api_output_folder, missing_images_fn)
        
        missing_images,non_images = \
            prepare_api_submission.generate_resubmission_list(
                task_status_endpoint_url,task_id,missing_images_fn)

        if len(missing_images) < max_tolerable_missing_images:
            continue

        print('Warning: {} missing images for task {}'.format(len(missing_images),task_id))
        
        job_name = job_set_name + '_' + folder_names[i_task_group] + '_' + str(task_id) + '_missing_images'
        remote_path = 'api_inputs/' + job_set_name + '/' + job_name + '.json'
        print('Job {}: uploading {} to {}'.format(
            job_name,missing_images_fn,remote_path))
        prepare_api_submission.copy_file_to_blob(storage_account_name,read_write_sas_token,
                                                     container_name,missing_images_fn,
                                                     remote_path)
        list_url = read_only_sas_url.replace('?','/' + remote_path + '?')                
        s,d = prepare_api_submission.generate_api_query(read_only_sas_url,
                                          list_url,
                                          job_name,
                                          caller,
                                          image_path_prefix=None)
        
        print('\nResbumission job for {}:\n'.format(task_id))
        print(s)
        n_resubmissions += 1
        
    # ...for each task
        
# ...for each task group

if n_resubmissions == 0:
    print('No resubmissions necessary')
    

#%% Resubmit jobs for failed shards, add to appropriate task groups

if False:
    
    #%%
    
    resubmission_tasks = [1222]
    for task_id in resubmission_tasks:
        response,status = prepare_api_submission.fetch_task_status(task_status_endpoint_url,task_id)
        assert status == 200
        print(response)
            
    task_groups = [[2233,9484,1222],[1197,1702,2764]]


#%% Pull results

task_id_to_results_file = {}

# i_task_group = 0; task_group = task_groups[i_task_group]; task_id = task_group[0]
for i_task_group,task_group in enumerate(task_groups):
    
    for task_id in task_group:
        
        response,status = prepare_api_submission.fetch_task_status(task_status_endpoint_url,task_id)
        assert status == 200

        output_file_urls = prepare_api_submission.get_output_file_urls(response)
        detections_url = output_file_urls['detections']
        fn = url_to_filename(detections_url)
        
        # n_failed_shards = int(response['status']['message']['num_failed_shards'])
        # assert n_failed_shards == 0
        
        # Each task group corresponds to one of our folders
        assert (folder_names[i_task_group] in fn) or \
            (prepare_api_submission.clean_request_name(folder_names[i_task_group]) in fn)
        assert 'chunk' in fn or 'missing' in fn
        
        output_file = os.path.join(raw_api_output_folder,fn)
        response = prepare_api_submission.download_url(detections_url,output_file)
        task_id_to_results_file[task_id] = output_file
        
    # ...for each task
        
# ...for each task group
 
    
#%% Combine results from task groups into final output files

folder_name_to_combined_output_file = {}

# i_folder = 0; folder_name = folder_names[i_folder]
for i_folder,folder_name_raw in enumerate(folder_names):
    
    folder_name = path_utils.clean_filename(folder_name_raw)
    print('Combining results for {}'.format(folder_name))
    
    task_group = task_groups[i_folder]
    results_files = []
    
    # task_id = task_group[0]
    for task_id in task_group:
        
        raw_output_file = task_id_to_results_file[task_id]
        results_files.append(raw_output_file)
    
    combined_api_output_file = os.path.join(combined_api_output_folder,job_set_name + 
                                            folder_name + '_detections.json')
    
    print('Combining the following into {}'.format(combined_api_output_file))
    for fn in results_files:
        print(fn)
        
    combine_api_outputs.combine_api_output_files(results_files,combined_api_output_file)
    folder_name_to_combined_output_file[folder_name] = combined_api_output_file

    # Check that we have (almost) all the images    
    list_file = file_lists_by_folder[i_folder]
    requested_images = json.load(open(list_file,'r'))
    results = json.load(open(combined_api_output_file,'r'))
    result_images = [im['file'] for im in results['images']]
    requested_images_set = set(requested_images)
    result_images_set = set(result_images)
    missing_files = requested_images_set - result_images_set
    missing_images = path_utils.find_image_strings(missing_files)
    if len(missing_images) > 0:
        print('Warning: {} missing images for folder [{}]'.format(len(missing_images),folder_name))    
    assert len(missing_images) < max_tolerable_missing_images

    # Something has gone bonkers if there are images in the results that
    # aren't in the request
    extra_images = result_images_set - requested_images_set
    assert len(extra_images) == 0
    
# ...for each folder
    
    
#%% Post-processing (no ground truth)

html_output_files = []

# i_folder = 0; folder_name_raw = folder_names[i_folder]
for i_folder,folder_name_raw in enumerate(folder_names):
    
    options = PostProcessingOptions()
    options.image_base_dir = read_only_sas_url
    options.parallelize_rendering = True
    options.include_almost_detections = True
    options.num_images_to_sample = 5000
    options.confidence_threshold = 0.8
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = None
    options.separate_detections_by_category = True
    
    folder_name = path_utils.clean_filename(folder_name_raw)
    if len(folder_name) == 0:
        folder_token = ''
    else:
        folder_token = folder_name + '_'
    output_base = os.path.join(postprocessing_output_folder,folder_token + \
        job_set_name + '_{:.3f}'.format(options.confidence_threshold))
    os.makedirs(output_base,exist_ok=True)
    print('Processing {} to {}'.format(folder_name,output_base))
    api_output_file = folder_name_to_combined_output_file[folder_name]

    options.api_output_file = api_output_file
    options.output_dir = output_base
    ppresults = process_batch_results(options)
    html_output_files.append(ppresults.output_html_file)
    
for fn in html_output_files:
    os.startfile(fn)
    
    
#%% Manual processing follows
    
#
# Everything after this should be considered mostly manual, and no longer includes
# looping over folders.    
#
    
    
#%% Repeat detection elimination, phase 1

# Deliberately leaving these imports here, rather than at the top, because this cell is not 
# typically executed
from api.batch_processing.postprocessing.repeat_detection_elimination import repeat_detections_core
import path_utils
job_index = 0

options = repeat_detections_core.RepeatDetectionOptions()

options.confidenceMin = 0.6
options.confidenceMax = 1.01 
options.iouThreshold = 0.85
options.occurrenceThreshold = 10
options.maxSuspiciousDetectionSize = 0.2

options.bRenderHtml = False
options.imageBase = read_only_sas_url
rde_string = 'rde_{:.2f}_{:.2f}_{}_{:.2f}'.format(
    options.confidenceMin,options.iouThreshold,
    options.occurrenceThreshold,options.maxSuspiciousDetectionSize)
options.outputBase = os.path.join(filename_base,rde_string)
options.filenameReplacements = {'':''}

options.debugMaxDir = -1
options.debugMaxRenderDir = -1
options.debugMaxRenderDetection = -1
options.debugMaxRenderInstance = -1

api_output_filename = list(folder_name_to_combined_output_file.values())[job_index]
filtered_output_filename = path_utils.insert_before_extension(api_output_filename,'filtered_{}'.format(rde_string))

suspiciousDetectionResults = repeat_detections_core.find_repeat_detections(api_output_filename,
                                                                           None,
                                                                           options)

clipboard.copy(os.path.dirname(suspiciousDetectionResults.filterFile))


#%% Manual RDE step

## DELETE THE ANIMALS ##


#%% Re-filtering

from api.batch_processing.postprocessing.repeat_detection_elimination import remove_repeat_detections

remove_repeat_detections.remove_repeat_detections(
    inputFile=api_output_filename,
    outputFile=filtered_output_filename,
    filteringDir=os.path.dirname(suspiciousDetectionResults.filterFile),
    options=options
    )


#%% Post-processing (post-RDE)

html_output_files = []

# i_folder = 0; folder_name_raw = folder_names[i_folder]
for i_folder,folder_name_raw in enumerate(folder_names):
    
    options = PostProcessingOptions()
    options.image_base_dir = read_only_sas_url
    options.parallelize_rendering = True
    options.include_almost_detections = True
    options.num_images_to_sample = 5000
    options.confidence_threshold = 0.8
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = None
    options.separate_detections_by_category = True
    
    folder_name = path_utils.clean_filename(folder_name_raw)
    if len(folder_name) == 0:
        folder_token = ''
    else:
        folder_token = folder_name + '_'
    output_base = os.path.join(postprocessing_output_folder,folder_token + \
        job_set_name + '_{}_{:.3f}'.format(rde_string,options.confidence_threshold))
    os.makedirs(output_base,exist_ok=True)
    print('Processing {} to {}'.format(folder_name,output_base))
    # api_output_file = folder_name_to_combined_output_file[folder_name]

    options.api_output_file = filtered_output_filename
    options.output_dir = output_base
    ppresults = process_batch_results(options)
    html_output_files.append(ppresults.output_html_file)
    
for fn in html_output_files:
    os.startfile(fn)
    

#%% Subsetting

data = None

from api.batch_processing.postprocessing.subset_json_detector_output import subset_json_detector_output
from api.batch_processing.postprocessing.subset_json_detector_output import SubsetJsonDetectorOutputOptions

input_filename = inputFilename = list(folder_name_to_combined_output_file.values())[0]
output_base = os.path.join(filename_base,'json_subsets')

folders = os.listdir(image_base)

if data is None:
    with open(input_filename) as f:
        data = json.load(f)
        
print('Data set contains {} images'.format(len(data['images'])))

# i_folder = 0; folder_name = folders[i_folder]
for i_folder,folder_name in enumerate(folders):
    
    output_filename = os.path.join(output_base,folder_name + '.json')
    print('Processing folder {} of {} ({}) to {}'.format(i_folder,len(folders),folder_name,
          output_filename))
    
    options = SubsetJsonDetectorOutputOptions()
    options.confidence_threshold = 0.4
    options.overwrite_json_files = True
    options.make_folder_relative = True
    options.query = folder_name + '\\'
    
    subset_data = subset_json_detector_output(input_filename,output_filename,options,data)

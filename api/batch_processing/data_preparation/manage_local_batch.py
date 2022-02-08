"""

    manage_local_batch.py
    
    Semi-automated process for managing a local MegaDetector job, including
    standard postprocessing steps.
    
"""

#%% Imports and constants

import sys
import json
import os
import subprocess

from datetime import date

import humanfriendly

import ai4e_azure_utils  # from ai4eutils
import path_utils        # from ai4eutils

from api.batch_processing.postprocessing.postprocess_batch_results import (
    PostProcessingOptions, process_batch_results)

max_task_name_length = 92

# Turn warnings into errors if more than this many images are missing
max_tolerable_failed_images = 100

n_rendering_threads = 50


#%% Constants I set per script

input_path = os.path.expanduser('~/data/organization/2021-12-24') 
model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb')
organization_name_short = 'organization'
postprocessing_base = os.path.expanduser('~/postprocessing')

# Number of jobs to split data into, typically equal to the number of available GPUs
n_jobs = 2
n_gpus = n_jobs

# Only used to print out a time estimate
gpu_images_per_second = 2.9

checkpoint_frequency = 10000

base_task_name = organization_name_short + '-' + date.today().strftime('%Y-%m-%d')
base_output_folder_name = os.path.join(postprocessing_base,organization_name_short)
os.makedirs(base_output_folder_name,exist_ok=True)


#%% Derived variables, path setup

# local folders
filename_base = os.path.join(base_output_folder_name, base_task_name)
combined_api_output_folder = os.path.join(filename_base, 'combined_api_outputs')
postprocessing_output_folder = os.path.join(filename_base, 'postprocessing')

os.makedirs(filename_base, exist_ok=True)
os.makedirs(combined_api_output_folder, exist_ok=True)
os.makedirs(postprocessing_output_folder, exist_ok=True)


#%% Support functions

def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])
        

#%% Enumerate files

all_images = path_utils.find_images(input_path,recursive=True)

print('Enumerated {} image files in {}'.format(len(all_images),input_path))


#%% Divide images into chunks 

def split_list(L, n):
    k, m = divmod(len(L), n)
    return list(L[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

folder_chunks = split_list(all_images,n_jobs)


#%% Estimate total time

n_images = len(all_images)
execution_seconds = n_images / gpu_images_per_second
wallclock_seconds = execution_seconds / n_gpus
print('Expected time: {}'.format(humanfriendly.format_timespan(wallclock_seconds)))

seconds_per_chunk = len(folder_chunks[0]) / gpu_images_per_second
print('Expected time per chunk: {}'.format(humanfriendly.format_timespan(seconds_per_chunk)))


#%% Write file lists

task_info = []

for i_chunk,chunk_list in enumerate(folder_chunks):
    
    chunk_fn = os.path.join(filename_base,'chunk{}.json'.format(str(i_chunk).zfill(3)))
    task_info.append({'id':i_chunk,'input_file':chunk_fn})
    ai4e_azure_utils.write_list_to_file(chunk_fn, chunk_list)
    
    
#%% Generate commands

# i_task = 0; task = task_info[i_task]
for i_task,task in enumerate(task_info):
    
    chunk_file = task['input_file']
    output_fn = chunk_file.replace('.json','_results.json')
    
    task['output_file'] = output_fn
    
    cuda_string = ''
    if n_jobs > 1:
        gpu_number = i_task % n_gpus
        cuda_string = f'CUDA_VISIBLE_DEVICES={gpu_number}'
    else:
        gpu_number = 0
    
    checkpoint_frequency_string = ''
    checkpoint_path_string = ''
    if checkpoint_frequency is not None and checkpoint_frequency > 0:
        checkpoint_frequency_string = f'--checkpoint_frequency {checkpoint_frequency}'
        checkpoint_path_string = '--checkpoint_path {}'.format(chunk_file.replace(
            '.json','_checkpoint.json'))
            
    cmd = f'{cuda_string} python run_tf_detector_batch.py {model_file} {chunk_file} {output_fn} {checkpoint_frequency_string} {checkpoint_path_string}'
    
    cmd_file = os.path.join(filename_base,'run_chunk_{}_gpu_{}.sh'.format(str(i_task).zfill(2),
                            str(gpu_number).zfill(2)))
    
    with open(cmd_file,'w') as f:
        f.write(cmd + '\n')
        
    import stat
    st = os.stat(cmd_file)
    os.chmod(cmd_file, st.st_mode | stat.S_IEXEC)
    
    task['command'] = cmd
    task['command_file'] = cmd_file


#%% Generate combined commands for a handful of tasks

if False:

    #%%    

    task_set = [8,10,12,14,16]; gpu_number = 0; sleep_time_between_tasks = 60; sleep_time_before_tasks = 0
    commands = []
    
    # i_task = 8
    for i_task in task_set:
        
        if i_task == task_set[0]:
            commands.append('sleep {}'.format(str(sleep_time_before_tasks)))            
        
        task = task_info[i_task]
        chunk_file = task['input_file']
        output_fn = chunk_file.replace('.json','_results.json')
        
        task['output_file'] = output_fn
        
        if gpu_number is not None:
            cuda_string = f'CUDA_VISIBLE_DEVICES={gpu_number}'
        else:
            cuda_string = ''
            gpu_number = 0
        
        checkpoint_frequency_string = ''
        checkpoint_path_string = ''
        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            checkpoint_frequency_string = f'--checkpoint_frequency {checkpoint_frequency}'
            checkpoint_path_string = '--checkpoint_path {}'.format(chunk_file.replace(
                '.json','_checkpoint.json'))
                
        cmd = f'{cuda_string} python run_tf_detector_batch.py {model_file} {chunk_file} {output_fn} {checkpoint_frequency_string} {checkpoint_path_string}'
        
        task['command'] = cmd
        commands.append(cmd)
        if i_task != task_set[-1]:
            commands.append('sleep {}'.format(str(sleep_time_between_tasks)))            
        
    # ...for each task
    
    task_strings = [str(k).zfill(2) for k in task_set]
    task_set_string = '_'.join(task_strings)
    cmd_file = os.path.join(filename_base,'run_chunk_{}_gpu_{}.sh'.format(task_set_string,
                            str(gpu_number).zfill(2)))
    
    with open(cmd_file,'w') as f:
        for cmd in commands:
            f.write(cmd + '\n')
        
    import stat
    st = os.stat(cmd_file)
    os.chmod(cmd_file, st.st_mode | stat.S_IEXEC)
    

#%% Run the tasks

# Prefer to run manually


#%% Load results, look for failed or missing images in each task

n_total_failures = 0

# i_task = 0; task = task_info[i_task]
for i_task,task in enumerate(task_info):
    
    chunk_file = task['input_file']
    output_file = task['output_file']
    
    with open(chunk_file,'r') as f:
        task_images = json.load(f)
    with open(output_file,'r') as f:
        task_results = json.load(f)
    
    task_images_set = set(task_images)
    filename_to_results = {}
    
    n_task_failures = 0
    
    # im = task_results['images'][0]
    for im in task_results['images']:
        assert im['file'].startswith(input_path)
        assert im['file'] in task_images_set
        filename_to_results[im['file']] = im
        if 'failure' in im:
            assert im['failure'] is not None
            n_task_failures += 1
    
    task['n_failures'] = n_task_failures
    task['results'] = task_results
    
    for fn in task_images:
        assert fn in filename_to_results
    
    n_total_failures += n_task_failures

# ...for each task

assert n_total_failures < max_tolerable_failed_images,\
    '{} failures (max tolerable set to {})'.format(n_total_failures,
                                                   max_tolerable_failed_images)

print('Processed all {} images with {} failures'.format(
    len(all_images),n_total_failures))
        
        
#%% Merge results files and make images relative

import copy

combined_results = None

for i_task,task in enumerate(task_info):

    if i_task == 0:
        combined_results = copy.deepcopy(task['results'])
        combined_results['images'] = copy.deepcopy(task['results']['images'])
        continue
    task_results = task['results']
    assert task_results['info']['format_version'] == combined_results['info']['format_version']
    assert task_results['detection_categories'] == combined_results['detection_categories']
    combined_results['images'].extend(copy.deepcopy(task_results['images']))
    
assert len(combined_results['images']) == len(all_images)

result_filenames = [im['file'] for im in combined_results['images']]
assert len(combined_results['images']) == len(set(result_filenames))

# im = combined_results['images'][0]
for im in combined_results['images']:
    assert im['file'].startswith(input_path + '/')
    im['file']= im['file'].replace(input_path + '/','',1)    
    
combined_api_output_file = os.path.join(
    combined_api_output_folder,
    '{}_detections.json'.format(base_task_name))

with open(combined_api_output_file,'w') as f:
    json.dump(combined_results,f,indent=2)

print('Wrote results to {}'.format(combined_api_output_file))


#%% Post-processing (no ground truth)

render_animals_only = False

options = PostProcessingOptions()
options.image_base_dir = input_path
options.parallelize_rendering = True
options.include_almost_detections = True
options.num_images_to_sample = 7500
options.parallelize_rendering_n_cores = n_rendering_threads
options.confidence_threshold = 0.8
options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
options.ground_truth_json_file = None
options.separate_detections_by_category = True
# options.sample_seed = 0

if render_animals_only:
    # Omit some pages from the output, useful when animals are rare
    options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                     'detections_person_vehicle','non_detections']

output_base = os.path.join(postprocessing_output_folder,
    base_task_name + '_{:.3f}'.format(options.confidence_threshold))
if render_animals_only:
    output_base = output_base + '_animals_only'

os.makedirs(output_base, exist_ok=True)
print('Processing to {}'.format(output_base))

options.api_output_file = combined_api_output_file
options.output_dir = output_base
ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file
open_file(html_output_file)


#%% RDE (sample directory collapsing)

def remove_overflow_folders(relativePath):
    
    import re
    
    # In this example, the camera created folders called "100EK113", "101EK113", etc., for every N images
    pat = '\/\d+EK\d+\/'
    
    relativePath = relativePath.replace('\\','/')    
    relativePath = re.sub(pat,'/',relativePath)
    dirName = os.path.dirname(relativePath)
    
    return dirName

if False:
    
    relativePath = 'a/b/c/d/100EK113/blah.jpg'
    print(remove_overflow_folders(relativePath))


#%% Repeat detection elimination, phase 1

folder_name_to_filtered_output_filename = {}

# Deliberately leaving these imports here, rather than at the top, because this cell is not
# typically executed
from api.batch_processing.postprocessing.repeat_detection_elimination import repeat_detections_core
import path_utils
task_index = 0

options = repeat_detections_core.RepeatDetectionOptions()

options.confidenceMin = 0.6
options.confidenceMax = 1.01
options.iouThreshold = 0.85
options.occurrenceThreshold = 10
options.maxSuspiciousDetectionSize = 0.2

# To invoke custom collapsing of folders for a particular manufacturer's naming scheme
# options.customDirNameFunction = remove_overflow_folders

options.bRenderHtml = False
options.imageBase = input_path
rde_string = 'rde_{:.2f}_{:.2f}_{}_{:.2f}'.format(
    options.confidenceMin, options.iouThreshold,
    options.occurrenceThreshold, options.maxSuspiciousDetectionSize)
options.outputBase = os.path.join(filename_base, rde_string + '_task_{}'.format(task_index))
options.filenameReplacements = {'':''}

# Exclude people and vehicles from RDE
# options.excludeClasses = [2,3]

options.debugMaxDir = -1
options.debugMaxRenderDir = -1
options.debugMaxRenderDetection = -1
options.debugMaxRenderInstance = -1

suspiciousDetectionResults = repeat_detections_core.find_repeat_detections(combined_api_output_file,
                                                                           None,
                                                                           options)

# import clipboard; clipboard.copy(os.path.dirname(suspiciousDetectionResults.filterFile))
# open_file(os.path.dirname(suspiciousDetectionResults.filterFile))


#%% Manual RDE step

## DELETE THE VALID DETECTIONS ##


#%% Re-filtering

from api.batch_processing.postprocessing.repeat_detection_elimination import remove_repeat_detections

filtered_output_filename = path_utils.insert_before_extension(combined_api_output_file, 'filtered_{}'.format(rde_string))

remove_repeat_detections.remove_repeat_detections(
    inputFile=combined_api_output_file,
    outputFile=filtered_output_filename,
    filteringDir=os.path.dirname(suspiciousDetectionResults.filterFile)
    )


#%% Post-processing (post-RDE)

render_animals_only = False

options = PostProcessingOptions()
options.image_base_dir = input_path
options.parallelize_rendering = True
options.include_almost_detections = True
options.num_images_to_sample = 7500
options.confidence_threshold = 0.8
options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
options.ground_truth_json_file = None
options.separate_detections_by_category = True
# options.sample_seed = 0

if render_animals_only:
    # Omit some pages from the output, useful when animals are rare
    options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                      'detections_person_vehicle','non_detections']    

output_base = os.path.join(postprocessing_output_folder, 
    base_task_name + '_{}_{:.3f}'.format(rde_string, options.confidence_threshold))    

if render_animals_only:
    output_base = output_base + '_render_animals_only'
os.makedirs(output_base, exist_ok=True)

print('Processing post-RDE to {}'.format(output_base))

options.api_output_file = filtered_output_filename
options.output_dir = output_base
ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file

open_file(html_output_file)


#%% Scrap

# ...and so ends the process for 90% of jobs; the remaining cells are things
# we do for special cases, but often enough to keep the code handy.


#%% Create a new category for large boxes

from api.batch_processing.postprocessing import categorize_detections_by_size

options = categorize_detections_by_size.SizeCategorizationOptions()
options.threshold = 0.85
input_file = r"g:\organization\file.json"
size_separated_file = input_file.replace('.json','-size-separated-{}.json'.format(options.threshold))
d = categorize_detections_by_size.categorize_detections_by_size(input_file,size_separated_file,options)


#%% Subsetting

data = None

from api.batch_processing.postprocessing.subset_json_detector_output import (
    subset_json_detector_output, SubsetJsonDetectorOutputOptions)

input_filename = filtered_output_filename
output_base = os.path.join(filename_base,'json_subsets')

folders = os.listdir(input_path)

if data is None:
    with open(input_filename) as f:
        data = json.load(f)

print('Data set contains {} images'.format(len(data['images'])))

# i_folder = 0; folder_name = folders[i_folder]
for i_folder, folder_name in enumerate(folders):

    output_filename = os.path.join(output_base, folder_name + '.json')
    print('Processing folder {} of {} ({}) to {}'.format(i_folder, len(folders), folder_name,
          output_filename))

    options = SubsetJsonDetectorOutputOptions()
    options.confidence_threshold = 0.4
    options.overwrite_json_files = True
    options.make_folder_relative = True
    options.query = folder_name + '\\'

    subset_data = subset_json_detector_output(input_filename, output_filename, options, data)


#%% String replacement
    
data = None

from api.batch_processing.postprocessing.subset_json_detector_output import (
    subset_json_detector_output, SubsetJsonDetectorOutputOptions)

input_filename = filtered_output_filename
output_filename = input_filename.replace('.json','_replaced.json')

options = SubsetJsonDetectorOutputOptions()
options.query = folder_name + '/'
options.replacement = ''
subset_json_detector_output(input_filename,output_filename,options)


#%% Folder splitting

from api.batch_processing.postprocessing.separate_detections_into_folders import (
    separate_detections_into_folders, SeparateDetectionsIntoFoldersOptions)

default_threshold = 0.8
base_output_folder = r'e:\{}-{}-separated'.format(base_task_name,default_threshold)

options = SeparateDetectionsIntoFoldersOptions(default_threshold)

options.results_file = filtered_output_filename
options.base_input_folder = input_path
options.base_output_folder = os.path.join(base_output_folder,folder_name)
options.n_threads = 100
options.allow_existing_directory = False

separate_detections_into_folders(options)

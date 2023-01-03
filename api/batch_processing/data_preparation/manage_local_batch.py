"""

    manage_local_batch.py
    
    Semi-automated process for managing a local MegaDetector job, including
    standard postprocessing steps.
    
"""

#%% Imports and constants

import json
import os
import stat
import time

import humanfriendly

from tqdm import tqdm

# from ai4eutils
import ai4e_azure_utils 
import path_utils

from detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file
from detection.run_detector import DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD

from api.batch_processing.postprocessing.postprocess_batch_results import (
    PostProcessingOptions, process_batch_results)
from detection.run_detector import get_detector_version_from_filename

max_task_name_length = 92

# To specify a non-default confidence threshold for including detections in the .json file
json_threshold = None

# Turn warnings into errors if more than this many images are missing
max_tolerable_failed_images = 100

n_rendering_threads = 50

use_image_queue = False

# Only relevant when we're using a single GPU
default_gpu_number = 0

quiet_mode = True

# Specify a target image size when running MD... strongly recommended to leave this at "None"
image_size = None

# Only relevant when running on CPU
ncores = 1

# OS-specific script line continuation character
slcc = '\\'

# OS-specific script comment character
scc = '#' 

if os.name == 'nt':
    slcc = '^'
    scc = 'REM'


#%% Constants I set per script

input_path = '/datadrive/organization/data'

organization_name_short = 'organization'
job_date = None # '2022-12-02'
assert job_date is not None and organization_name_short != 'organization'

# Optional descriptor
job_tag = None

if job_tag is None:
    job_description_string = ''
else:
    job_description_string = '-' + job_tag

model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
# model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5b.0.0.pt')
# model_file = os.path.expanduser('~/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb')

postprocessing_base = os.path.expanduser('~/postprocessing')

# Number of jobs to split data into, typically equal to the number of available GPUs
n_jobs = 2
n_gpus = 2

# Only used to print out a time estimate
if ('v5') in model_file:
    gpu_images_per_second = 10
else:
    gpu_images_per_second = 2.9

checkpoint_frequency = 10000

base_task_name = organization_name_short + '-' + job_date + job_description_string + '-' + get_detector_version_from_filename(model_file)
base_output_folder_name = os.path.join(postprocessing_base,organization_name_short)
os.makedirs(base_output_folder_name,exist_ok=True)


#%% Derived variables, path setup

filename_base = os.path.join(base_output_folder_name, base_task_name)
combined_api_output_folder = os.path.join(filename_base, 'combined_api_outputs')
postprocessing_output_folder = os.path.join(filename_base, 'preview')

os.makedirs(filename_base, exist_ok=True)
os.makedirs(combined_api_output_folder, exist_ok=True)
os.makedirs(postprocessing_output_folder, exist_ok=True)

if input_path.endswith('/'):
    input_path = input_path[0:-1]

print('Output folder:\n{}'.format(filename_base))


#%% Enumerate files

all_images = path_utils.find_images(input_path,recursive=True)

print('Enumerated {} image files in {}'.format(len(all_images),input_path))

if False:

    pass 
    
    #%% Load files from prior enumeration
    
    import re    
    chunk_files = os.listdir(filename_base)
    pattern = re.compile('chunk\d+.json')
    chunk_files = [fn for fn in chunk_files if pattern.match(fn)]
    all_images = []
    for fn in chunk_files:
        with open(os.path.join(filename_base,fn),'r') as f:
            chunk = json.load(f)
            assert isinstance(chunk,list)
            all_images.extend(chunk)
    print('Loaded {} image files from chunks in {}'.format(len(all_images),filename_base))
    

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
    
    if n_jobs > 1:
        gpu_number = i_task % n_gpus        
    else:
        gpu_number = default_gpu_number
        
    cuda_string = f'CUDA_VISIBLE_DEVICES={gpu_number}'
    
    checkpoint_frequency_string = ''
    checkpoint_path_string = ''
    checkpoint_filename = chunk_file.replace('.json','_checkpoint.json')
    
    if checkpoint_frequency is not None and checkpoint_frequency > 0:
        checkpoint_frequency_string = f'--checkpoint_frequency {checkpoint_frequency}'
        checkpoint_path_string = '--checkpoint_path "{}"'.format(checkpoint_filename)
            
    use_image_queue_string = ''
    if (use_image_queue):
        use_image_queue_string = '--use_image_queue'

    ncores_string = ''
    if (ncores > 1):
        ncores_string = '--ncores {}'.format(ncores)
        
    quiet_string = ''
    if quiet_mode:
        quiet_string = '--quiet'

    image_size_string = ''
    if image_size is not None:
        image_size_string = '--image_size {}'.format(image_size)
        
    # Generate the script to run MD
        
    cmd = f'{cuda_string} python run_detector_batch.py "{model_file}" "{chunk_file}" "{output_fn}" {checkpoint_frequency_string} {checkpoint_path_string} {use_image_queue_string} {ncores_string} {quiet_string} {image_size_string}'
    
    cmd_file = os.path.join(filename_base,'run_chunk_{}_gpu_{}.sh'.format(str(i_task).zfill(2),
                            str(gpu_number).zfill(2)))
    
    with open(cmd_file,'w') as f:
        f.write(cmd + '\n')
    
    st = os.stat(cmd_file)
    os.chmod(cmd_file, st.st_mode | stat.S_IEXEC)
    
    task['command'] = cmd
    task['command_file'] = cmd_file

    # Generate the script to resume from the checkpoint
    
    resume_string = ' --resume_from_checkpoint "{}"'.format(checkpoint_filename)
    resume_cmd = cmd + resume_string
    resume_cmd_file = os.path.join(filename_base,'resume_chunk_{}_gpu_{}.sh'.format(str(i_task).zfill(2),
                            str(gpu_number).zfill(2)))
    
    with open(resume_cmd_file,'w') as f:
        f.write(resume_cmd + '\n')
    
    st = os.stat(resume_cmd_file)
    os.chmod(resume_cmd_file, st.st_mode | stat.S_IEXEC)
    
    task['resume_command'] = resume_cmd
    task['resume_command_file'] = resume_cmd_file


#%% Run the tasks

"""
I strongly prefer to manually run the scripts we just generated, but this cell demonstrates
how one would invoke run_detector_batch programmatically.  Normally when I run manually on 
a multi-GPU machine, I run the scripts in N separate shells, one for each GPU.  This programmatic
approach does not yet know how to do something like that, so all chunks will run serially.
This is a no-op if you're on a single-GPU machine.
"""

if False:
    
    #%%% Run the tasks (commented out)

    # i_task = 0; task = task_info[i_task]
    for i_task,task in enumerate(task_info):
    
        chunk_file = task['input_file']
        output_fn = task['output_file']
        
        checkpoint_filename = chunk_file.replace('.json','_checkpoint.json')
        
        if json_threshold is not None:
            confidence_threshold = json_threshold
        else:
            confidence_threshold = DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
            
        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            cp_freq_arg = checkpoint_frequency
        else:
            cp_freq_arg = -1
            
        start_time = time.time()
        results = load_and_run_detector_batch(model_file=model_file, 
                                              image_file_names=chunk_file, 
                                              checkpoint_path=checkpoint_filename, 
                                              confidence_threshold=confidence_threshold,
                                              checkpoint_frequency=cp_freq_arg, 
                                              results=None,
                                              n_cores=ncores, 
                                              use_image_queue=use_image_queue,
                                              quiet=quiet_mode,
                                              image_size=image_size)        
        elapsed = time.time() - start_time
        
        print('Task {}: finished inference for {} images in {}'.format(
            i_task, len(results),humanfriendly.format_timespan(elapsed)))

        # This will write absolute paths to the file, we'll fix this later
        write_results_to_file(results, output_fn, detector_file=model_file)

        if checkpoint_frequency is not None and checkpoint_frequency > 0:
            if os.path.isfile(checkpoint_filename):                
                os.remove(checkpoint_filename)
                print('Deleted checkpoint file {}'.format(checkpoint_filename))
                
    # ...for each chunk
    
# ...if False

    
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
    
assert len(combined_results['images']) == len(all_images), \
    'Expected {} images in combined results, found {}'.format(
        len(all_images),len(combined_results['images']))

result_filenames = [im['file'] for im in combined_results['images']]
assert len(combined_results['images']) == len(set(result_filenames))

# im = combined_results['images'][0]
for im in combined_results['images']:
    assert im['file'].startswith(input_path + os.path.sep)
    im['file']= im['file'].replace(input_path + os.path.sep,'',1)    
    
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
options.confidence_threshold = 0.2
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
path_utils.open_file(html_output_file)


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

    pass

    #%%
    
    relativePath = 'a/b/c/d/100EK113/blah.jpg'
    print(remove_overflow_folders(relativePath))
    
    #%%
    
    with open(combined_api_output_file,'r') as f:
        d = json.load(f)
    image_filenames = [im['file'] for im in d['images']]
    
    #%%
    
    dirNames = set()
    
    # relativePath = image_filenames[0]
    for relativePath in tqdm(image_filenames):
        dirName = remove_overflow_folders(relativePath)
        dirNames.add(dirName)
        
    dirNames = list(dirNames)
    dirNames.sort()


#%% Repeat detection elimination, phase 1

# Deliberately leaving these imports here, rather than at the top, because this
# cell is not typically executed
from api.batch_processing.postprocessing.repeat_detection_elimination import repeat_detections_core
import path_utils
task_index = 0

options = repeat_detections_core.RepeatDetectionOptions()

options.confidenceMin = 0.15
options.confidenceMax = 1.01
options.iouThreshold = 0.85
options.occurrenceThreshold = 10
options.maxSuspiciousDetectionSize = 0.2
# options.minSuspiciousDetectionSize = 0.05

# options.parallelizationUsesThreads = True
# options.nWorkers = 10

# This will cause a very light gray box to get drawn around all the detections
# we're *not* considering as suspicious.
options.bRenderOtherDetections = True
options.otherDetectionsThreshold = options.confidenceMin

# options.lineThickness = 5
# options.boxExpansion = 8

# To invoke custom collapsing of folders for a particular manufacturer's naming scheme
# options.customDirNameFunction = remove_overflow_folders

options.bRenderHtml = False
options.imageBase = input_path
rde_string = 'rde_{:.2f}_{:.2f}_{}_{:.2f}'.format(
    options.confidenceMin, options.iouThreshold,
    options.occurrenceThreshold, options.maxSuspiciousDetectionSize)
options.outputBase = os.path.join(filename_base, rde_string + '_task_{}'.format(task_index))
options.filenameReplacements = None # {'':''}

# Exclude people and vehicles from RDE
# options.excludeClasses = [2,3]

# options.maxImagesPerFolder = 50000
# options.includeFolders = ['a/b/c']
# options.excludeFolder = ['a/b/c']

options.debugMaxDir = -1
options.debugMaxRenderDir = -1
options.debugMaxRenderDetection = -1
options.debugMaxRenderInstance = -1

# Can be None, 'xsort', or 'clustersort'
options.smartSort = 'xsort'

suspiciousDetectionResults = repeat_detections_core.find_repeat_detections(combined_api_output_file,
                                                                           None,
                                                                           options)

# import clipboard; clipboard.copy(os.path.dirname(suspiciousDetectionResults.filterFile))
# path_utils.open_file(os.path.dirname(suspiciousDetectionResults.filterFile))


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
options.confidence_threshold = 0.2
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

path_utils.open_file(html_output_file)


#%% Run MegaClassifier (actually, write out a script that runs MegaClassifier)

classifier_name_short = 'megaclassifier'
threshold_str = '0.15' # 0.6
classifier_name = 'megaclassifier_v0.1_efficientnet-b3'

organization_name = organization_name_short
job_name = base_task_name
input_filename = filtered_output_filename # combined_api_output_file
input_files = [input_filename]
image_base = input_path
crop_path = os.path.join(os.path.expanduser('~/crops'),job_name + '_crops')
output_base = combined_api_output_folder
device_id = 0

output_file = os.path.join(filename_base,'run_{}_'.format(classifier_name_short) + job_name +  '.sh')

classifier_base = os.path.expanduser('~/models/camera_traps/megaclassifier/v0.1/')
assert os.path.isdir(classifier_base)

checkpoint_path = os.path.join(classifier_base,'v0.1_efficientnet-b3_compiled.pt')
assert os.path.isfile(checkpoint_path)

classifier_categories_path = os.path.join(classifier_base,'v0.1_index_to_name.json')
assert os.path.isfile(classifier_categories_path)

target_mapping_path = os.path.join(classifier_base,'idfg_to_megaclassifier_labels.json')
assert os.path.isfile(target_mapping_path)

classifier_output_suffix = '_megaclassifier_output.csv.gz'
final_output_suffix = '_megaclassifier.json'

n_threads_str = '50'
image_size_str = '300'
batch_size_str = '64'
num_workers_str = '8'
classification_threshold_str = '0.05'

logdir = filename_base

# This is just passed along to the metadata in the output file, it has no impact
# on how the classification scripts run.
typical_classification_threshold_str = '0.75'

##%% Set up environment

commands = []
# commands.append('cd CameraTraps/classification\n')
# commands.append('conda activate cameratraps-classifier\n')

##%% Crop images

commands.append('\n' + scc + ' Cropping ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    crop_cmd = ''
    
    crop_comment = '\n' + scc + ' Cropping {}\n'.format(fn)
    crop_cmd += crop_comment
    
    crop_cmd += "python crop_detections.py " + slcc + "\n" + \
    	 ' ' + input_file_path + ' ' + slcc + '\n' + \
         ' ' + crop_path + ' ' + slcc + '\n' + \
         ' ' + '--images-dir "' + image_base + '"' + ' ' + slcc + '\n' + \
         ' ' + '--threshold "' + threshold_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--square-crops ' + ' ' + slcc + '\n' + \
         ' ' + '--threads "' + n_threads_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--logdir "' + logdir + '"' + '\n' + \
         ' ' + '\n'
    crop_cmd = '{}'.format(crop_cmd)
    commands.append(crop_cmd)


##%% Run classifier

commands.append('\n' + scc + ' Classifying ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    
    classify_cmd = ''
    
    classify_comment = '\n' + scc + ' Classifying {}\n'.format(fn)
    classify_cmd += classify_comment
    
    classify_cmd += "python run_classifier.py " + slcc + "\n" + \
    	 ' ' + checkpoint_path + ' ' + slcc + '\n' + \
         ' ' + crop_path + ' ' + slcc + '\n' + \
         ' ' + classifier_output_path + ' ' + slcc + '\n' + \
         ' ' + '--detections-json "' + input_file_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--classifier-categories "' + classifier_categories_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--image-size "' + image_size_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--batch-size "' + batch_size_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--num-workers "' + num_workers_str + '"' + ' ' + slcc + '\n'
    
    if device_id is not None:
        classify_cmd += ' ' + '--device {}'.format(device_id)
        
    classify_cmd += '\n\n'        
    classify_cmd = '{}'.format(classify_cmd)
    commands.append(classify_cmd)
		

##%% Remap classifier outputs

commands.append('\n' + scc + ' Remapping ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    classifier_output_path_remapped = \
        classifier_output_path.replace(".csv.gz","_remapped.csv.gz")
    assert not (classifier_output_path == classifier_output_path_remapped)
    
    output_label_index = classifier_output_path_remapped.replace(
        "_remapped.csv.gz","_label_index_remapped.json")
                                       
    remap_cmd = ''
    
    remap_comment = '\n' + scc + ' Remapping {}\n'.format(fn)
    remap_cmd += remap_comment
    
    remap_cmd += "python aggregate_classifier_probs.py " + slcc + "\n" + \
        ' ' + classifier_output_path + ' ' + slcc + '\n' + \
        ' ' + '--target-mapping "' + target_mapping_path + '"' + ' ' + slcc + '\n' + \
        ' ' + '--output-csv "' + classifier_output_path_remapped + '"' + ' ' + slcc + '\n' + \
        ' ' + '--output-label-index "' + output_label_index + '"' \
        '\n'
     
    remap_cmd = '{}'.format(remap_cmd)
    commands.append(remap_cmd)
    

##%% Merge classification and detection outputs

commands.append('\n' + scc + ' Merging ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    
    classifier_output_path_remapped = \
        classifier_output_path.replace(".csv.gz","_remapped.csv.gz")
    
    output_label_index = classifier_output_path_remapped.replace(
        "_remapped.csv.gz","_label_index_remapped.json")
    
    final_output_path = os.path.join(output_base,
                                     os.path.basename(classifier_output_path)).\
        replace(classifier_output_suffix,
        final_output_suffix)
    final_output_path = final_output_path.replace('_detections','')
    final_output_path = final_output_path.replace('_crops','')
    final_output_path_mc = final_output_path
    
    merge_cmd = ''
    
    merge_comment = '\n' + scc + ' Merging {}\n'.format(fn)
    merge_cmd += merge_comment
    
    merge_cmd += "python merge_classification_detection_output.py " + slcc + "\n" + \
    	 ' ' + classifier_output_path_remapped + ' ' + slcc + '\n' + \
         ' ' + output_label_index + ' ' + slcc + '\n' + \
         ' ' + '--output-json "' + final_output_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--detection-json "' + input_file_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--classifier-name "' + classifier_name + '"' + ' ' + slcc + '\n' + \
         ' ' + '--threshold "' + classification_threshold_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--typical-confidence-threshold "' + typical_classification_threshold_str + '"' + '\n' + \
         '\n'
    merge_cmd = '{}'.format(merge_cmd)
    commands.append(merge_cmd)


##%% Write  out classification script

with open(output_file,'w') as f:
    for s in commands:
        f.write('{}'.format(s))

import stat
st = os.stat(output_file)
os.chmod(output_file, st.st_mode | stat.S_IEXEC)


#%% Run a non-MegaClassifier classifier (i.e., a classifier with no output mapping)

classifier_name_short = 'idfgclassifier'
threshold_str = '0.15' # 0.6
classifier_name = 'idfg_classifier_ckpt_14_compiled'

organization_name = organization_name_short
job_name = base_task_name
input_filename = filtered_output_filename # combined_api_output_file
input_files = [input_filename]
image_base = input_path
crop_path = os.path.join(os.path.expanduser('~/crops'),job_name + '_crops')
output_base = combined_api_output_folder
device_id = 1

output_file = os.path.join(filename_base,'run_{}_'.format(classifier_name_short) + job_name +  '.sh')

classifier_base = os.path.expanduser('~/models/camera_traps/idfg_classifier/idfg_classifier_20200905_042558')
assert os.path.isdir(classifier_base)

checkpoint_path = os.path.join(classifier_base,'idfg_classifier_ckpt_14_compiled.pt')
assert os.path.isfile(checkpoint_path)

classifier_categories_path = os.path.join(classifier_base,'label_index.json')
assert os.path.isfile(classifier_categories_path)

classifier_output_suffix = '_{}_output.csv.gz'.format(classifier_name_short)
final_output_suffix = '_{}.json'.format(classifier_name_short)

threshold_str = '0.65'
n_threads_str = '50'
image_size_str = '300'
batch_size_str = '64'
num_workers_str = '8'
logdir = filename_base

classification_threshold_str = '0.05'

# This is just passed along to the metadata in the output file, it has no impact
# on how the classification scripts run.
typical_classification_threshold_str = '0.75'


##%% Set up environment

commands = []


##%% Crop images
    
commands.append('\n' + scc + ' Cropping ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    crop_cmd = ''
    
    crop_comment = '\n' + scc + ' Cropping {}\n'.format(fn)
    crop_cmd += crop_comment
    
    crop_cmd += "python crop_detections.py " + slcc + "\n" + \
    	 ' ' + input_file_path + ' ' + slcc + '\n' + \
         ' ' + crop_path + ' ' + slcc + '\n' + \
         ' ' + '--images-dir "' + image_base + '"' + ' ' + slcc + '\n' + \
         ' ' + '--threshold "' + threshold_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--square-crops ' + ' ' + slcc + '\n' + \
         ' ' + '--threads "' + n_threads_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--logdir "' + logdir + '"' + '\n' + \
         '\n'
    crop_cmd = '{}'.format(crop_cmd)
    commands.append(crop_cmd)


##%% Run classifier

commands.append('\n' + scc + ' Classifying ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    
    classify_cmd = ''
    
    classify_comment = '\n' + scc + ' Classifying {}\n'.format(fn)
    classify_cmd += classify_comment
    
    classify_cmd += "python run_classifier.py " + slcc + "\n" + \
    	 ' ' + checkpoint_path + ' ' + slcc + '\n' + \
         ' ' + crop_path + ' ' + slcc + '\n' + \
         ' ' + classifier_output_path + ' ' + slcc + '\n' + \
         ' ' + '--detections-json "' + input_file_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--classifier-categories "' + classifier_categories_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--image-size "' + image_size_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--batch-size "' + batch_size_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--num-workers "' + num_workers_str + '"' + ' ' + slcc + '\n'
    
    if device_id is not None:
        classify_cmd += ' ' + '--device {}'.format(device_id)
        
    classify_cmd += '\n\n'    
    classify_cmd = '{}'.format(classify_cmd)
    commands.append(classify_cmd)
		

##%% Merge classification and detection outputs

commands.append('\n' + scc + ' Merging ' + scc + '\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    final_output_path = os.path.join(output_base,
                                     os.path.basename(classifier_output_path)).\
                                     replace(classifier_output_suffix,
                                     final_output_suffix)
    final_output_path = final_output_path.replace('_detections','')
    final_output_path = final_output_path.replace('_crops','')
    final_output_path_ic = final_output_path
    
    merge_cmd = ''
    
    merge_comment = '\n' + scc + ' Merging {}\n'.format(fn)
    merge_cmd += merge_comment
    
    merge_cmd += "python merge_classification_detection_output.py " + slcc + "\n" + \
    	 ' ' + classifier_output_path + ' ' + slcc + '\n' + \
         ' ' + classifier_categories_path + ' ' + slcc + '\n' + \
         ' ' + '--output-json "' + final_output_path_ic + '"' + ' ' + slcc + '\n' + \
         ' ' + '--detection-json "' + input_file_path + '"' + ' ' + slcc + '\n' + \
         ' ' + '--classifier-name "' + classifier_name + '"' + ' ' + slcc + '\n' + \
         ' ' + '--threshold "' + classification_threshold_str + '"' + ' ' + slcc + '\n' + \
         ' ' + '--typical-confidence-threshold "' + typical_classification_threshold_str + '"' + '\n' + \
         '\n'
    merge_cmd = '{}'.format(merge_cmd)
    commands.append(merge_cmd)


##%% Write everything out

with open(output_file,'w') as f:
    for s in commands:
        f.write('{}'.format(s))

import stat
st = os.stat(output_file)
os.chmod(output_file, st.st_mode | stat.S_IEXEC)


#%% Within-image classification smoothing

# Only count detections with a classification confidence threshold above
# *classification_confidence_threshold*, which in practice means we're only
# looking at one category per detection.
#
# If an image has at least *min_detections_above_threshold* such detections
# in the most common category, and no more than *max_detections_secondary_class*
# in the second-most-common category, flip all detections to the most common
# category.
#
# Optionally treat some classes as particularly unreliable, typically used to overwrite an 
# "other" class.

classification_detection_files = [    
    final_output_path_mc,
    final_output_path_ic    
    ]

assert all([os.path.isfile(fn) for fn in classification_detection_files])

# Only count detections with a classification confidence threshold above
# *classification_confidence_threshold*, which in practice means we're only
# looking at one category per detection.
#
# If an image has at least *min_detections_above_threshold* such detections
# in the most common category, and no more than *max_detections_secondary_class*
# in the second-most-common category, flip all detections to the most common
# category.
#
# Optionally treat some classes as particularly unreliable, typically used to overwrite an 
# "other" class.

smoothed_classification_files = []

for final_output_path in classification_detection_files:

    classifier_output_path = final_output_path
    classifier_output_path_within_image_smoothing = classifier_output_path.replace(
        '.json','_within_image_smoothing.json')
    
    with open(classifier_output_path,'r') as f:
        d = json.load(f)
    
    # d['classification_categories']
    
    # im['detections']
    
    # path_utils.open_file(os.path.join(input_path,im['file']))
    
    from collections import defaultdict
    
    min_detections_above_threshold = 4
    max_detections_secondary_class = 3
    
    min_detections_to_overwrite_other = 2
    other_category_names = ['other']
    
    classification_confidence_threshold = 0.6
    
    # Which classifications should we even bother over-writing?
    classification_overwrite_threshold = 0.3 # classification_confidence_threshold
    
    # Detection confidence threshold for things we count
    detection_confidence_threshold = 0.2
    
    # Which detections should we even bother over-writing?
    detection_overwrite_threshold = 0.05
        
    category_name_to_id = {d['classification_categories'][k]:k for k in d['classification_categories']}
    other_category_ids = []
    for s in other_category_names:
        if s in category_name_to_id:
            other_category_ids.append(category_name_to_id[s])
        else:
            print('Warning: "other" category {} not present in file {}'.format(
                s,classifier_output_path))
    
    n_other_classifications_changed = 0
    n_other_images_changed = 0
    
    n_detections_flipped = 0
    n_images_changed = 0
    
    # im = d['images'][0]    
    for im in tqdm(d['images']):
        
        if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
            continue
        
        detections = im['detections']
    
        category_to_count = defaultdict(int)
        for det in detections:
            if ('classifications' in det) and (det['conf'] >= detection_confidence_threshold):
                for c in det['classifications']:
                    if c[1] >= classification_confidence_threshold:
                        category_to_count[c[0]] += 1
                # ...for each classification
            # ...if there are classifications for this detection
        # ...for each detection
                        
        if len(category_to_count) <= 1:
            continue
        
        category_to_count = {k: v for k, v in sorted(category_to_count.items(),
                                                     key=lambda item: item[1], 
                                                     reverse=True)}
        
        keys = list(category_to_count.keys())
        
        # Handle a quirky special case: if the most common category is "other" and 
        # it's "tied" with the second-most-common category, swap them
        if (len(keys) > 1) and \
            (keys[0] in other_category_ids) and \
            (keys[1] not in other_category_ids) and \
            (category_to_count[keys[0]] == category_to_count[keys[1]]):
                keys[1], keys[0] = keys[0], keys[1]
        
        max_count = category_to_count[keys[0]]
        # secondary_count = category_to_count[keys[1]]
        # The 'secondary count' is the most common non-other class
        secondary_count = 0
        for i_key in range(1,len(keys)):
            if keys[i_key] not in other_category_ids:
                secondary_count = category_to_count[keys[i_key]]
                break

        most_common_category = keys[0]
        
        assert max_count >= secondary_count
        
        # If we have at least *min_detections_to_overwrite_other* in a category that isn't
        # "other", change all "other" classifications to that category
        if max_count >= min_detections_to_overwrite_other and \
            most_common_category not in other_category_ids:
            
            other_change_made = False
            
            for det in detections:
                
                if ('classifications' in det) and (det['conf'] >= detection_overwrite_threshold): 
                    
                    for c in det['classifications']:                
                        
                        if c[1] >= classification_overwrite_threshold and \
                            c[0] in other_category_ids:
                                
                            n_other_classifications_changed += 1
                            other_change_made = True
                            c[0] = most_common_category
                            
                    # ...for each classification
                    
                # ...if there are classifications for this detection
                
            # ...for each detection
            
            if other_change_made:
                n_other_images_changed += 1
            
        # ...if we should overwrite all "other" classifications
    
        if max_count < min_detections_above_threshold:
            continue
        
        if secondary_count >= max_detections_secondary_class:
            continue
        
        # At this point, we know we have a dominant category; change all other above-threshold
        # classifications to that category.  That category may have been "other", in which
        # case we may have already made the relevant changes.
        
        n_detections_flipped_this_image = 0
        
        # det = detections[0]
        for det in detections:
            
            if ('classifications' in det) and (det['conf'] >= detection_overwrite_threshold):
                
                for c in det['classifications']:
                    if c[1] >= classification_overwrite_threshold and \
                        c[0] != most_common_category:
                            
                        c[0] = most_common_category
                        n_detections_flipped += 1
                        n_detections_flipped_this_image += 1
                
                # ...for each classification
                
            # ...if there are classifications for this detection
            
        # ...for each detection
        
        if n_detections_flipped_this_image > 0:
            n_images_changed += 1
    
    # ...for each image    
    
    print('Classification smoothing: changed {} detections on {} images'.format(
        n_detections_flipped,n_images_changed))
    
    print('"Other" smoothing: changed {} detections on {} images'.format(
          n_other_classifications_changed,n_other_images_changed))
    
    with open(classifier_output_path_within_image_smoothing,'w') as f:
        json.dump(d,f,indent=2)
        
    print('Wrote results to:\n{}'.format(classifier_output_path_within_image_smoothing))
    smoothed_classification_files.append(classifier_output_path_within_image_smoothing)

# ...for each file we want to smooth


#%% Post-processing (post-classification)

classification_detection_files = smoothed_classification_files
    
assert all([os.path.isfile(fn) for fn in classification_detection_files])
    
# classification_detection_file = classification_detection_files[1]
for classification_detection_file in classification_detection_files:
    
    options = PostProcessingOptions()
    options.image_base_dir = input_path
    options.parallelize_rendering = True
    options.include_almost_detections = True
    options.num_images_to_sample = 10000
    options.confidence_threshold = 0.2
    options.classification_confidence_threshold = 0.75
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = None
    options.separate_detections_by_category = True
    
    folder_token = classification_detection_file.split(os.path.sep)[-1].replace('classifier.json','')
    
    output_base = os.path.join(postprocessing_output_folder, folder_token + \
        base_task_name + '_{:.3f}'.format(options.confidence_threshold))
    os.makedirs(output_base, exist_ok=True)
    print('Processing {} to {}'.format(base_task_name, output_base))
    
    options.api_output_file = classification_detection_file
    options.output_dir = output_base
    ppresults = process_batch_results(options)
    path_utils.open_file(ppresults.output_html_file)


#%% Zip .json files

json_files = os.listdir(combined_api_output_folder)
json_files = [fn for fn in json_files if fn.endswith('.json')]
json_files = [os.path.join(combined_api_output_folder,fn) for fn in json_files]

import zipfile
from zipfile import ZipFile

output_path = combined_api_output_folder

def zip_json_file(fn):
    
    assert fn.endswith('.json')
    basename = os.path.basename(fn)
    zip_file_name = os.path.join(output_path,basename + '.zip')
    print('Zipping {} to {}'.format(fn,zip_file_name))
    
    with ZipFile(zip_file_name,'w',zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(fn,arcname=basename,compresslevel=9,compress_type=zipfile.ZIP_DEFLATED)

from multiprocessing.pool import ThreadPool
pool = ThreadPool(len(json_files))
with tqdm(total=len(json_files)) as pbar:
    for i,_ in enumerate(pool.imap_unordered(zip_json_file,json_files)):
        pbar.update()


#%% 99.9% of jobs end here

# Everything after this is run ad hoc and/or requires some manual editing.


#%% Compare results files for different model versions (or before/after RDE)

import itertools

from api.batch_processing.postprocessing.compare_batch_results import (
    BatchComparisonOptions,PairwiseBatchComparisonOptions,compare_batch_results)

options = BatchComparisonOptions()

options.job_name = organization_name_short
options.output_folder = os.path.join(postprocessing_output_folder,'model_comparison')
options.image_folder = input_path

options.pairwise_options = []

filenames = [
    '/postprocessing/organization/mdv4_results.json',
    '/postprocessing/organization/mdv5a_results.json',
    '/postprocessing/organization/mdv5b_results.json'    
    ]

detection_thresholds = [0.7,0.15,0.15]

assert len(detection_thresholds) == len(filenames)

rendering_thresholds = [(x*0.6666) for x in detection_thresholds]

# Choose all pairwise combinations of the files in [filenames]
for i, j in itertools.combinations(list(range(0,len(filenames))),2):
        
    pairwise_options = PairwiseBatchComparisonOptions()
    
    pairwise_options.results_filename_a = filenames[i]
    pairwise_options.results_filename_b = filenames[j]
    
    pairwise_options.rendering_confidence_threshold_a = rendering_thresholds[i]
    pairwise_options.rendering_confidence_threshold_b = rendering_thresholds[j]
    
    pairwise_options.detection_thresholds_a = {'animal':detection_thresholds[i],
                                               'person':detection_thresholds[i],
                                               'vehicle':detection_thresholds[i]}
    pairwise_options.detection_thresholds_b = {'animal':detection_thresholds[j],
                                               'person':detection_thresholds[j],
                                               'vehicle':detection_thresholds[j]}
    options.pairwise_options.append(pairwise_options)

results = compare_batch_results(options)

from path_utils import open_file # from ai4eutils
open_file(results.html_output_file)


#%% Merge in high-confidence detections from another results file

from api.batch_processing.postprocessing.merge_detections import MergeDetectionsOptions,merge_detections

source_files = ['']
target_file = ''
output_file = target_file.replace('.json','_merged.json')

options = MergeDetectionsOptions()
options.max_detection_size = 1.0
options.target_confidence_threshold = 0.25
options.categories_to_include = [1]
options.source_confidence_thresholds = [0.2]
merge_detections(source_files, target_file, output_file, options)

merged_detections_file = output_file


#%% Create a new category for large boxes

from api.batch_processing.postprocessing import categorize_detections_by_size

size_options = categorize_detections_by_size.SizeCategorizationOptions()

# This is a size threshold, not a confidence threshold
size_options.threshold = 0.9
size_options.output_category_name = 'large_detections'
# size_options.categories_to_separate = [3]
size_options.measurement = 'size' # 'width'

input_file = filtered_output_filename
size_separated_file = input_file.replace('.json','-size-separated-{}.json'.format(
    size_options.threshold))
d = categorize_detections_by_size.categorize_detections_by_size(input_file,size_separated_file,
                                                                size_options)


#%% Preview large boxes

output_base_large_boxes = os.path.join(postprocessing_output_folder, 
    base_task_name + '_{}_{:.3f}_large_boxes'.format(rde_string, options.confidence_threshold))    
os.makedirs(output_base_large_boxes, exist_ok=True)
print('Processing post-RDE, post-size-separation to {}'.format(output_base_large_boxes))

options.api_output_file = size_separated_file
options.output_dir = output_base_large_boxes

ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file
path_utils.open_file(html_output_file)


#%% .json splitting

data = None

from api.batch_processing.postprocessing.subset_json_detector_output import (
    subset_json_detector_output, SubsetJsonDetectorOutputOptions)

input_filename = filtered_output_filename
output_base = os.path.join(filename_base,'json_subsets')

if False:
    if data is None:
        with open(input_filename) as f:
            data = json.load(f)
    print('Data set contains {} images'.format(len(data['images'])))

print('Processing file {} to {}'.format(input_filename,output_base))          

options = SubsetJsonDetectorOutputOptions()
# options.query = None
# options.replacement = None

options.split_folders = True
options.make_folder_relative = True

# Reminder: 'n_from_bottom' with a parameter of zero is the same as 'bottom'
options.split_folder_mode = 'bottom'  # 'top', 'n_from_top', 'n_from_bottom'
options.split_folder_param = 0
options.overwrite_json_files = False
options.confidence_threshold = 0.01

subset_data = subset_json_detector_output(input_filename, output_base, options, data)


#%% Custom splitting/subsetting

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
    options.confidence_threshold = 0.01
    options.overwrite_json_files = True
    options.query = folder_name + '/'

    # This doesn't do anything in this case, since we're not splitting folders
    # options.make_folder_relative = True        
    
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


#%% Splitting images into folders

from api.batch_processing.postprocessing.separate_detections_into_folders import (
    separate_detections_into_folders, SeparateDetectionsIntoFoldersOptions)

default_threshold = 0.2
base_output_folder = os.path.expanduser('~/data/{}-{}-separated'.format(base_task_name,default_threshold))

options = SeparateDetectionsIntoFoldersOptions(default_threshold)

options.results_file = filtered_output_filename
options.base_input_folder = input_path
options.base_output_folder = os.path.join(base_output_folder,folder_name)
options.n_threads = 100
options.allow_existing_directory = False

separate_detections_into_folders(options)


#%% Generate commands for a subset of tasks

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

    cuda_string = f'CUDA_VISIBLE_DEVICES={gpu_number}'
    
    checkpoint_frequency_string = ''
    checkpoint_path_string = ''
    if checkpoint_frequency is not None and checkpoint_frequency > 0:
        checkpoint_frequency_string = f'--checkpoint_frequency {checkpoint_frequency}'
        checkpoint_path_string = '--checkpoint_path {}'.format(chunk_file.replace(
            '.json','_checkpoint.json'))
            
    use_image_queue_string = ''
    if (use_image_queue):
        use_image_queue_string = '--use_image_queue'

    ncores_string = ''
    if (ncores > 1):
        ncores_string = '--ncores {}'.format(ncores)
                
    quiet_string = ''
    if quiet_mode:
        quiet_string = '--quiet'
        
    cmd = f'{cuda_string} python run_detector_batch.py {model_file} {chunk_file} {output_fn} {checkpoint_frequency_string} {checkpoint_path_string} {use_image_queue_string} {ncores_string} {quiet_string}'
                    
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


#%% End notebook: turn this script into a notebook (how meta!)

import os
import nbformat as nbf

input_py_file = os.path.expanduser('~/git/CameraTraps/api/batch_processing/data_preparation/manage_local_batch.py')
assert os.path.isfile(input_py_file)
output_ipynb_file = input_py_file.replace('.py','.ipynb')

nb_header = '# Managing a local MegaDetector batch'

nb_header += '\n'

nb_header += \
"""
This notebook represents an interactive process for running MegaDetector on large batches of images, including typical and optional postprocessing steps.  Everything after "Merge results..." is basically optional, and we typically do a mix of these optional steps, depending on the job.

This notebook is auto-generated from manage_local_batch.py (a cell-delimited .py file that is used the same way, typically in Spyder or VS Code).    

"""

with open(input_py_file,'r') as f:
    lines = f.readlines()

nb = nbf.v4.new_notebook()
nb['cells'].append(nbf.v4.new_markdown_cell(nb_header))

i_line = 0

# Exclude everything before the first cell
while(not lines[i_line].startswith('#%%')):
    i_line += 1

current_cell = []

def write_code_cell(c):
    
    first_non_empty_line = None
    last_non_empty_line = None
    
    for i_code_line,code_line in enumerate(c):
        if len(code_line.strip()) > 0:
            if first_non_empty_line is None:
                first_non_empty_line = i_code_line
            last_non_empty_line = i_code_line
            
    # Remove the first [first_non_empty_lines] from the list
    c = c[first_non_empty_line:]
    last_non_empty_line -= first_non_empty_line
    c = c[:last_non_empty_line+1]
    
    nb['cells'].append(nbf.v4.new_code_cell('\n'.join(c)))
        
while(True):    
            
    line = lines[i_line].rstrip()
    
    if 'end notebook' in line.lower():
        break
    
    if lines[i_line].startswith('#%% '):
        if len(current_cell) > 0:
            write_code_cell(current_cell)
            current_cell = []
        markdown_content = line.replace('#%%','##')
        nb['cells'].append(nbf.v4.new_markdown_cell(markdown_content))
    else:
        current_cell.append(line)

    i_line += 1

# Add the last cell
write_code_cell(current_cell)

nbf.write(nb,output_ipynb_file)

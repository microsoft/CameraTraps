#%% Constants and imports

import os
import path_utils

model_filename = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
image_size = 1280 * 1.3
conf_thres = '0.001'
batch_size = 1
device_string = '0'

augment = True

if augment:
    task_name = 'augment'    
else:
    task_name = 'no-augment'

output_folder_base = os.path.expanduser('~/tmp/yolov5-inference-experiments')
input_folder = os.path.expanduser('~/data/yolov5-inference-test-images')

output_folder = os.path.join(output_folder_base,task_name)

assert os.path.isdir(input_folder)
assert os.path.isfile(model_filename)

os.makedirs(output_folder,exist_ok=True)


#%% MegaDetector command

md_output_file = os.path.join(output_folder_base,'md-output-standard.json')
cmd = 'python run_detector_batch.py "{}" "{}" "{}"'.format(
    model_filename,input_folder,md_output_file)
cmd += ' --recursive --output_relative_filenames --quiet'

# import clipboard; clipboard.copy(cmd); print(cmd)


#%% Create symlinks to give a unique ID to each image

image_files_absolute = path_utils.find_images(input_folder,recursive=True)

symlink_dir = os.path.join(output_folder,'symlink_images')
os.makedirs(symlink_dir,exist_ok=True)

image_id_to_file = {}

def safe_create_link(link_exists,link_new):
    
    if os.path.exists(link_new):
        assert os.path.islink(link_new)
        if not os.readlink(link_new) == link_exists:
            os.remove(link_new)
            os.symlink(link_exists,link_new)
    else:
        os.symlink(link_exists,link_new)

# i_image = 0; image_fn = image_files_absolute[i_image]
for i_image,image_fn in enumerate(image_files_absolute):
    
    ext = os.path.splitext(image_fn)[1]
    
    image_id_string = str(i_image).zfill(10)
    image_id_to_file[image_id_string] = image_fn
    symlink_name = image_id_string + ext
    symlink_full_path = os.path.join(symlink_dir,symlink_name)
    safe_create_link(image_fn,symlink_full_path)
    
# ...for each image


#%% Create the dataset file

# These are one less than the MD convention
yolo_category_id_to_name = {0:'animal',1:'person',2:'vehicle'}
 
from detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP
for category_id in yolo_category_id_to_name:
    assert DEFAULT_DETECTOR_LABEL_MAP[str(category_id+1)] == \
        yolo_category_id_to_name[category_id]

# Category IDs need to be continuous integers starting at 0
category_ids = sorted(list(yolo_category_id_to_name.keys()))
assert category_ids[0] == 0
assert len(category_ids) == 1 + category_ids[-1]

dataset_file = os.path.join(output_folder,'dataset.yaml')

with open(dataset_file,'w') as f:
    f.write('path: {}\n'.format(symlink_dir))
    f.write('train: .\n')
    f.write('val: .\n')
    f.write('test: .\n')
    f.write('\n')
    f.write('nc: {}\n'.format(len(yolo_category_id_to_name)))
    f.write('\n')
    f.write('names:\n')
    for category_id in category_ids:
        assert isinstance(category_id,int)
        f.write('  {}: {}\n'.format(category_id,yolo_category_id_to_name[category_id]))


#%% Prepare YOLOv5 command

cmd = 'python val.py --data "{}"'.format(dataset_file)
cmd += ' --weights "{}"'.format(model_filename)
cmd += ' --batch-size {} --imgsz {} --conf-thres {} --task test'.format(
    batch_size,image_size,conf_thres)
cmd += ' --device "{}" --save-json'.format(device_string)
cmd += ' --project "{}" --name "{}" --exist-ok'.format(output_folder,'yolo_results')

if augment:
    cmd += ' --augment'


#%% Run YOLOv5 command
    
yolov5_folder = os.path.expanduser('~/git/yolov5')

os.chdir(yolov5_folder)

import subprocess
result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    

#%% Convert results to MD format

import glob
yolo_results_folder = os.path.join(output_folder,'yolo_results')
json_files = glob.glob(yolo_results_folder + '/*.json')
assert len(json_files) == 1

yolo_json_file = json_files[0]
md_json_file = os.path.join(output_folder,'md-output.json')

from data_management import yolo_output_to_md_output

image_id_to_relative_path = {}
for image_id in image_id_to_file:
    fn = image_id_to_file[image_id]
    assert input_folder in fn
    relative_path = os.path.relpath(fn,input_folder)
    image_id_to_relative_path[image_id] = relative_path
    
yolo_output_to_md_output.yolo_json_output_to_md_output(
    yolo_json_file=yolo_json_file,
    image_folder=input_folder,
    output_file=md_json_file,
    yolo_category_id_to_name=yolo_category_id_to_name,
    detector_name=os.path.basename(model_filename),
    image_id_to_relative_path=image_id_to_relative_path)


#%% Preview results

postprocessing_output_folder = os.path.join(output_folder,'preview')

import path_utils
import json

from api.batch_processing.postprocessing.postprocess_batch_results import (
    PostProcessingOptions, process_batch_results)

with open(md_json_file,'r') as f:
    d = json.load(f)

base_task_name = os.path.basename(md_json_file)

options = PostProcessingOptions()
options.image_base_dir = input_folder
options.include_almost_detections = True
options.num_images_to_sample = None
options.confidence_threshold = 0.05
options.almost_detection_confidence_threshold = options.confidence_threshold - 0.025
options.ground_truth_json_file = None
options.separate_detections_by_category = True
# options.sample_seed = 0

options.parallelize_rendering = True
options.parallelize_rendering_n_cores = 16
options.parallelize_rendering_with_threads = False

output_base = os.path.join(postprocessing_output_folder,
    base_task_name + '_{:.3f}'.format(options.confidence_threshold))

os.makedirs(output_base, exist_ok=True)
print('Processing to {}'.format(output_base))

options.api_output_file = md_json_file
options.output_dir = output_base
ppresults = process_batch_results(options)
html_output_file = ppresults.output_html_file

path_utils.open_file(html_output_file)

# ...for each prediction file


#%% Compare results

import itertools

from api.batch_processing.postprocessing.compare_batch_results import (
    BatchComparisonOptions,PairwiseBatchComparisonOptions,compare_batch_results)

options = BatchComparisonOptions()

options.job_name = 'yolov5-inference-comparison'
options.output_folder = os.path.join(output_folder_base,'model_comparison')
options.image_folder = input_folder

options.pairwise_options = []

filenames = [
    '/home/user/tmp/yolov5-inference-experiments/md-output-standard.json',
    '/home/user/tmp/yolov5-inference-experiments/augment/md-output.json',
    '/home/user/tmp/yolov5-inference-experiments/no-augment/md-output.json'
    ]

descriptions = ['MDv5a','YOLO w/augment','YOLO no augment']

if False:
    results = []
    
    for fn in filenames:
        with open(fn,'r') as f:
            d = json.load(f)
        results.append(d)
    
detection_thresholds = [0.15,0.15,0.15]

assert len(detection_thresholds) == len(filenames)

rendering_thresholds = [(x*0.6666) for x in detection_thresholds]

# Choose all pairwise combinations of the files in [filenames]
for i, j in itertools.combinations(list(range(0,len(filenames))),2):
        
    pairwise_options = PairwiseBatchComparisonOptions()
    
    pairwise_options.results_filename_a = filenames[i]
    pairwise_options.results_filename_b = filenames[j]
    
    pairwise_options.results_description_a = descriptions[i]
    pairwise_options.results_description_b = descriptions[j]
    
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


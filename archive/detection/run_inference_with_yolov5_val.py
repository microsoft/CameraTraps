# 
# run_inference_with_yolov5_val.py
#
# Runs a folder of images through MegaDetector (or another YOLOv5 model) with YOLOv5's 
# val.py, converting the output to the standard MD format.  The main goal is to leverage
# YOLO's test-time augmentation tools.
#
# YOLOv5's val.py uses each file's base name as a unique identifier, which doesn't work 
# when you have typical camera trap images like:
#
# a/b/c/RECONYX0001.JPG
# d/e/f/RECONYX0001.JPG
#
# ...so this script jumps through a bunch of hoops to put a symlinks in a flat
# folder, run YOLOv5 on that folder, and map the results back to the real files.
#
# Currently requires the user to supply the path where a working YOLOv5 install lives,
# and assumes that the current conda environment is all set up for YOLOv5.
#
# TODO:
#
# * Figure out what happens when images are corrupted... right now this is the #1
#   reason not to use this script, it may be the case that corrupted images look the
#   same as empty images.
#
# * Multiple GPU support
#
# * Checkpointing
#
# * Windows support (I have no idea what all the symlink operations will do on Windows)
#
# * Support alternative class names at the command line (currently defaults to MD classes,
#   though other class names can be supplied programmatically)
#

#%% Imports

import os
import uuid
import glob
import tempfile
import path_utils
import process_utils
import shutil
import json

from detection.run_detector import DEFAULT_DETECTOR_LABEL_MAP
from data_management import yolo_output_to_md_output


#%% Options class

class YoloInferenceOptions:
        
    ## Required ##
    
    input_folder = None
    model_filename = None
    yolo_working_folder = None
    output_file = None

    ## Optional ##
    
    image_size = 1280 * 1.3
    conf_thres = '0.001'
    batch_size = 1
    device_string = '0'
    augment = True

    symlink_folder = None
    yolo_results_folder = None
    
    remove_symlink_folder = True
    remove_yolo_results_folder = True
    
    yolo_category_id_to_name = {0:'animal',1:'person',2:'vehicle'}
            
    
#%% Main function

def run_inference_with_yolo_val(options):

    ##%% Path handling
    
    assert os.path.isdir(options.input_folder) or os.path.isfile(options.input_folder), \
        'Could not find input {}'.format(options.input_folder)
    assert os.path.isdir(options.yolo_working_folder), \
        'Could not find working folder {}'.format(options.yolo_working_folder)
    assert os.path.isfile(options.model_filename), \
        'Could not find model file {}'.format(options.model_filename)
    
    os.makedirs(os.path.dirname(options.output_file),exist_ok=True)
    
    temporary_folder = None
    symlink_folder_is_temp_folder = False
    yolo_folder_is_temp_folder = False
    
    def get_job_temporary_folder(tf):
        if tf is not None:
            return tf
        tempdir_base = tempfile.gettempdir()
        tf = os.path.join(tempdir_base,'md_to_yolo','md_to_yolo_' + str(uuid.uuid1()))
        os.makedirs(tf,exist_ok=True)
        return tf
        
    symlink_folder = options.symlink_folder
    yolo_results_folder = options.yolo_results_folder
    
    if symlink_folder is None:
        temporary_folder = get_job_temporary_folder(temporary_folder)
        symlink_folder = os.path.join(temporary_folder,'symlinks')
        symlink_folder_is_temp_folder = True
    
    if yolo_results_folder is None:
        temporary_folder = get_job_temporary_folder(temporary_folder)
        yolo_results_folder = os.path.join(temporary_folder,'yolo_results')
        yolo_folder_is_temp_folder = True
        
    os.makedirs(symlink_folder,exist_ok=True)
    os.makedirs(yolo_results_folder,exist_ok=True)
    

    ##%% Enumerate images
    
    if os.path.isdir(options.input_folder):
        image_files_absolute = path_utils.find_images(options.input_folder,recursive=True)
    else:
        assert os.path.isfile(options.input_folder)
        with open(options.input_folder,'r') as f:            
            image_files_absolute = json.load(f)
            assert isinstance(image_files_absolute,list)
            for fn in image_files_absolute:
                assert os.path.isfile(fn), 'Could not find image file {}'.format(fn)
    
    
    ##%% Create symlinks to give a unique ID to each image
    
    image_id_to_file = {}    
    
    # i_image = 0; image_fn = image_files_absolute[i_image]
    for i_image,image_fn in enumerate(image_files_absolute):
        
        ext = os.path.splitext(image_fn)[1]
        
        image_id_string = str(i_image).zfill(10)
        image_id_to_file[image_id_string] = image_fn
        symlink_name = image_id_string + ext
        symlink_full_path = os.path.join(symlink_folder,symlink_name)
        path_utils.safe_create_link(image_fn,symlink_full_path)
        
    # ...for each image


    ##%% Create the dataset file
    
    if False:
        for category_id in options.yolo_category_id_to_name:
            assert DEFAULT_DETECTOR_LABEL_MAP[str(category_id+1)] == \
                options.yolo_category_id_to_name[category_id]
        
    # Category IDs need to be continuous integers starting at 0
    category_ids = sorted(list(options.yolo_category_id_to_name.keys()))
    assert category_ids[0] == 0
    assert len(category_ids) == 1 + category_ids[-1]
    
    dataset_file = os.path.join(yolo_results_folder,'dataset.yaml')
    
    with open(dataset_file,'w') as f:
        f.write('path: {}\n'.format(symlink_folder))
        f.write('train: .\n')
        f.write('val: .\n')
        f.write('test: .\n')
        f.write('\n')
        f.write('nc: {}\n'.format(len(options.yolo_category_id_to_name)))
        f.write('\n')
        f.write('names:\n')
        for category_id in category_ids:
            assert isinstance(category_id,int)
            f.write('  {}: {}\n'.format(category_id,
                                        options.yolo_category_id_to_name[category_id]))


    ##%% Prepare YOLOv5 command
    
    image_size_string = str(round(options.image_size))
    cmd = 'python val.py --data "{}"'.format(dataset_file)
    cmd += ' --weights "{}"'.format(options.model_filename)
    cmd += ' --batch-size {} --imgsz {} --conf-thres {} --task test'.format(
        options.batch_size,image_size_string,options.conf_thres)
    cmd += ' --device "{}" --save-json'.format(options.device_string)
    cmd += ' --project "{}" --name "{}" --exist-ok'.format(yolo_results_folder,'yolo_results')
    
    if options.augment:
        cmd += ' --augment'
    

    ##%% Run YOLOv5 command
    
    current_dir = os.getcwd()
    os.chdir(options.yolo_working_folder)    
    _ = process_utils.execute_and_print(cmd)
    os.chdir(current_dir)
        
    
    ##%% Convert results to MD format
    
    json_files = glob.glob(yolo_results_folder + '/yolo_results/*.json')
    assert len(json_files) == 1    
    yolo_json_file = json_files[0]

    image_id_to_relative_path = {}
    for image_id in image_id_to_file:
        fn = image_id_to_file[image_id]
        if os.path.isdir(options.input_folder):
            assert options.input_folder in fn
            relative_path = os.path.relpath(fn,options.input_folder)
        else:
            assert os.path.isfile(options.input_folder)
            # We'll use the absolute path as a relative path, and pass '/'
            # as the base path in this case.
            relative_path = fn
        image_id_to_relative_path[image_id] = relative_path
        
    if os.path.isdir(options.input_folder):
        image_base = options.input_folder
    else:
        assert os.path.isfile(options.input_folder)
        image_base = '/'
        
    yolo_output_to_md_output.yolo_json_output_to_md_output(
        yolo_json_file=yolo_json_file,
        image_folder=image_base,
        output_file=options.output_file,
        yolo_category_id_to_name=options.yolo_category_id_to_name,
        detector_name=os.path.basename(options.model_filename),
        image_id_to_relative_path=image_id_to_relative_path)


    ##%% Clean up
    
    if options.remove_symlink_folder:
        shutil.rmtree(symlink_folder)
    elif symlink_folder_is_temp_folder:
        print('Warning: using temporary symlink folder {}, but not removing it'.format(
            symlink_folder))
        
    if options.remove_yolo_results_folder:
        shutil.rmtree(yolo_results_folder)
    elif yolo_folder_is_temp_folder:
        print('Warning: using temporary YOLO results folder {}, but not removing it'.format(
            yolo_results_folder))
        
    if options.remove_yolo_results_folder and \
        options.remove_symlink_folder and \
        temporary_folder is not None:
            
        pass
    
# ...def run_inference_with_yolo_val()


#%% Command-line driver

import argparse,sys
from ct_utils import args_to_object

def main():
    
    options = YoloInferenceOptions()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_filename',type=str,
        help='model file name')
    parser.add_argument(
        'input_folder',type=str,
        help='folder on which to recursively run the model, or a .json list of filenames')
    parser.add_argument(
        'output_file',type=str,
        help='.json file where output will be written')
    parser.add_argument(
        'yolo_working_folder',type=str,
        help='folder in which to execute val.py')
    
    parser.add_argument(
        '--image_size', default=options.image_size, type=int,
        help='image size for model execution (default {})'.format(
            options.image_size))
    parser.add_argument(
        '--conf_thres', default=options.conf_thres, type=float,
        help='confidence threshold for including detections in the output file (default {})'.format(
            options.conf_thres))
    parser.add_argument(
        '--batch_size', default=options.batch_size, type=int,
        help='inference batch size (default {})'.format(options.batch_size))
    parser.add_argument(
        '--device_string', default=options.device_string, type=str,
        help='CUDA device specifier, e.g. "0" or "cpu" (default {})'.format(options.device_string))
    
    parser.add_argument(
        '--symlink_folder', type=str,
        help='temporary folder for symlinks')
    parser.add_argument(
        '--yolo_results_folder', type=str,
        help='temporary folder for YOLO intermediate output')
    
    parser.add_argument(
        '--no_remove_symlink_folder', action='store_true',
        help='don\'t remove the temporary folder full of symlinks')
    parser.add_argument(
        '--no_remove_yolo_results_folder', action='store_true',
        help='don\'t remove the temporary folder full of YOLO intermediate files')
    
    if options.augment:
        default_augment_enabled = 1
    else:
        default_augment_enabled = 0
    parser.add_argument(
        '--augment_enabled', default=default_augment_enabled, type=int,
        help='enable/disable augmentation (default {})'.format(default_augment_enabled))
        
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    
    args_to_object(args, options)
    options.remove_symlink_folder = (not options.no_remove_symlink_folder)
    options.remove_yolo_results_folder = (not options.no_remove_yolo_results_folder)
    options.augment = (options.augment_enabled > 0)        
            
    print(options.__dict__)
    
    run_inference_with_yolo_val(options)
    

if __name__ == '__main__':
    main()


#%% Scrap

if False:
    
    #%% Test driver (folder)
    
    project_name = ''
    input_folder = os.path.expanduser(f'~/data/{project_name}')
    output_folder = os.path.expanduser('~/tmp/{project_name}')
    model_filename = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
    yolo_working_folder = os.path.expanduser('~/git/yolov5')
    model_name = os.path.splitext(os.path.basename(model_filename))[0]
        
    options = YoloInferenceOptions()
    
    options.yolo_working_folder = yolo_working_folder
    
    options.output_file = os.path.join(output_folder,'{}_{}-md_format.json'.format(
        project_name,model_name))
    
    options.image_size = 1280 * 1.3
    options.conf_thres = '0.001'
    options.batch_size = 1
    options.device_string = '0'
    options.augment = True

    options.input_folder = input_folder
    options.model_filename = model_filename
    
    options.yolo_results_folder = None # os.path.join(output_file + '_yolo.json')        
    options.symlink_folder = None # os.path.join(output_folder,'symlinks')
    output_file = None
    
    options.remove_temporary_symlink_folder = True
    options.remove_yolo_results_file = True
    

    #%% Test driver (file)
    
    input_folder = '/home/user/postprocessing/test/test-2023-04-18-v5a.0.0/chunk001.json'
    output_file = '/home/user/postprocessing/test/test-2023-04-18-v5a.0.0/chunk001_results.json'
    
    model_filename = os.path.expanduser('~/models/camera_traps/megadetector/md_v5.0.0/md_v5a.0.0.pt')
    yolo_working_folder = os.path.expanduser('~/git/yolov5')
    model_name = os.path.splitext(os.path.basename(model_filename))[0]
        
    options = YoloInferenceOptions()
    
    options.yolo_working_folder = yolo_working_folder
    
    options.output_file = output_file
    
    options.image_size = 1280 * 1.3
    options.conf_thres = '0.001'
    options.batch_size = 1
    options.device_string = '0'
    options.augment = True

    options.input_folder = input_folder
    options.model_filename = model_filename
    
    options.yolo_results_folder = '/home/user/postprocessing/test/test-2023-04-18-v5a.0.0/yolo_results/yolo_results_001'
    options.symlink_folder = '/home/user/postprocessing/test/test-2023-04-18-v5a.0.0/symlinks/symlinks_001'
    
    options.remove_temporary_symlink_folder = False
    options.remove_yolo_results_file = False
    
    
    #%% Preview results
    
    postprocessing_output_folder = os.path.join(output_folder,'yolo-aug-preview')
    md_json_file = options.output_file
    
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
    
    organization_name = ''
    project_name = ''
    
    options.job_name = f'{organization_name}-comparison'
    options.output_folder = os.path.join(output_folder,'model_comparison')
    options.image_folder = input_folder
    
    options.pairwise_options = []
    
    filenames = [
        f'/home/user/tmp/{project_name}/{project_name}_md_v5a.0.0-md_format.json',
        f'/home/user/postprocessing/{organization_name}/{organization_name}-2023-04-06-v5a.0.0/combined_api_outputs/{organization_name}-2023-04-06-v5a.0.0_detections.json',
        f'/home/user/postprocessing/{organization_name}/{organization_name}-2023-04-06-v5b.0.0/combined_api_outputs/{organization_name}-2023-04-06-v5b.0.0_detections.json'
        ]
    
    descriptions = ['YOLO w/augment','MDv5a','MDv5b']
    
    if False:
        results = []
        
        for fn in filenames:
            with open(fn,'r') as f:
                d = json.load(f)
            results.append(d)
        
    detection_thresholds = [0.1,0.1,0.1]
    
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
    

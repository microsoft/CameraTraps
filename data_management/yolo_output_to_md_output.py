#
# yolo_output_to_md_output.py
#
# Converts the output of YOLOv5's detect.py to the MD API output format.  Command-line
# driver not done yet, this has only been run interactively.
#
# Does not currently support recursive results, since detect.py doesn't save filenames
# in a way that allows easy inference of folder names..  Requires access to the input
# images, because the YOLO format uses the *absence* of a results file to indicate that
# no detections are present.
#
# YOLOv5 output has one text file per image, like so:
#
# 0 0.0141693 0.469758 0.0283385 0.131552 0.761428 
#
# That's [class, x_center, y_center, width_of_box, height_of_box, confidence]
#

#%% Imports and constants

import json
import os
import csv

import path_utils
import ct_utils


#%% Support functions

def yolo_output_to_md_output(input_results_folder,input_image_folder,output_file):
    
    assert os.path.isdir(input_results_folder)
    assert os.path.isdir(input_image_folder)
    
    ## Enumerate results files and image files
    
    yolo_results_files = os.listdir(input_results_folder)
    yolo_results_files = [f for f in yolo_results_files if f.lower().endswith('.txt')]
    # print('Found {} results files'.format(len(yolo_results_files)))
    
    image_files = path_utils.find_images(input_image_folder,recursive=False)
    image_files_relative = [os.path.basename(f) for f in image_files]
    # print('Found {} images'.format(len(image_files)))
            
    image_files_relative_no_extension = [os.path.splitext(f)[0] for f in image_files_relative]
    
    ## Make sure that every results file corresponds to an image
    
    for f in yolo_results_files:
        result_no_extension = os.path.splitext(f)[0]
        assert result_no_extension in image_files_relative_no_extension
    
    ## Build MD output data
    
    # Map 0-indexed YOLO categories to 1-indexed MD categories
    yolo_cat_map = { 0: 1, 1: 2, 2: 3 }
    
    images_entries = []

    # image_fn = image_files_relative[0]
    for image_fn in image_files_relative:
        
        image_name, ext = os.path.splitext(image_fn)        
        label_fn = image_name + '.txt'
        label_path = os.path.join(input_results_folder, label_fn)
            
        detections = []
        max_conf = 0.0
        
        if not os.path.exists(label_path):
            # This is assumed to be an image with no detections
            pass
        else:
            with open(label_path, newline='') as f:
                reader = csv.reader(f, delimiter=' ')
                for row in reader:
                    category = yolo_cat_map[int(row[0])]    
                    api_box = ct_utils.convert_yolo_to_xywh([float(row[1]), float(row[2]), 
                                                             float(row[3]), float(row[4])])
    
                    conf = ct_utils.truncate_float(float(row[5]), precision=4)
                    max_conf = max(max_conf, conf)
    
                    detections.append({
                        'category': str(category),
                        'conf': conf,
                        'bbox': ct_utils.truncate_float_array(api_box, precision=4)
                    })
                
        images_entries.append({
            'file': image_fn,
            'max_detection_conf': max_conf,
            'detections': detections
        })
    
    # ...for each image
    
    ## Save output file
    
    output_content = {
        'info': {
            'detector': 'converted_from_yolo_format',
            'detector_metadata': {},
            'format_version': '1.2'
        },
        'detection_categories': {
            '1': 'animal',
            '2': 'person',
            '3': 'vehicle'
        },
        'images': images_entries
    }
    
    with open(output_file,'w') as f:
        json.dump(output_content,f,indent=1)
    
# ...def yolo_output_to_md_output()


#%% Interactive driver

if False:
    
    pass

    #%%    
    
    input_results_folder = os.path.expanduser('~/tmp/model-version-experiments/pt-test-kru/exp/labels')
    input_image_folder = os.path.expanduser('~/data/KRU-test')
    output_file = os.path.expanduser('~/data/mdv5a-yolo-pt-kru.json')    
    yolo_output_to_md_output(input_results_folder,input_image_folder,output_file)
    
    #%%
    
    input_results_folder = os.path.expanduser('~/tmp/model-version-experiments/pt-test-bc/exp/labels')
    input_image_folder = os.path.expanduser('~/data/bc-test')
    output_file = os.path.expanduser('~/data/mdv5a-yolo-pt-bc.json')    
    yolo_output_to_md_output(input_results_folder,input_image_folder,output_file)
    
    #%%
    
    input_results_folder = os.path.expanduser('~/tmp/model-version-experiments/onnx-test-kru/exp/labels')
    input_image_folder = os.path.expanduser('~/data/KRU-test')
    output_file = os.path.expanduser('~/data/mdv5a-yolo-onnx-kru.json')    
    yolo_output_to_md_output(input_results_folder,input_image_folder,output_file)
    
    #%%
    
    input_results_folder = os.path.expanduser('~/tmp/model-version-experiments/onnx-test-bc/exp/labels')
    input_image_folder = os.path.expanduser('~/data/bc-test')
    output_file = os.path.expanduser('~/data/mdv5a-yolo-onnx-bc.json')    
    yolo_output_to_md_output(input_results_folder,input_image_folder,output_file)
    
    
#%% Command-line driver

# TODO


#%% Scrap

if False:
    
    pass

    #%% Visualize one set of converted results
    
    output_file = os.path.expanduser('~/data/mdv5a-yolo-onnx-bc.json')    
    input_image_folder = os.path.expanduser('~/data/bc-test')
    
    output_file = os.path.expanduser('~/data/mdv5a-yolo-onnx-kru.json')    
    input_image_folder = os.path.expanduser('~/data/KRU-test')
    
    from visualization import visualize_detector_output
    
    preview_dir = os.path.expanduser('~/tmp/yolo-conversion-preview')
    visualize_detector_output.visualize_detector_output(
                              detector_output_path=output_file,
                              out_dir=preview_dir,
                              confidence=0.1,
                              images_dir=input_image_folder,
                              # output_image_width=-1,
                              # sample=100,
                              render_detections_only=False)
    
    path_utils.open_file(preview_dir)


    #%% Compare a set of converted results to direct results
    
    output_file = os.path.expanduser('~/data/mdv5a-yolo-onnx-kru.json')    
    input_image_folder = os.path.expanduser('~/data/KRU-test')
    
    output_file = os.path.expanduser('~/data/mdv5a-yolo-onnx-bc.json')    
    input_image_folder = os.path.expanduser('~/data/bc-test')
    
    import itertools

    from api.batch_processing.postprocessing.compare_batch_results import (
        BatchComparisonOptions,PairwiseBatchComparisonOptions,compare_batch_results)

    options = BatchComparisonOptions()

    options.job_name = 'yolo-export-test'
    options.output_folder = os.path.join(preview_dir,'model_comparison')
    options.image_folder = input_image_folder

    options.pairwise_options = []

    filenames = [
        output_file,
        output_file.replace('-yolo-onnx','_batch')        
        ]

    detection_thresholds = [0.1,0.1]

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

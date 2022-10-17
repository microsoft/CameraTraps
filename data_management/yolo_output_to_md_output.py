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

def yolo_output_to_md_output(input_results_folder,input_image_folder,output_file,detector_tag=None):
    
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
    
    detector_string = 'converted_from_yolo_format'
    
    if detector_tag is not None:
        detector_string = detector_tag
        
    output_content = {
        'info': {
            'detector': detector_string,
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
    
    
#%% Command-line driver

# TODO

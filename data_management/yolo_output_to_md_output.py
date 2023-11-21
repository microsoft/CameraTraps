#
# yolo_output_to_md_output.py
#
# Converts the output of YOLOv5's detect.py or val.py to the MD API output format.
#
# Command-line driver not done yet, this has only been run interactively.
#

### Converting .txt files ###

#
# detect.py writes a .txt file per image, in YOLO training format.  Converting from this
# format does not currently support recursive results, since detect.py doesn't save filenames
# in a way that allows easy inference of folder names.  Requires access to the input
# images, because the YOLO format uses the *absence* of a results file to indicate that
# no detections are present.
#
# YOLOv5 output has one text file per image, like so:
#
# 0 0.0141693 0.469758 0.0283385 0.131552 0.761428 
#
# That's [class, x_center, y_center, width_of_box, height_of_box, confidence]
#
# val.py can write in this format as well, using the --save-txt argument.
#
# In both cases, a confidence value is only written to each line if you include the --save-conf
# argument.  Confidence values are required by this conversion script.
#

### Converting .json files ###

#
# val.py can also write a .json file in COCO-ish format.  It's "COCO-ish" because it's
# just the "images" portion of a COCO .json file.
#
# Converting from this format also requires access to the original images, since the format
# written by YOLOv5 uses absolute coordinates, but MD results are in relative coordinates.
#

#%% Imports and constants

import json
import os
import csv

from collections import defaultdict
from tqdm import tqdm

import path_utils
import ct_utils

from visualization import visualization_utils as visutils

from detection.run_detector import CONF_DIGITS, COORD_DIGITS


#%% Support functions

def yolo_json_output_to_md_output(yolo_json_file, image_folder,
                                  output_file, yolo_category_id_to_name,                              
                                  detector_name='unknown',
                                  image_id_to_relative_path=None,
                                  offset_yolo_class_ids=True,
                                  truncate_to_standard_md_precision=True):
    
    assert os.path.isfile(yolo_json_file), \
        'Could not find YOLO .json file {}'.format(yolo_json_file)
    assert os.path.isdir(image_folder), \
        'Could not find image folder {}'.format(image_folder)
        
    print('Converting {} to MD format'.format(yolo_json_file))
    
    if image_id_to_relative_path is None:
        
        image_files = path_utils.find_images(image_folder,recursive=True)
        image_files = [os.path.relpath(fn,image_folder) for fn in image_files]
        
        # YOLOv5 identifies images in .json output by ID, which is the filename without
        # extension.  If a mapping is not provided, these need to be unique.
        image_id_to_relative_path = {}
        
        for fn in image_files:
            image_id = os.path.splitext(os.path.basename(fn))[0]
            if image_id in image_id_to_relative_path:
                print('Error image ID {} refers to:\n{}\n{}'.format(
                    image_id,image_id_to_relative_path[image_id],fn))
                raise ValueError('Duplicate image ID {}'.format(image_id))
            image_id_to_relative_path[image_id] = fn

    image_files_relative = sorted(list(image_id_to_relative_path.values()))
    
    image_file_relative_to_image_id = {}
    for image_id in image_id_to_relative_path:
        relative_path = image_id_to_relative_path[image_id]
        assert relative_path not in image_file_relative_to_image_id
        image_file_relative_to_image_id[relative_path] = image_id
        
    with open(yolo_json_file,'r') as f:
        detections = json.load(f)
    assert isinstance(detections,list)
    
    image_id_to_detections = defaultdict(list)
    
    int_formatted_image_ids = False
    
    # det = detections[0]
    for det in detections:
        
        # This could be a string, but if the YOLOv5 inference script sees that the strings
        # are really ints, it converts to ints.
        image_id = det['image_id']
        image_id_to_detections[image_id].append(det)
        if isinstance(image_id,int):
            int_formatted_image_ids = True
    
    # If there are any ints present, everything should be ints
    if int_formatted_image_ids:
        for det in detections:
            assert isinstance(det['image_id'],int), \
                'Found mixed int and string image IDs'
                
    output_images = []
    
    # image_file_relative = image_files_relative[10]
    for image_file_relative in tqdm(image_files_relative):
        
        im = {}
        im['file'] = image_file_relative
        im['detections'] = []
        image_id = image_file_relative_to_image_id[image_file_relative]
        if int_formatted_image_ids:
            image_id = int(image_id)
        if image_id not in image_id_to_detections:
            detections = []
        else:
            detections = image_id_to_detections[image_id]
        
        image_full_path = os.path.join(image_folder,image_file_relative)
        pil_im = visutils.open_image(image_full_path)
        
        image_w = pil_im.size[0]
        image_h = pil_im.size[1]
        
        # det = detections[0]
        for det in detections:
            
            output_det = {}
            
            yolo_cat_id = int(det['category_id'])
            if offset_yolo_class_ids:
                yolo_cat_id += 1
            output_det['category'] = str(int(yolo_cat_id))
            conf = det['score']
            if truncate_to_standard_md_precision:
                conf = ct_utils.truncate_float(conf,CONF_DIGITS)
            output_det['conf'] = conf
            input_bbox = det['bbox']
            
            # YOLO's COCO .json is not *that* COCO-like, but it is COCO-like in
            # that the boxes are already [xmin/ymin/w/h]
            box_xmin_absolute = input_bbox[0]
            box_ymin_absolute = input_bbox[1]
            box_width_absolute = input_bbox[2]
            box_height_absolute = input_bbox[3]
            
            box_xmin_relative = box_xmin_absolute / image_w
            box_ymin_relative = box_ymin_absolute / image_h
            box_width_relative = box_width_absolute / image_w
            box_height_relative = box_height_absolute / image_h
            
            output_bbox = [box_xmin_relative,box_ymin_relative,
                           box_width_relative,box_height_relative]
            
            if truncate_to_standard_md_precision:
                output_bbox = ct_utils.truncate_float_array(output_bbox,COORD_DIGITS)
                
            output_det['bbox'] = output_bbox
            im['detections'].append(output_det)
            
        # ...for each detection            
        
        output_images.append(im)
        
    # ...for each image file
    
    d = {}
    d['images'] = output_images
    d['info'] = {'format_version':1.3,'detector':detector_name}
    d['detection_categories'] = {}
    
    for cat_id in yolo_category_id_to_name:
        yolo_cat_id = int(cat_id)
        if offset_yolo_class_ids:
            yolo_cat_id += 1
        d['detection_categories'][str(yolo_cat_id)] = yolo_category_id_to_name[cat_id]
    
    with open(output_file,'w') as f:
        json.dump(d,f,indent=1)
            
    
def yolo_txt_output_to_md_output(input_results_folder, image_folder,
                                 output_file, detector_tag=None):
    
    assert os.path.isdir(input_results_folder)
    assert os.path.isdir(image_folder)
    
    ## Enumerate results files and image files
    
    yolo_results_files = os.listdir(input_results_folder)
    yolo_results_files = [f for f in yolo_results_files if f.lower().endswith('.txt')]
    # print('Found {} results files'.format(len(yolo_results_files)))
    
    image_files = path_utils.find_images(image_folder,recursive=False)
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
            'format_version': '1.3'
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
    image_folder = os.path.expanduser('~/data/KRU-test')
    output_file = os.path.expanduser('~/data/mdv5a-yolo-pt-kru.json')    
    yolo_txt_output_to_md_output(input_results_folder,image_folder,output_file)
    
    
#%% Command-line driver

# TODO

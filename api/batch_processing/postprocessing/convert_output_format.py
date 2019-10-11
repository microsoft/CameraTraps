########
#
# convert_output_format.py
#
# Converts between file formats output by our batch processing API.  Currently
# supports json <--> csv conversion, but this should be the landing place for any
# conversion - including between future .json versions - that we support in the 
# future.
#
########

#%% Imports

import argparse
import csv
import json
import os
from tqdm import tqdm

from api.batch_processing.postprocessing.load_api_results import load_api_results_csv
from data_management.annotations import annotation_constants


#%% Conversion functions

def convert_json_to_csv(input_path,output_path=None,min_confidence=None,omit_bounding_boxes=False):
    
    if output_path is None:
        output_path = os.path.splitext(input_path)[0]+'.csv'
        
    print('Loading json results from {}...'.format(input_path))
    json_output = json.load(open(input_path))

    rows = []
    
    # We add an output column for each class other than 'empty', 
    # containing the maximum probability of  that class for each image
    n_non_empty_categories = len(annotation_constants.bbox_categories) - 1
    category_column_names = []
    assert annotation_constants.bbox_category_id_to_name[0] == 'empty'
    for cat_id in range(1,n_non_empty_categories+1):
        cat_name = annotation_constants.bbox_category_id_to_name[cat_id]
        category_column_names.append('max_conf_' + cat_name)
        
    print('Iterating through results...')
    for im in tqdm(json_output['images']):
        
        image_id = im['file']
        max_conf = im['max_detection_conf']
        detections = []
        max_category_probabilities = [None] * n_non_empty_categories
                
        for d in im['detections']:
            
            # Skip sub-threshold detections
            if (min_confidence is not None) and (d['conf'] < min_confidence):
                continue
            
            detection = d['bbox']
            
            # Our .json format is xmin/ymin/w/h
            #
            # Our .csv format was ymin/xmin/ymax/xmax
            xmin = detection[0]
            ymin = detection[1]
            xmax = detection[0] + detection[2]
            ymax = detection[1] + detection[3]
            detection = [ymin, xmin, ymax, xmax]
                
            detection.append(d['conf'])
            
            # Category 0 is empty, for which we don't have a column, so the max
            # confidence for category N goes in column N-1
            category_id = int(d['category'])
            assert category_id > 0 and category_id <= n_non_empty_categories
            category_column = category_id - 1
            category_max = max_category_probabilities[category_column]
            if category_max is None or d['conf'] > category_max:
                max_category_probabilities[category_column] = d['conf']
            
            detection.append(category_id)
            detections.append(detection)
            
        # ...for each detection
        
        detection_string = ''
        if not omit_bounding_boxes:
            detection_string = json.dumps(detections)
            
        row = [image_id, max_conf, detection_string]
        row.extend(max_category_probabilities)
        rows.append(row)
        
    # ...for each image

    print('Writing to csv...')
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        header = ['image_path', 'max_confidence', 'detections']
        header.extend(category_column_names)
        writer.writerow(header)
        writer.writerows(rows)

    
def convert_csv_to_json(input_path,output_path=None):
    
    if output_path is None:
        output_path = os.path.splitext(input_path)[0]+'.json'
        
    # Format spec:
    #
    # https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing
    
    print('Loading csv results...')
    df = load_api_results_csv(input_path)

    info = {
        "detector": "unknown",
        "detection_completion_time" : "unknown",
        "classifier": "unknown",
        "classification_completion_time": "unknown"
    }
    
    classification_categories = {}
    detection_categories = annotation_constants.bbox_category_id_to_name

    images = []
    
    # iFile = 0; row = df.iloc[iFile]
    for iFile,row in df.iterrows():
        
        image = {}
        image['file'] = row['image_path']
        image['max_detection_conf'] = row['max_confidence']
        src_detections = row['detections']        
        out_detections = []
        
        for iDetection,detection in enumerate(src_detections):
            
            # Our .csv format was ymin/xmin/ymax/xmax
            #
            # Our .json format is xmin/ymin/w/h
            ymin = detection[0]
            xmin = detection[1]
            ymax = detection[2]
            xmax = detection[3]
            bbox = [xmin, ymin, xmax-xmin, ymax-ymin]
            conf = detection[4]
            iClass = detection[5]
            out_detection = {}
            out_detection['category'] = str(iClass)
            out_detection['conf'] = conf
            out_detection['bbox'] = bbox            
            out_detections.append(out_detection)
            
        # ...for each detection
        
        image['detections'] = out_detections
        images.append(image)
        
    # ...for each image        
    json_out = {}
    json_out['info'] = info
    json_out['detection_categories'] = detection_categories
    json_out['classification_categories'] = classification_categories
    json_out['images'] = images
    
    json.dump(json_out,open(output_path,'w'),indent=1)


#%% Interactive driver

if False:    

    #%%
    
    min_confidence = None
    input_path = r'd:\wildlife_data\bellevue_camera_traps\706_detections_bellevuecameratraps20190718_20190718162519.json'
    output_path = input_path + '.csv'
    convert_json_to_csv(input_path,output_path,min_confidence=min_confidence,omit_bounding_boxes=False)
        
    #%% 
    
    min_confidence = None    
    input_paths = [r'D:\temp\idfg_json_to_csv\detections_idfg_20190625_refiltered.json',
                   r'D:\temp\idfg_json_to_csv\idfg_20190801-hddrop_combined.refiltered_trimmed_renamed.json']
    for input_path in input_paths:
        output_path = input_path + '.csv'
        convert_json_to_csv(input_path,output_path,min_confidence=min_confidence,omit_bounding_boxes=True)
    
    
#%% Command-line driver
        
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    if args.input_path.endswith('.csv') and args.output_path.endswith('.json'):
        convert_csv_to_json(args.input_path,args.output_path)
    elif args.input_path.endswith('.json') and args.output_path.endswith('.csv'):
        convert_json_to_csv(args.output_path,args.input_path)
    else:
        raise ValueError('Illegal format combination')            

if __name__ == '__main__':
    main()
    
    

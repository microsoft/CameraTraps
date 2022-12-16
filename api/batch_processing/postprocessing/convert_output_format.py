########
#
# convert_output_format.py
#
# Converts between file formats output by our batch processing API.  Currently
# supports json <--> csv conversion, but this should be the landing place for any
# conversion - including between future .json versions - that we support in the 
# future.
#
# Does not support classification results.
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

CONF_DIGITS = 3


#%% Conversion functions

def convert_json_to_csv(input_path,output_path=None,min_confidence=None,
                        omit_bounding_boxes=False,output_encoding=None):
    
    if output_path is None:
        output_path = os.path.splitext(input_path)[0]+'.csv'
        
    print('Loading json results from {}...'.format(input_path))
    json_output = json.load(open(input_path))

    rows = []
    
    # We add an output column for each class other than 'empty', 
    # containing the maximum probability of  that class for each image
    n_non_empty_categories = len(annotation_constants.annotation_bbox_categories) - 1
    category_column_names = []
    assert annotation_constants.annotation_bbox_category_id_to_name[0] == 'empty'
    for cat_id in range(1,n_non_empty_categories+1):
        cat_name = annotation_constants.annotation_bbox_category_id_to_name[cat_id]
        category_column_names.append('max_conf_' + cat_name)
        
    print('Iterating through results...')
    for im in tqdm(json_output['images']):
        
        image_id = im['file']
        
        if 'failure' in im and im['failure'] is not None:
            row = [image_id, 'failure', im['failure']]
            rows.append(row)
            print('Skipping failed image {} ({})'.format(im['file'],im['failure']))
            continue

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
    with open(output_path, 'w', newline='', encoding=output_encoding) as f:
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
    detection_categories = annotation_constants.annotation_bbox_category_id_to_name

    images = []
    
    # iFile = 0; row = df.iloc[iFile]
    for iFile,row in df.iterrows():
        
        image = {}
        image['file'] = row['image_path']
        image['max_detection_conf'] = round(row['max_confidence'], CONF_DIGITS)
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
    input_path = r'c:\temp\test.json'
    output_path = input_path + '.csv'
    convert_json_to_csv(input_path,output_path,min_confidence=min_confidence,omit_bounding_boxes=False)
            
    #%%
    
    base_path = r'c:\temp\json'
    input_paths = os.listdir(base_path)
    input_paths = [os.path.join(base_path,s) for s in input_paths]
    
    min_confidence = None    
    for input_path in input_paths:
        output_path = input_path + '.csv'
        convert_json_to_csv(input_path,output_path,min_confidence=min_confidence,omit_bounding_boxes=True)    
    
    #%% Concatenate .csv files from a folder

    import glob
    csv_files = glob.glob(os.path.join(base_path,'*.json.csv' ))
    master_csv = os.path.join(base_path,'all.csv')
    
    print('Concatenating {} files to {}'.format(len(csv_files),master_csv))
    
    header = None
    with open(master_csv, 'w') as fout:
        
        for filename in tqdm(csv_files):
            
            with open(filename) as fin:
                
                lines = fin.readlines()
                
                if header is not None:
                    assert lines[0] == header
                else:
                    header = lines[0]
                    fout.write(header)
                    
                for line in lines[1:]:
                    if len(line.strip()) == 0:
                        continue                    
                    fout.write(line)
                    
        # ...for each .csv file
        
    # with open(master_csv)
    
    
#%% Command-line driver
        
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path',type=str,
                        help='Input filename ending in .json or .csv')
    parser.add_argument('--output_path',type=str,default=None,
                        help='Output filename ending in .json or .csv (defaults to ' + \
                            'input file, with .json/.csv replaced by .csv/.json)')
    args = parser.parse_args()
    
    if args.output_path is None:
        if args.input_path.endswith('.csv'):
            args.output_path = args.input_path[:-4] + '.json'
        elif args.input_path.endswith('.json'):
            args.output_path = args.input_path[:-5] + '.csv'
        else:
            raise ValueError('Illegal input file extension')    
    
    if args.input_path.endswith('.csv') and args.output_path.endswith('.json'):
        convert_csv_to_json(args.input_path,args.output_path)
    elif args.input_path.endswith('.json') and args.output_path.endswith('.csv'):
        convert_json_to_csv(args.input_path,args.output_path)
    else:
        raise ValueError('Illegal format combination')            

if __name__ == '__main__':
    main()
    
    

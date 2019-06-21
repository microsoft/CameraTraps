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

from api.batch_processing.postprocessing.load_api_results import load_api_results
from data_management.annotations import annotation_constants


#%% Conversion functions

def convert_json_to_csv(input_path,output_path):
    
    print('Loading json results...')
    json_output = json.load(open(input_path))

    rows = []

    print('Iterating through results...')
    for i in json_output['images']:
        image_id = i['file']
        max_conf = i['max_detection_conf']
        detections = []
        for d in i['detections']:
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
            detection.append(int(d['category']))
            detections.append(detection)
        rows.append((image_id, max_conf, json.dumps(detections)))

    print('Writing to csv...')
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['image_path', 'max_confidence', 'detections'])
        writer.writerows(rows)


def convert_csv_to_json(input_path,output_path):
    
    # Format spec:
    #
    # https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing
    
    print('Loading csv results...')
    df = load_api_results(input_path)

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
            
            # Our .csv format was xmin/ymin/xmax/ymax
            #
            # Our .json format is xmin/ymin/w/h
            bbox = detection[0:4]
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
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
    
    

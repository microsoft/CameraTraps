# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import redis
import time
from datetime import datetime
from io import BytesIO

from requests_toolbelt.multipart.encoder import MultipartEncoder

import api_config
from run_tf_detector import TFDetector
import visualization.visualization_utils as viz_utils

db = redis.StrictRedis(host=api_config.REDIS_HOST, port=api_config.REDIS_PORT)
detector = TFDetector(api_config.DETECTOR_MODEL_PATH)

current_directory = os.path.dirname(os.path.realpath(__file__))

def detect_process():
    detection_results = []
    inference_time_detector = []

    while True:
        serialized_entry = db.lpop(api_config.REDIS_QUEUE)
        
        detection_results = []
        inference_time_detector = []
        if serialized_entry:
            entry = json.loads(serialized_entry)
            id = entry['id']
            render_boxes = entry['render_boxes']
            detection_confidence = entry['detection_confidence']

            try:
                temp_direc = f'{current_directory}/{api_config.TEMP_FOLDER}/{id}'
                for filename in os.listdir(temp_direc):
                    image = open(f'{temp_direc}/{filename}', "rb")
                    image = viz_utils.load_image(image)

                    start_time = time.time()
                    result = detector.generate_detections_one_image(image, filename)
                    detection_results.append(result)

                    elapsed = time.time() - start_time
                    inference_time_detector.append(elapsed)
            except Exception as e:
                print('Error performing detection on the images: ' + str(e))
                
                db.set(entry['id'], json.dumps({ 
                    'status': 500,
                    'error': 'Error performing detection on the images: ' + str(e)
                }))

            # filter the detections by the confidence threshold
            filtered_results = {}  # json to return to the user along with the rendered images if they opted for it
            # each result is [ymin, xmin, ymax, xmax, confidence, category]
            try:
                for result in detection_results:
                    image_name = result['file']
                    detections = result.get('detections', None)
                    filtered_results[image_name] = []

                    if detections is None:
                        continue

                    for d in detections:
                        if d['conf'] > detection_confidence:
                            res = TFDetector.convert_to_tf_coords(d['bbox'])
                            res.append(d['conf'])
                            res.append(int(d['category']))  # category is an int here, not string as in the async API
                            filtered_results[image_name].append(res)
    
                ## TODO: log
                db.set(entry['id'], json.dumps({ 
                    'status': 200,
                    'detection_results': detection_results,
                    'filtered_results': filtered_results,
                    'inference_time_detector': inference_time_detector
                }))
              
            except Exception as e:
                print('Error consolidating the detection boxes: ' + str(e))
                
                db.set(entry['id'], json.dumps({ 
                    'status': 500,
                    'error': 'Error consolidating the detection boxes:' + str(e)
                }))

    
if __name__ == '__main__':
    detect_process() 

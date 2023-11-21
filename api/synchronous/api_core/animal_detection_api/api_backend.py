#
# api_backend.py
#
# Defines the model execution service, which pulls requests (one or more images)
# from the shared Redis queue, and runs them through the TF model.
#

#%% Imports

import os
import json
import time
import redis
import argparse
import PIL

from io import BytesIO

from detection.run_detector import load_detector, convert_to_tf_coords
import config
import visualization.visualization_utils as viz_utils 

#%% Initialization

db = redis.StrictRedis(host=config.REDIS_HOST, port=config.REDIS_PORT)
current_directory = os.path.dirname(os.path.realpath(__file__))


#%% Main loop

def detect_process():
     
     while True:
        
        # TODO: convert to a blocking read and eliminate the sleep() statement in this loop
        serialized_entry = db.lpop(config.REDIS_QUEUE_NAME)
        all_detection_results = []
        inference_time_detector = []
        
        if serialized_entry:
            
            entry = json.loads(serialized_entry)
            id = entry['id']
            print('Processing images from request id:', id)
            return_confidence_threshold = entry['return_confidence_threshold']

            try:
                
                temp_direc = os.path.join(config.TEMP_FOLDER,id)
                assert os.path.isdir(temp_direc), 'Could not find temporary folder {}'.format(temp_direc)
                
                for filename in os.listdir(temp_direc):
                    
                    image_path = f'{temp_direc}/{filename}'
                    print('Reading image from {}'.format(image_path))
                    image = open(image_path, 'rb')
                    image = viz_utils.load_image(image)

                    start_time = time.time()
                    result = detector.generate_detections_one_image(image, filename, detection_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD)
                    all_detection_results.append(result)

                    elapsed = time.time() - start_time
                    inference_time_detector.append(elapsed)
                    
            except Exception as e:

                print('Detection error: ' + str(e))
                
                db.set(entry['id'], json.dumps({ 
                    'status': 500,
                    'error': 'Detection error: ' + str(e)
                }))

                continue

            # Filter the detections by the confidence threshold
            #
            # Each result is [ymin, xmin, ymax, xmax, confidence, category]
            #
            # Coordinates are relative, with the origin in the upper-left
            detections = {}  
            
            try:
                
                for result in all_detection_results:
                    
                    image_name = result['file']
                    _detections = result.get('detections', None)
                    detections[image_name] = []

                    if _detections is None:
                        continue

                    for d in _detections:
                        if d['conf'] > return_confidence_threshold:
                            res = convert_to_tf_coords(d['bbox'])
                            res.append(d['conf'])
                            res.append(int(d['category']))
                            detections[image_name].append(res)
    
                db.set(entry['id'], json.dumps({ 
                    'status': 200,
                    'detections': detections,
                    'inference_time_detector': inference_time_detector
                }))
              
            except Exception as e:
                print('Error consolidating the detection boxes: ' + str(e))
                
                db.set(entry['id'], json.dumps({ 
                    'status': 500,
                    'error': 'Error consolidating the detection boxes:' + str(e)
                }))

        # ...if serialized_entry
        
        else:
            time.sleep(0.005)
    
    # ...while(True)
    
# ...def detect_process()    
        

#%% Command-line driver

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='api backend')

    # use --non-docker if you are testing without Docker
    #
    # python api_frontend.py --non-docker
    parser.add_argument('--non-docker', action='store_true', default=False)
    args = parser.parse_args()

    if args.non_docker:
        model_path = config.DETECTOR_MODEL_PATH_DEBUG
    else:
        model_path = config.DETECTOR_MODEL_PATH

    detector = load_detector(model_path)

    # run detections on a test image to load the model
    print('Running initial detection to load model...')
    test_image = PIL.Image.new(mode="RGB", size=(200, 200))
    result = detector.generate_detections_one_image(test_image, "test_image", detection_threshold=config.DEFAULT_CONFIDENCE_THRESHOLD)
    print(result)
    print('\n')

    detect_process() 



# 
# api_frontend.py
#
# Defines the Flask app, which takes requests (one or more images) from
# remote callers and pushes the images onto the shared Redis queue, to be processed
# by the main service in api_backend.py .
#

#%% Imports

import os
import json
import time
import uuid
import redis
import shutil
import argparse
import traceback

from io import BytesIO
from flask import Flask, Response, jsonify, make_response, request
from requests_toolbelt.multipart.encoder import MultipartEncoder

import visualization.visualization_utils as viz_utils
import config


#%% Initialization

app = Flask(__name__)
db = redis.StrictRedis(host=config.REDIS_HOST, port=config.REDIS_PORT)


#%% Support functions

def _make_error_object(error_code, error_message):
    
    # Make a dict that the request_processing_function can return to the endpoint 
    # function to notify it of an error
    return {
        'error_message': error_message,
        'error_code': error_code
    }


def _make_error_response(error_code, error_message):

    return make_response(jsonify({'error': error_message}), error_code)


def has_access(request):
    
    if not os.path.exists(config.API_KEYS_FILE):
        return True
    else:
        if not request.headers.get('key'):
            print('Key header not available')
            return False
        else:
            API_key = request.headers.get('key').strip().lower()
            with open(config.API_KEYS_FILE, "r") as f:
                for line in f:
                    valid_key = line.strip().lower()
                    if valid_key == API_key:
                        return True

    return False


def check_posted_data(request):
 
    files = request.files
    params = request.args

    # Verify that the content uploaded is not too big
    #
    # request.content_length is the length of the total payload
    content_length = request.content_length
    if not content_length:
        return _make_error_object(411, 'No image(s) were sent, or content length cannot be determined.')
    if content_length > config.MAX_CONTENT_LENGTH_IN_MB * 1024 * 1024:
        return _make_error_object(413, ('Payload size {:.2f} MB exceeds the maximum allowed of {} MB. '
                    'Please upload fewer or more compressed images.').format(
            content_length / (1024 * 1024), config.MAX_CONTENT_LENGTH_IN_MB))

    render_boxes = True if params.get('render', '').lower() == 'true' else False

    if 'min_confidence' in params:
        return_confidence_threshold = float(params['min_confidence'])
        print('runserver, post_detect_sync, user specified detection confidence: ', return_confidence_threshold)
        if return_confidence_threshold < 0.0 or return_confidence_threshold > 1.0:
            return _make_error_object(400, 'Detection confidence threshold {} is invalid, should be between 0.0 and 1.0.'.format(
                return_confidence_threshold))
    else:
        return_confidence_threshold = config.DEFAULT_CONFIDENCE_THRESHOLD
        
    if 'min_rendering_confidence' in params:
        rendering_confidence_threshold = float(params['min_rendering_confidence'])
        print('runserver, post_detect_sync, user specified rendering confidence: ', rendering_confidence_threshold) 
        if rendering_confidence_threshold < 0.0 or rendering_confidence_threshold > 1.0:
            return _make_error_object(400, 'Rendering confidence threshold {} is invalid, should be between 0.0 and 1.0.'.format(
                rendering_confidence_threshold))
    else:
        rendering_confidence_threshold =  config.DEFAULT_RENDERING_CONFIDENCE_THRESHOLD
    
    # Verify that the number of images is acceptable
    num_images = sum([1 if file.content_type in config.IMAGE_CONTENT_TYPES else 0 for file in files.values()])
    print('runserver, post_detect_sync, number of images received: ', num_images)

    if num_images > config.MAX_IMAGES_ACCEPTED:
        return _make_error_object(413, 'Too many images. Maximum number of images that can be processed in one call is {}.'.format(config.MAX_IMAGES_ACCEPTED))
    elif num_images == 0:
        return _make_error_object(400, 'No image(s) of accepted types (image/jpeg, image/png, application/octet-stream) received.')
    
    return {
        'render_boxes': render_boxes,
        'return_confidence_threshold': return_confidence_threshold,
        'rendering_confidence_threshold': rendering_confidence_threshold
    }

# ...def check_posted_data(request)
    

#%% Main loop

@app.route(config.API_PREFIX + '/detect', methods = ['POST'])
def detect_sync():
    
    if not has_access(request):
        print('Access denied, please provide a valid API key')
        return _make_error_response(403, 'Access denied, please provide a valid API key')

    # Check whether the request_processing_function had an error
    post_data = check_posted_data(request)
    if post_data.get('error_code', None) is not None:
        return _make_error_response(post_data.get('error_code'), post_data.get('error_message'))

    render_boxes = post_data.get('render_boxes')
    return_confidence_threshold = post_data.get('return_confidence_threshold')
    rendering_confidence_threshold = post_data.get('rendering_confidence_threshold')
  
    redis_id = str(uuid.uuid4())
    d = {'id': redis_id, 'render_boxes': render_boxes, 'return_confidence_threshold': return_confidence_threshold}
    temp_direc = os.path.join(config.TEMP_FOLDER, redis_id)
    
    try:
        
        try:
            # Write images to temporary files
            #
            # TODO: read from memory rather than using intermediate files
            os.makedirs(temp_direc,exist_ok=True)
            for name, file in request.files.items():
                if file.content_type in config.IMAGE_CONTENT_TYPES:
                    filename = request.files[name].filename
                    image_path = os.path.join(temp_direc, filename)
                    print('Saving image {} to {}'.format(name,image_path))
                    file.save(image_path)
                    assert os.path.isfile(image_path),'Error creating file {}'.format(image_path)
        
        except Exception as e:
            return _make_error_object(500, 'Error saving images: ' + str(e))
        
        # Submit the image(s) for processing by api_backend.py, who is waiting on this queue
        db.rpush(config.REDIS_QUEUE_NAME, json.dumps(d))
        
        while True:
            
            # TODO: convert to a blocking read and eliminate the sleep() statement in this loop
            result = db.get(redis_id)
            
            if result:
                
                result = json.loads(result.decode())
                print('Processing result {}'.format(str(result)))
                
                if result['status'] == 200:
                    detections = result['detections']
                    db.delete(redis_id)
                
                else:
                    db.delete(redis_id)
                    print('Detection error: ' + str(result))
                    return _make_error_response(500, 'Detection error: ' + str(result))

                try:
                    print('detect_sync: postprocessing and sending images back...')
                    fields = {
                        'detection_result': ('detection_result', json.dumps(detections), 'application/json'),
                    }

                    if render_boxes and result['status'] == 200:

                        print('Rendering images')

                        for image_name, detections in detections.items():
                            
                            #image = Image.open(os.path.join(temp_direc, image_name))
                            image = open(f'{temp_direc}/{image_name}', "rb")
                            image = viz_utils.load_image(image)
                            width, height = image.size

                            _detections = []
                            for d in detections:
                                y1,x1,y2,x2 = d[0:4]
                                width = x2 - x1
                                height = y2 - y1
                                bbox = [x1,y1,width,height]
                                _detections.append({'bbox': bbox, 'conf': d[4], 'category': d[5]}) 
                            
                            viz_utils.render_detection_bounding_boxes(_detections, image, 
                            confidence_threshold=rendering_confidence_threshold)
                            
                            output_img_stream = BytesIO()
                            image.save(output_img_stream, format='jpeg')
                            output_img_stream.seek(0)
                            fields[image_name] = (image_name, output_img_stream, 'image/jpeg')
                        print('Done rendering images')
                        
                    m = MultipartEncoder(fields=fields)                    
                    return Response(m.to_string(), mimetype=m.content_type)

                except Exception as e:
                    
                    print(traceback.format_exc())
                    print('Error returning result or rendering the detection boxes: ' + str(e))

                finally:
                    
                    try:
                        print('Removing temporary files')
                        shutil.rmtree(temp_direc)
                    except Exception as e:
                        print('Error removing temporary folder {}: {}'.format(temp_direc,str(e)))
                    
            else:
                time.sleep(0.005)
                
            # ...if we do/don't have a request available on the queue
            
        # ...while(True)

    except Exception as e:
        
        print(traceback.format_exc())
        return _make_error_object(500, 'Error processing images: ' + str(e))

# ...def detect_sync()


#%% Command-line driver
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='api frontend')

    # use --non-docker if you are testing without Docker
    #
    # python api_frontend.py --non-docker
    parser.add_argument('--non-docker', action="store_true", default=False)
    args = parser.parse_args()

    if args.non_docker:
        app.run(host='0.0.0.0', port=5050)
    else:
        app.run()



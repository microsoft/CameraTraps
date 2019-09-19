# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# # /ai4e_api_tools has been added to the PYTHONPATH, so we can reference those
# libraries directly.
import json
from datetime import datetime
from io import BytesIO

from ai4e_app_insights_wrapper import AI4EAppInsights
from ai4e_service import APIService
from flask import Flask, Response, jsonify, abort
from flask import make_response
from requests_toolbelt.multipart.encoder import MultipartEncoder

import api_config
from tf_detector import TFDetector
from tf_classifer import TFClassifier


print('Creating Application')
app = Flask(__name__)

# Use the AI4EAppInsights library to send log messages. NOT REQURIED
log = AI4EAppInsights()

# Use the AI4EService to execute your functions within a logging trace, which supports long-running/async
# functions, handles SIGTERM signals from AKS, etc., and handles concurrent requests.
with app.app_context():
    ai4e_service = APIService(app, log)

detector = TFDetector(api_config.MODEL_PATH)
classifier = TFClassifier(api_config.CLASSIFICATION_MODEL_PATHS, api_config.CLASSIFICATION_CLASS_NAMES)


# function for processing the request data to the /detect endpoint. It loads data or files into a
# dictionary for access in the API function. It is passed as a parameter to the API setup.
def _detect_process_request_data(request):
    files = request.files
    params = request.args

    
    # check that the content uploaded is not too big
    # request.content_length is the length of the total payload
    content_length = request.content_length if request.content_length \
        else api_config.MAX_CONTENT_LENGTH_IN_MB * 1024 * 1024 + 1
    if content_length > api_config.MAX_CONTENT_LENGTH_IN_MB * 1024 * 1024:
        abort(413, ('Payload size {:.2f} MB exceeds the maximum allowed of {} MB, or payload content length'
                            ' cannot be determined. Please upload fewer or more compressed images.').format(
            content_length / (1024 * 1024), api_config.MAX_CONTENT_LENGTH_IN_MB))

    render_boxes = True if params.get('render', '') in ['True', 'true'] else False

    # validate detection confidence value
    if 'confidence' in params:
        detection_confidence = float(params['confidence'])
        print('runserver, post_detect_sync, detection confidence: ', detection_confidence)
        if detection_confidence < 0.0 or detection_confidence > 1.0:
            abort(400, 'Detection confidence {} is invalid. Needs to be between 0.0 and 1.0.'.format(
                detection_confidence))
    else:
        detection_confidence = api_config.DEFAULT_DETECTION_CONFIDENCE

    # check that the number of images is acceptable for this synchronous API
    num_images = sum([1 if file.content_type in api_config.IMAGE_CONTENT_TYPES else 0 for file in files.values()])
    print('runserver, post_detect_sync, number of images received: ', num_images)
    if num_images > api_config.MAX_IMAGES_ACCEPTED:
        abort(413,
                      'Too many images. Maximum number of images that can be processed in one call is {}.'.format(str(
                        api_config.MAX_IMAGES_ACCEPTED)))
    elif num_images == 0:
        abort(400, 'No image(s) of accepted types (image/jpeg, image/png, application/octet-stream) received.')

    if 'classification' in params:
        classification = params['classification']

        if classification not in api_config.CLASSIFICATION_CLASS_NAMES.keys():
            error_message = 'Classification name provided is not supported, The classifiers supported are {}'\
                .format(str(list(api_config.CLASSIFICATION_CLASS_NAMES.keys())).replace('[', '').replace(']', ''))

            abort(413, error_message)

            return make_response(jsonify({'error', error_message}), 500)

            
    else:
        classification = None

    # read input images and parameters
    try:
        print('runserver, post_detect_sync, reading input images...')
        images, image_names = [], []
        for k, file in files.items():
            # file of type SpooledTemporaryFile has attributes content_type and a read() method
            if file.content_type in api_config.IMAGE_CONTENT_TYPES:
                images.append(TFDetector.open_image(file))
                image_names.append(k)
    except Exception as e:
        log.log_exception('Error reading the images: ' + str(e))
        abort(500, 'Error reading the images: ' + str(e))

    return {
        'render_boxes': render_boxes,
        'detection_confidence': detection_confidence,
        'images': images,
        'image_names': image_names,
        'classification': classification
    }


def convert_numpy_floats(np_array):
    new = []
    for i in np_array:
        new.append(float(i))
    return new


@ai4e_service.api_sync_func(api_path='/detect',
                            methods=['POST'],
                            request_processing_function=_detect_process_request_data,  # data process function
                            # if the number of requests exceed this limit, a 503 is returned to the caller.
                            maximum_concurrent_requests=10,
                            trace_name='post:detect_sync')
def detect_sync(*args, **kwargs):
    render_boxes = kwargs.get('render_boxes')
    classification = kwargs.get('classification')
    detection_confidence = kwargs.get('detection_confidence')

    images = kwargs.get('images')
    image_names = kwargs.get('image_names')

    # consolidate the images into batches and perform detection on them
    try:
        print('runserver, post_detect_sync, batching and inferencing...')
        # detections is an array of dicts
        tic = datetime.now()
        detections = detector.generate_detections_batch(images)
        toc = datetime.now()
        detection_inference_duration = toc - tic
        print('runserver, post_detect_sync, detection inference duration: {} seconds.'\
            .format(detection_inference_duration))
    except Exception as e:
        print('Error performing detection on the images: ' + str(e))
        log.log_exception('Error performing detection on the images: ' + str(e))
        abort(500, 'Error performing detection on the images: ' + str(e))

    # filter the detections by the confidence threshold
    try:
        result = {}  # json to return to the user along with the rendered images if they opted for it
        for image_name, d in zip(image_names, detections):
            result[image_name] = []
            for box, score, category in zip(d['box'], d['score'], d['category']):
                if score > detection_confidence:
                    # each result is [ymin, xmin, ymax, xmax, confidence, category]
                    res = convert_numpy_floats(box)  # numpy float doesn't jsonify
                    res.append(float(score))
                    res.append(int(category))  # category is an int here, not string as in the async API
                    result[image_name].append(res)  

    except Exception as e:
        print('Error consolidating the detection boxes: ' + str(e))
        log.log_exception('Error consolidating the detection boxes: ' + str(e))
        abort(500, 'Error consolidating the detection boxes: ' + str(e))


    #classification
    try:
        if classification:
            print('runserver, classification...')
            tic = datetime.now()
            classification_result = classifier.classify_boxes(images, image_names, result, classification)
            toc = datetime.now()
            classification_inference_duration = toc - tic
            print('runserver, classification, classifcation inference duraction: {}'\
                .format({classification_inference_duration}))

        else:
            classification_result = {}

    except Exception as e:
        print('Error performing classification on the images: ' + str(e))
        log.log_exception('Error performing classification on the images: ' + str(e))
        abort(500, 'Error performing classification on the images: ' + str(e))
        

    # return results; optionally render the detections on the images and send the annotated images back
    try:
        print('runserver, post_detect_sync, rendering and sending images back...')
        files = {
            'detection_result': ('result', json.dumps(result), 'application/json'),
            'classification_result': ('result', json.dumps(classification_result), 'application/json')
        }

        if render_boxes:
            for image_name, d in zip(image_names, detections):
                image = d['image']
                TFDetector.render_bounding_boxes(d['box'], d['score'], d['category'], image,
                                                 confidence_threshold=detection_confidence)
                output_img_stream = BytesIO()
                image.save(output_img_stream, format='jpeg')
                output_img_stream.seek(0)
                files[image_name] = (image_name, output_img_stream, 'image/jpeg')
        m = MultipartEncoder(fields=files)

        log.log_info('runserver, post_detect_sync, detection inference duration: {} seconds, \
                        classification inference duration: {} seconds.'\
                            .format(detection_inference_duration, classification_inference_duration),
                                additionalProperties={
                                    'detection_inference_duration': str(detection_inference_duration),
                                    'classification_inference_duration': str(classification_inference_duration),
                                    'num_images': len(image_names),
                                    'render_boxes': render_boxes,
                                    'detection_confidence': detection_confidence
                                })
        return Response(m.to_string(), mimetype=m.content_type)
    except Exception as e:
        print('Error returning result or rendering the detection boxes: ' + str(e))
        log.log_exception('Error returning result or rendering the detection boxes: ' + str(e))
        abort(500, 'Error returning result or rendering the detection boxes: ' + str(e))


@ai4e_service.api_sync_func(api_path='/detector_model_version',
                            methods=['GET'],
                            maximum_concurrent_requests=1000,
                            trace_name='get:get_model_detector_version')
def get_model_detector_version(*args, **kwargs):
    try:
        return api_config.MODEL_VERSION
    except Exception as e:
        return 'Detection Model version unknown. Error: {}'.format(str(e))

@ai4e_service.api_sync_func(api_path='/supported_classifiers',
                            methods=['GET'],
                            maximum_concurrent_requests=1000,
                            trace_name='get:get_supported_classifiers')
def get_supported_classifiers(*args, **kwargs):
    try:
        return list(api_config.CLASSIFICATION_CLASS_NAMES.keys())
    except Exception as e:
        return 'Supported classifiers unknown. Error: {}'.format(str(e))

if __name__ == '__main__':
    app.run()

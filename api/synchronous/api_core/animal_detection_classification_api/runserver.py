# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# # /ai4e_api_tools has been added to the PYTHONPATH, so we can reference those libraries directly.

import json
import time
from datetime import datetime
from io import BytesIO

from ai4e_app_insights_wrapper import AI4EAppInsights
from ai4e_service import APIService
from flask import Flask, Response, jsonify, make_response
from requests_toolbelt.multipart.encoder import MultipartEncoder

import api_config
from run_tf_detector import TFDetector
# from tf_classifer import TFClassifier
import visualization.visualization_utils as viz_utils

print('Creating application')
app = Flask(__name__)

# Use the AI4EAppInsights library to send log messages.
log = AI4EAppInsights()


# Use the APIService to executes your functions within a logging trace, supports long-running/async functions,
# handles SIGTERM signals from AKS, etc., and handles concurrent requests.
with app.app_context():
    ai4e_service = APIService(app, log)


# Load the models when the API starts

start_time = time.time()
detector = TFDetector(api_config.DETECTOR_MODEL_PATH)
elapsed = time.time() - start_time
log.log_info('detector loading time', elapsed)
print('detector loading time: ', elapsed)

# TODO classifier = TFClassifier(api_config.CLASSIFICATION_MODEL_PATHS, api_config.CLASSIFICATION_CLASS_NAMES)


def _make_error_object(error_code, error_message):
    # here we make a dict that the request_processing_function can return to the endpoint function
    # to notify it of an error
    return {
        'error_message': error_message,
        'error_code': error_code
    }

def _make_error_response(error_code, error_message):
    return make_response(jsonify({'error': error_message}), error_code)


def _detect_process_request_data(request):
    """
    Processes the request data to the /detect endpoint. It loads data or files into a
    dictionary for access in the API function. It is passed as a parameter to the API setup.
    Args:
        request: The request object from the @ai4e_service.api_sync_func

    Returns:
        A dict with the parameters parsed from user input
    """
    files = request.files
    params = request.args

    # check that the content uploaded is not too big
    # request.content_length is the length of the total payload
    # also will not proceed if cannot find content_length, hence in the else we exceed the max limit
    content_length = request.content_length
    if not content_length:
        return _make_error_object(411, 'No image(s) are sent, or content length cannot be determined.')
    if content_length > api_config.MAX_CONTENT_LENGTH_IN_MB * 1024 * 1024:
        return _make_error_object(413, ('Payload size {:.2f} MB exceeds the maximum allowed of {} MB. '
                    'Please upload fewer or more compressed images.').format(
            content_length / (1024 * 1024), api_config.MAX_CONTENT_LENGTH_IN_MB))

    render_boxes = True if params.get('render', '') in ['True', 'true'] else False

    # validate detection confidence value
    if 'confidence' in params:
        detection_confidence = float(params['confidence'])
        print('runserver, post_detect_sync, user specified detection confidence: ', detection_confidence)  # TODO delete
        if detection_confidence < 0.0 or detection_confidence > 1.0:
            return _make_error_object(400, 'Detection confidence {} is invalid. Needs to be between 0.0 and 1.0.'.format(
                detection_confidence))
    else:
        detection_confidence = api_config.DEFAULT_DETECTION_CONFIDENCE
    log.log_info('detection confidence', detection_confidence)

    # check that the number of images is acceptable
    num_images = sum([1 if file.content_type in api_config.IMAGE_CONTENT_TYPES else 0 for file in files.values()])
    print('runserver, post_detect_sync, number of images received: ', num_images)
    log.log_info('number of images received', num_images)

    if num_images > api_config.MAX_IMAGES_ACCEPTED:
        return _make_error_object(413, 'Too many images. Maximum number of images that can be processed in one call is {}.'.format(api_config.MAX_IMAGES_ACCEPTED))
    elif num_images == 0:
        return _make_error_object(400, 'No image(s) of accepted types (image/jpeg, image/png, application/octet-stream) received.')

    # check if classification is requested and if so, which classifier to use
    if 'classification' in params:
        classification = params['classification']

        if classification not in api_config.CLASSIFICATION_CLASS_NAMES.keys():
            supported = str(list(api_config.CLASSIFICATION_CLASS_NAMES.keys())
                                        ).replace('[', '').replace(']', '')

            error_message = 'Classification name provided is not supported, The classifiers supported are {}'.format(supported)
            return _make_error_object(400, error_message)
    else:
        classification = None

    # read input images and parameters
    try:
        print('runserver, _detect_process_request_data, reading input images...')
        images, image_names = [], []
        for k, file in files.items():
            # file of type SpooledTemporaryFile has attributes content_type and a read() method
            if file.content_type in api_config.IMAGE_CONTENT_TYPES:
                images.append(viz_utils.load_image(file))
                image_names.append(k)
    except Exception as e:
        log.log_exception('Error reading the images: ' + str(e))
        return _make_error_object(500, 'Error reading the images: ' + str(e))

    return {
        'render_boxes': render_boxes,
        'detection_confidence': detection_confidence,
        'images': images,
        'image_names': image_names,
        'classification': classification
    }


def _convert_numpy_floats(np_array):
    new = []
    for i in np_array:
        new.append(float(i))
    return new


@ai4e_service.api_sync_func(api_path='/detect',
                            methods=['POST'],
                            request_processing_function=_detect_process_request_data,  # data process function
                            # if the number of requests exceed this limit, a 503 is returned to the caller.
                            maximum_concurrent_requests=2,
                            trace_name='post:detect_sync')
def detect_sync(*args, **kwargs):
    # check if the request_processing_function had an error while parsing user specified parameters
    if kwargs.get('error_code', None) is not None:
        return _make_error_response(kwargs.get('error_code'), kwargs.get('error_message'))

    render_boxes = kwargs.get('render_boxes')
    classification = kwargs.get('classification')
    detection_confidence = kwargs.get('detection_confidence')
    images = kwargs.get('images')
    image_names = kwargs.get('image_names')

    detection_results = []
    inference_time_detector = []

    try:
        print('runserver, post_detect_sync, scoring images...')

        for image_name, image in zip(image_names, images):
            start_time = time.time()

            result = detector.generate_detections_one_image(image, image_name)
            detection_results.append(result)

            elapsed = time.time() - start_time
            inference_time_detector.append(elapsed)

    except Exception as e:
        print('Error performing detection on the images: ' + str(e))
        log.log_exception('Error performing detection on the images: ' + str(e))
        return _make_error_response(500, 'Error performing detection on the images: ' + str(e))

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

    except Exception as e:
        print('Error consolidating the detection boxes: ' + str(e))
        log.log_exception('Error consolidating the detection boxes: ' + str(e))
        return _make_error_response(500, 'Error consolidating the detection boxes: ' + str(e))

    # classification
    classification_result = {}
    # TODO
    # try:
    #     if classification:
    #         print('runserver, classification...')
    #         tic = datetime.now()
    #         classification_result = classifier.classify_boxes(images, image_names, result, classification)
    #         toc = datetime.now()
    #         classification_inference_duration = toc - tic
    #         print('runserver, classification, classifcation inference duraction: {}' \
    #               .format({classification_inference_duration}))
    #
    #     else:
    #         classification_result = {}
    #
    # except Exception as e:
    #     print('Error performing classification on the images: ' + str(e))
    #     log.log_exception('Error performing classification on the images: ' + str(e))
    #     abort(500, 'Error performing classification on the images: ' + str(e))

    # return results; optionally render the detections on the images and send the annotated images back
    try:
        print('runserver, post_detect_sync, rendering and sending images back...')
        fields = {
            'detection_result': ('detection_result', json.dumps(filtered_results), 'application/json'),
            'classification_result': ('classification_result', json.dumps(classification_result), 'application/json')
        }

        if render_boxes:
            for image_name, image, result in zip(image_names, images, detection_results):
                detections = result.get('detections', None)
                if detections is None:
                    continue
                viz_utils.render_detection_bounding_boxes(detections, image, confidence_threshold=detection_confidence)

                output_img_stream = BytesIO()
                image.save(output_img_stream, format='jpeg')
                output_img_stream.seek(0)
                fields[image_name] = (image_name, output_img_stream, 'image/jpeg')

        m = MultipartEncoder(fields=fields)

        if len(inference_time_detector) > 0:
            mean_inference_time_detector = sum(inference_time_detector) / len(inference_time_detector)
        else:
            mean_inference_time_detector = -1

        log.log_info('detector mean inference time', mean_inference_time_detector,
                     additionalProperties={
                         'detector mean inference time': str(mean_inference_time_detector),
                         # TODO 'classification mean inference time': str(''),
                         'num_images': len(image_names),
                         'render_boxes': render_boxes,
                         'detection_confidence': detection_confidence
                     })
        return Response(m.to_string(), mimetype=m.content_type)
    except Exception as e:
        print('Error returning result or rendering the detection boxes: ' + str(e))
        log.log_exception('Error returning result or rendering the detection boxes: ' + str(e))
        return _make_error_response(500, 'Error returning result or rendering the detection boxes: ' + str(e))


@ai4e_service.api_sync_func(api_path='/detector_model_version',
                            methods=['GET'],
                            maximum_concurrent_requests=1000,
                            trace_name='get:get_model_detector_version')
def get_model_detector_version(*args, **kwargs):
    try:
        return api_config.DETECTOR_MODEL_VERSION
    except Exception as e:
        return 'Detection model version unknown. Error: {}'.format(str(e))


# TODO
# @ai4e_service.api_sync_func(api_path='/supported_classifiers',
#                             methods=['GET'],
#                             maximum_concurrent_requests=1000,
#                             trace_name='get:get_supported_classifiers')
# def get_supported_classifiers(*args, **kwargs):
#     try:
#         return list(api_config.CLASSIFICATION_CLASS_NAMES.keys())
#     except Exception as e:
#         return 'Supported classifiers unknown. Error: {}'.format(str(e))


if __name__ == '__main__':
    app.run()

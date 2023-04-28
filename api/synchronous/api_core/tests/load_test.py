
import os
import json
import io
import random
import requests

from PIL import Image
from multiprocessing import Pool
from datetime import datetime
from requests_toolbelt import MultipartEncoder
from requests_toolbelt.multipart import decoder


ip_address = '100.100.200.200'
port = 5050

base_url = 'http://{}:{}/v1/camera-trap/sync/'.format(ip_address, port)  


def call_api(args):
    start = datetime.now()
    
    index, url, params, data, headers = args['index'],args['url'], args['params'], args['data'], args['headers']
    print('calling api: {} starttime: {}'.format(index, start))

    response = requests.post(url, params=params, data=data, headers=headers)
    elapsed_time = datetime.now() - start
    print('\napi {} status code: {}, elapsed time in seconds {}'.format(index, response.status_code, elapsed_time.total_seconds()))
    
    get_detections(response)   
    return response

def get_detections(response):
    results = decoder.MultipartDecoder.from_response(response)
    text_results = {}
    images = {}
    for part in results.parts:
        # part is a BodyPart object with b'Content-Type', and b'Content-Disposition', the later includes 'name' and 'filename' info
        headers = {}
        for k, v in part.headers.items():
            headers[k.decode(part.encoding)] = v.decode(part.encoding)
       
        if headers.get('Content-Type', None) == 'application/json':
            text_result = json.loads(part.content.decode())

    print(text_result)


def test_load(num_requests, params, max_images=1):
    requests = []
    
    # read the images anew for each request
    index = 0
    for i in range(num_requests):
        index += 1
        files = {}
        sample_input_dir = '../../../api/synchronous/sample_input/test_images'

        image_files = os.listdir(sample_input_dir)
        random.shuffle(image_files)

        num_images = 0
        for i, image_name in enumerate(image_files):
            if not image_name.lower().endswith('.jpg'):
                continue

            if num_images >= max_images:
                break
            else:
                num_images += 1

            img_path = os.path.join(sample_input_dir, image_name)
            with open(img_path, 'rb') as f:
                content = f.read()
            files[image_name] = (image_name, content, 'image/jpeg')

        m = MultipartEncoder(fields=files)
        args = {
            'index': index,
            'url': base_url + 'detect',
            'params': params,
            'data': m,
            'headers': {'Content-Type': m.content_type}
        }
        requests.append(args)
    
    print('starting', num_requests, 'threads...')
    # images are read and in each request by the time we call the API in map()
    with Pool(num_requests) as pool:
        results = pool.map(call_api, requests)

    return results


if __name__ == "__main__":
    params = {
    'min_confidence': 0.05,
    'min_rendering_confidence': 0.2,
    'render': True
    }
    
    num_requests = 10
    max_images = 1

    start = datetime.now()
    responses = test_load(num_requests, params, max_images=max_images)
    end = datetime.now()
    total_time = end - start
    print('Total time for {} requests: {}'.format(num_requests, total_time))
import io
import json
import os
import random

from locust import HttpUser, task, between
from requests_toolbelt.multipart import decoder
from PIL import Image


"""
Load testing using Locust.

Installation instructions: https://docs.locust.io/en/stable/quickstart.html

Once Locust is installed, to run the tests:
locust --host=http://example.com/api/

and visit http://127.0.0.1:8089 in a browser (local testing)
"""

sample_input_dir = './sample_input/test_images'
test_image_names = sorted(os.listdir(sample_input_dir))
test_image_paths = [os.path.join(sample_input_dir, image_name) for image_name in test_image_names if
                    image_name.lower().endswith('.jpg')]

params = {
    'confidence': 0.8,
    'render': True
}

headers = {
    'Ocp-Apim-Subscription-Key': os.environ.get('API_KEY', '')
}

class QuickstartUser(HttpUser):
    wait_time = between(5, 9) #simulated users wait between 5 and 9 seconds after each task is executed

    @task
    def request_detection(self):
        num_to_upload = random.randint(1, 8)  # API accepts 1 to 8 images

        files = {}
        for i in range(num_to_upload):
            image_name, file_item = QuickstartUser.get_test_image()
            files[image_name] = file_item

        response = self.client.post('detect', name='detect:num_images:{}'.format(num_to_upload),
                                                params=params,
                                                files=files,
                                                headers=headers)
        QuickstartUser.open_detection_results(response)

    @staticmethod
    def get_test_image():
        image_i = random.randint(0, 9)  # we have 10 test images
        image_name = test_image_names[image_i]
        image_path = test_image_paths[image_i]

        return image_name, (image_name, open(image_path, 'rb'), 'image/jpeg')

    @staticmethod
    def open_detection_results(response):
        results = decoder.MultipartDecoder.from_response(response)

        text_results = {}
        images = {}
        for part in results.parts:
            # part is a BodyPart object with b'Content-Type', and b'Content-Disposition', the later includes 'name' and 'filename' info
            headers = {}
            for k, v in part.headers.items():
                headers[k.decode(part.encoding)] = v.decode(part.encoding)
            if headers.get('Content-Type', None) == 'image/jpeg':
                # images[part.headers['filename']] = part.content
                c = headers.get('Content-Disposition')
                image_name = c.split('name="')[1].split('"')[
                    0]  # somehow all the filename and name info is all in one string with no obvious forma
                image = Image.open(io.BytesIO(part.content))

                images[image_name] = image

            elif headers.get('Content-Type', None) == 'application/json':
                text_result = json.loads(part.content.decode())

        print(text_result)
        for img_name, img in sorted(images.items()):
            print(img_name)
            img.close()
        print()

import json
import os
import sys
import unittest

import requests
from requests_toolbelt.multipart import decoder


# get rid of formatting differences
API_RESULT = json.dumps(json.loads('''
{"detection_result": {"S1_D04_R6_PICT0020.JPG": [[0.006578,
    0,
    0.9797779999999999,
    0.9401,
    0.998,
    1]],
  "S1_D04_R6_PICT0021.JPG": [[0.01237, 0, 0.93057, 0.9344, 0.998, 1]]},
 "classification_result": {}}
'''))


class TestSynchronousAPI(unittest.TestCase):
    API_URL = 'http://boto.eastus.cloudapp.azure.com:6002/v1/camera-trap/sync/'
    API_KEY = ''

    def setUp(self):
        self.api_url = TestSynchronousAPI.API_URL  # hack
        self.headers = {
            'Ocp-Apim-Subscription-Key': TestSynchronousAPI.API_KEY  # insert API key if testing production endpoint
        }
        self.sample_input_dir = './sample_input/test_images'

    def test_detector_version(self):
        version_info = requests.get(self.api_url + 'detector_model_version', headers=self.headers)
        self.assertEqual(version_info.status_code, 200)
        self.assertEqual(version_info.text, 'v4.1.0')

    def test_detector_output(self):
        params = {
            'confidence': 0.8,
            'render': False
        }

        num_images_to_upload = 2

        files = {}
        open_files = []
        for i, image_name in enumerate(sorted(os.listdir(self.sample_input_dir))):
            if not image_name.lower().endswith('.jpg'):
                continue

            if len(open_files) >= num_images_to_upload:
                break

            fd = open(os.path.join(self.sample_input_dir, image_name), 'rb')
            open_files.append(fd)
            files[image_name] = (image_name, fd, 'image/jpeg')

        r = requests.post(self.api_url + 'detect',
                          params=params,
                          files=files, headers=self.headers)

        if not r.ok:
            print('Response not okay, reason and text:')
            print(r.reason)
            print(r.text)

        self.assertEqual(r.status_code, 200)
        print(f'\nTime spent on the call in seconds: {r.elapsed.total_seconds()}')

        for fd in open_files:
            fd.close()

        # compare result
        res = decoder.MultipartDecoder.from_response(r)

        results = {}
        # images = {}

        for part in res.parts:
            # part is a BodyPart object with b'Content-Type', and b'Content-Disposition', the later includes 'name' and 'filename' info
            headers = {}
            for k, v in part.headers.items():
                headers[k.decode(part.encoding)] = v.decode(part.encoding)

            # if headers.get('Content-Type', None) == 'image/jpeg':
            #     # images[part.headers['filename']] = part.content
            #     c = headers.get('Content-Disposition')
            #     image_name = c.split('name="')[1].split('"')[
            #         0]  # somehow all the filename and name info is all in one string with no obvious forma
            #     image = Image.open(io.BytesIO(part.content))
            #     images[image_name] = image
            if headers.get('Content-Type', None) == 'application/json':
                content_disposition = headers.get('Content-Disposition', '')
                if 'detection_result' in content_disposition:
                    results['detection_result'] = json.loads(part.content.decode())
                elif 'classification_result' in content_disposition:
                    results['classification_result'] = json.loads(part.content.decode())
        results_string = json.dumps(results)
        self.assertEqual(results_string, API_RESULT)


    def test_detector_too_many_images(self):
        pass

    def test_detector_wrong_image_file_type(self):
        pass

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':

    # https://stackoverflow.com/questions/11380413/python-unittest-passing-arguments
    if len(sys.argv) == 2:
        TestSynchronousAPI.API_URL = sys.argv.pop()
        assert TestSynchronousAPI.API_URL.endswith('/')
    elif len(sys.argv) == 3:
        TestSynchronousAPI.API_KEY = sys.argv.pop()
        TestSynchronousAPI.API_URL = sys.argv.pop()
        assert TestSynchronousAPI.API_URL.endswith('/')
    else:
        print('Need to input the API URL and if testing production version, the API key')
        sys.exit(1)

    print(f'\nAPI URL is: {TestSynchronousAPI.API_URL}\n')
    unittest.main(verbosity=2)

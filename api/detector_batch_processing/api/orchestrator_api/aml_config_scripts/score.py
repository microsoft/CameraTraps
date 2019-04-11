print('score.py, beginning - NEW')

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime
from urllib import parse

from azure.storage.blob import BlockBlobService
from azureml.core.model import Model
from azureml.core.run import Run

from tf_detector import TFDetector


class BatchScorer:

    def __init__(self, **kwargs):
        print('BatchScorer, __init__()')

        model_path = kwargs.get('model_path')
        self.detector = TFDetector(model_path)

        self.job_id = kwargs.get('job_id')

        self.input_container_sas = kwargs.get('input_container_sas')
        self.output_dir = kwargs.get('output_dir')

        self.detection_threshold = kwargs.get('detection_threshold')
        self.batch_size = kwargs.get('batch_size')

        self.image_ids = kwargs.get('image_ids')
        self.images = []

        self.detections = []
        self.image_ids = []  # all the IDs of the images that PIL successfully opened
        self.failed_images = []  # list of image_ids that failed to open

    @staticmethod
    def get_account_from_uri(sas_uri):
        url_parts = parse.urlsplit(sas_uri)
        loc = url_parts.netloc
        return loc.split('.')[0]

    @staticmethod
    def get_sas_key_from_uri(sas_uri):
        """Get the query part of the SAS token that contains permissions, access times and
        signature.

        Args:
            sas_uri: Azure blob storage SAS token

        Returns: Query part of the SAS token.
        """
        url_parts = parse.urlsplit(sas_uri)
        return url_parts.query

    @staticmethod
    def get_container_from_uri(sas_uri):
        url_parts = parse.urlsplit(sas_uri)

        raw_path = url_parts.path[1:]
        container = raw_path.split('/')[0]

        return container

    def download_images(self):
        print('BatchScorer, download_images()')

        blob_service = BlockBlobService(
            account_name=BatchScorer.get_account_from_uri(self.input_container_sas),
            sas_token=BatchScorer.get_sas_key_from_uri(self.input_container_sas))
        container_name = BatchScorer.get_container_from_uri(self.input_container_sas)

        for image_id in image_ids:
            try:
                stream = io.BytesIO()
                _ = blob_service.get_blob_to_stream(container_name, image_id, stream)
                image = TFDetector.open_image(stream)
                image = TFDetector.resize_image(image)  # image loaded here
                self.images.append(image)
                self.image_ids.append(image_id)
            except Exception as e:
                print('score.py, failed to download or open image {}, exception: {}'.format(image_id, str(e)))
                self.failed_images.append(image_id)
                continue

    def score(self):
        print('BatchScorer, score()')
        # self.image_ids does not include any failed images; self.image_ids is overwritten here
        self.detections, self.image_ids, failed_images_during_detection = \
            self.detector.generate_detections_batch(
                self.images, self.image_ids, self.batch_size, self.detection_threshold)

        self.failed_images.extend(failed_images_during_detection)

    def write_output(self):
        """Outputs csv where each row is image_path, True if there is detection,
           "[ [y1, x1, y2, x2, confidence], [...] ]"
        """
        print('BatchScorer, write_output()')
        detections_path = os.path.join(self.output_dir, 'detections_{}.csv'.format(self.job_id))

        with open(detections_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['image_path', 'max_confidence', 'detections'])
            for image_id, detections in zip(self.image_ids, self.detections):
                # has_detections = len(detections) > 0
                max_conf = 0.0  # detections should be sorted from higih to low confidence but to be sure
                for detection in detections:
                    conf = detection[4]
                    if conf > max_conf:
                        max_conf = conf
                writer.writerow([image_id, max_conf, str(detections)])

        failed_path = os.path.join(self.output_dir, 'failures_{}.csv'.format(self.job_id))
        with open(failed_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for i in self.failed_images:
                writer.writerow([i])


if __name__ == '__main__':
    print('score.py, in __main__')

    run = Run.get_context()
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description='Batch score images using an object detection model.')
    parser.add_argument('--job_id', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--input_container_sas', required=True)
    # API's internal container where the list of image paths is stored
    parser.add_argument('--internal_dir', required=True)

    # a json file containing a list of image paths this job should process
    parser.add_argument('--begin_index', type=int, default=0)  # if not provided, processing all images in this job
    parser.add_argument('--end_index', type=int, default=0)

    parser.add_argument('--output_dir', required=True)  # API's AML container storing jobs' outputs

    parser.add_argument('--detection_threshold', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    print('score.py, args to the scripts are', str(args))

    os.makedirs(args.output_dir, exist_ok=True)
    print('score.py, output_dir', args.output_dir)

    # get model from model registry
    model_path = Model.get_model_path(args.model_name)

    print('score.py, model_path', model_path)

    job_id = str(args.job_id)
    print('score.py, job_id', job_id)

    assert job_id.startswith('request'), 'Error in score.py: job_id does not start with "request"'
    request_id = job_id.split('request')[1].split('_jobindex')[0]

    list_images_path = os.path.join(args.internal_dir, request_id, '{}_images.json'.format(request_id))
    print('score.py, list_images_path: ', list_images_path)

    list_images = json.load(open(list_images_path))
    print('score.py, len(list_images)', len(list_images))

    if len(list_images) == 0:
        sys.exit(0)

    # exclude the end_index; default is to process all images in this request
    begin_index, end_index = 0, len(list_images)
    if args.begin_index == 0 and args.end_index == 0:
        print('score.py, both begin_index and end_index are zero. Exiting...')
        sys.exit(0)

    if args.begin_index >= 0 and args.begin_index < end_index:
        begin_index, end_index = args.begin_index, args.end_index
    else:
        raise ValueError('Indices {} and {} are not allowed for the image list of length {}'.format(
            args.begin_index, args.end_index, len(list_images)
        ))
    image_ids = list_images[begin_index:end_index]
    print('score.py, processing from index {} to {}. Length of images_shard is {}.'.format(
        begin_index, end_index, len(image_ids)))

    scorer = BatchScorer(model_path=model_path,
                         input_container_sas=args.input_container_sas,
                         job_id=args.job_id,
                         image_ids=image_ids,
                         output_dir=args.output_dir,
                         detection_threshold=args.detection_threshold,
                         batch_size=args.batch_size)

    try:
        scorer.download_images()
    except Exception as e:
        raise RuntimeError('Exception in scorer.download_images(): {}'.format(str(e)))

    try:
        scorer.score()
    except Exception as e:
        raise RuntimeError('Exception in scorer.score(): {}'.format(str(e)))

    try:
        scorer.write_output()  # write the results obtained thus far

        run_duration_seconds = datetime.now() - start_time
        print('score.py - run_duration_seconds (load, inference, writing output):', run_duration_seconds)
        run.log('run_duration_seconds', str(run_duration_seconds))
    except Exception as e:
        raise RuntimeError('Exception in writing output or logging to AML: {}'.format(str(e)))

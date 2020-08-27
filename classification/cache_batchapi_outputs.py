"""
Script to cache Batch Detection API outputs.
"""
import argparse
import json
import os
from typing import Optional

import requests

from classification.detect_and_crop import cache_detections
from api.batch_processing.data_preparation.prepare_api_submission import (
    TaskStatus, Task)

def main(json_path: str,
         dataset: str,
         detector_output_cache_base_dir: str,
         detector_version: Optional[str]) -> None:
    """Main."""
    with open(json_path, 'r') as f:
        response = json.load(f)

    # task finished successfully
    status = TaskStatus(response['Status']['request_status'])
    assert status == TaskStatus.COMPLETED

    # parse the task ID
    task_id = response['TaskId']

    message = response['Status']['message']
    detections_url = message['output_file_urls']['detections']
    assert detections_url.split('/')[-2] == task_id

    # print info about missing and failed images
    task = Task(name=task_id, task_id=task_id)
    task.response = response
    task.status = status
    task.get_missing_images(verbose=True)

    # get the detections and add to the detections cache
    detections = requests.get(detections_url).json()
    api_detector_version = detections['info']['detector'].split('v')[1]
    if detector_version is not None:
        assert api_detector_version == detector_version
    detector_output_cache_dir = os.path.join(
        detector_output_cache_base_dir, f'v{api_detector_version}')
    msg = cache_detections(
        detections=detections, dataset=dataset,
        detector_output_cache_dir=detector_output_cache_dir)
    print(msg)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Caches detector outputs.')
    parser.add_argument(
        'json_response',
        help='path to JSON file containing response of Batch Detection API')
    parser.add_argument(
        '-d', '--dataset', required=True,
        help='(required) name of dataset corresponding to the API task')
    parser.add_argument(
        '-c', '--detector-output-cache-dir', required=True,
        help='(required) path to directory where detector outputs are cached')
    parser.add_argument(
        '-v', '--detector-version',
        help='detector version string, e.g., "4.1", inferred from detections '
             'file if not given')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(json_path=args.json_response,
         dataset=args.dataset,
         detector_output_cache_base_dir=args.detector_output_cache_dir,
         detector_version=args.detector_version)

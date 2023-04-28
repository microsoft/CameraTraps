"""
Script to cache Batch Detection API outputs.

This script can handle either the Batch Detection API JSON Response or the
detections JSON.

Batch Detection API Response format:
    {
        "Status": {
            "request_status": "completed",
            "message": {
                "num_failed_shards": 0,
                "output_file_urls": {
                    "detections": "https://url/to/detections.json",
                    "failed_images": "https://url/to/failed_images.json",
                    "images": https://url/to/images.json",
                }
            },
        },
        "Endpoint": "/v3/camera-trap/detection-batch/request_detections",
        "TaskId": "ea26326e-7e0d-4524-a9ea-f57a5799d4ba"
    }

Detections JSON format:
    {
        "info": {...}
        "detection_categories": {...}
        "classification_categories": {...}
        "images": [
            {
                "file": "path/from/base/dir/image1.jpg",
                "max_detection_conf": 0.926,
                "detections": [{
                        "category": "1",
                        "conf": 0.061,
                        "bbox": [0.0451, 0.1849, 0.3642, 0.4636]
                }]
            }
        ]
    }


Batch Detection API Output Format:
github.com/ecologize/CameraTraps/tree/master/api/batch_processing#api-outputs
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from typing import Any, Optional

import requests

from api.batch_processing.data_preparation.prepare_api_submission import (
    TaskStatus, Task)
from api.batch_processing.postprocessing.combine_api_outputs import (
    combine_api_output_dictionaries)


def cache_json(json_path: str,
               is_detections: bool,
               dataset: str,
               detector_output_cache_base_dir: str,
               detector_version: Optional[str]) -> None:
    """
    Args:
        json_path: str, path to JSON file
        is_detections: bool, True if <json_path> is a detections JSON file,
            False if <json_path> is a API response JSON file
        dataset: str
        detector_output_cache_base_dir: str
        detector_version: str
    """
    with open(json_path, 'r') as f:
        js = json.load(f)

    if is_detections:
        detections = js

    else:
        response = js

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

        # get the detections
        detections = requests.get(detections_url).json()

    # add detections to the detections cache
    api_det_version = detections['info']['detector'].rsplit('v', maxsplit=1)[1]
    if detector_version is not None:
        assert api_det_version == detector_version
    detector_output_cache_dir = os.path.join(
        detector_output_cache_base_dir, f'v{api_det_version}')
    msg = cache_detections(
        detections=detections, dataset=dataset,
        detector_output_cache_dir=detector_output_cache_dir)
    print(msg)


def cache_detections(detections: Mapping[str, Any], dataset: str,
                     detector_output_cache_dir: str) -> str:
    """
    Args:
        detections: dict, represents JSON output of detector
        dataset: str, name of dataset
        detector_output_cache_dir: str, path to folder where detector outputs
            are cached, stored as 1 JSON file per dataset, directory must
            already exist

    Returns: str, message
    """
    # combine detections with cache
    dataset_cache_path = os.path.join(
        detector_output_cache_dir, f'{dataset}.json')
    merged_dataset_cache: Mapping[str, Any]
    if os.path.exists(dataset_cache_path):
        with open(dataset_cache_path, 'r') as f:
            dataset_cache = json.load(f)
        merged_dataset_cache = combine_api_output_dictionaries(
            input_dicts=[dataset_cache, detections], require_uniqueness=False)
        msg = f'Merging detection output with {dataset_cache_path}'
    else:
        merged_dataset_cache = detections
        msg = ('No cached detection outputs found. Saving detection output to '
               f'{dataset_cache_path}')

    # write combined detections back out to cache
    with open(dataset_cache_path, 'w') as f:
        json.dump(merged_dataset_cache, f, indent=1)
    return msg


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Caches detector outputs.')
    parser.add_argument(
        'json_file',
        help='path to JSON file containing response of Batch Detection API')
    parser.add_argument(
        '-f', '--format', choices=['response', 'detections'], required=True,
        help='(required) whether <json_file> is a Batch API response or a '
             'detections JSON file')
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
    cache_json(
        json_path=args.json_file,
        is_detections=(args.format == 'detections'),
        dataset=args.dataset,
        detector_output_cache_base_dir=args.detector_output_cache_dir,
        detector_version=args.detector_version)

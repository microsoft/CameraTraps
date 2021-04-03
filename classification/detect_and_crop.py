r"""
Run MegaDetector on images via Batch API, then save crops of the detected
bounding boxes.

The input to this script is a "queried images" JSON file, whose keys are paths
to images and values are dicts containing information relevant for training
a classifier, including labels and (optionally) ground-truth bounding boxes.
The image paths are in the format `<dataset-name>/<blob-name>` where we assume
that the dataset name does not contain '/'.

{
    "caltech/cct_images/59f79901-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  # class from dataset
        "bbox": [{"category": "animal",
                  "bbox": [0, 0.347, 0.237, 0.257]}],   # ground-truth bbox
        "label": ["monutain_lion"]  # labels to use in classifier
    },
    "caltech/cct_images/59f5fe2b-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  # class from dataset
        "label": ["monutain_lion"]  # labels to use in classifier
    },
    ...
}

We assume that no image contains over 100 bounding boxes, and we always save
crops as RGB .jpg files for consistency. For each image, each bounding box is
cropped and saved to a file with a suffix "___cropXX.jpg" (ground truth bbox) or
"___cropXX_mdvY.Y.jpg" (detected bbox) added to the filename of the original
image. "XX" ranges from "00" to "99" and "Y.Y" indicates the MegaDetector
version. If an image has ground truth bounding boxes, we assume that they are
exhaustive--i.e., there are no other objects of interest, so we don't need to
run MegaDetector on the image. If an image does not have ground truth bounding
boxes, we run MegaDetector on the image and label the detected boxes in order
from 00 up to 99. Based on the given confidence threshold, we may skip saving
certain bounding box crops, but we still increment the bounding box number for
skipped boxes.

Example cropped image path (with ground truth bbox from MegaDB)
    "path/to/crops/image.jpg___crop00.jpg"
Example cropped image path (with MegaDetector bbox)
    "path/to/crops/image.jpg___crop00_mdv4.1.jpg"

By default, the images are cropped exactly per the given bounding box
coordinates. However, if square crops are desired, pass the --square-crops
flag. This will always generate a square crop whose size is the larger of the
bounding box width or height. In the case that the square crop boundaries exceed
the original image size, the crop is padded with 0s.

This script currently only supports running MegaDetector via the Batch Detection
API. See the classification README for instructions on running MegaDetector
locally. If running the Batch Detection API, set the following environment
variables for the Azure Blob Storage container in which we save the intermediate
task lists:

    BATCH_DETECTION_API_URL                  # API URL
    CLASSIFICATION_BLOB_STORAGE_ACCOUNT      # storage account name
    CLASSIFICATION_BLOB_CONTAINER            # container name
    CLASSIFICATION_BLOB_CONTAINER_WRITE_SAS  # SAS token, without leading '?'
    DETECTION_API_CALLER                     # allow-listed API caller

This script allows specifying a directory where MegaDetector outputs are cached
via the --detector-output-cache-dir argument. This directory must be
organized as
    <cache-dir>/<MegaDetector-version>/<dataset-name>.json

    Example: If the `cameratrapssc/classifier-training` Azure blob storage
    container is mounted to the local machine via blobfuse, it may be used as
    a MegaDetector output cache directory by passing
        "cameratrapssc/classifier-training/mdcache/"
    as the value for --detector-output-cache-dir.

This script outputs either 1 or 3 files, depending on if the Batch Detection API
is run:

- <output_dir>/detect_and_crop_log_{timestamp}.json
    log of images missing detections and images that failed to properly
    download and crop
- <output_dir>/batchapi_tasklists/{task_id}.json
    (if --run-dectector) task lists uploaded to the Batch Detection API
- <output_dir>/batchapi_response/{task_id}.json
    (if --run-dectector) task status responses for completed tasks

Example command:

    python detect_and_crop.py \
        base_logdir/queried_images.json \
        base_logdir \
        --detector-output-cache-dir /path/to/classifier-training/mdcache \
        --detector-version 4.1 \
        --run-detector --resume-file base_logdir/resume.json \
        --cropped-images-dir /path/to/crops --square-crops --threshold 0.9 \
        --save-full-images --images-dir /path/to/images --threads 50
"""
from __future__ import annotations

import argparse
from collections.abc import Collection, Iterable, Mapping, Sequence
from concurrent import futures
from datetime import datetime
import json
import os
import pprint
import time
from typing import Any, Optional

from azure.storage.blob import ContainerClient
import requests
from tqdm import tqdm

from api.batch_processing.data_preparation.prepare_api_submission import (
    BatchAPIResponseError, Task, TaskStatus, divide_list_into_tasks)
from classification.cache_batchapi_outputs import cache_detections
from classification.crop_detections import load_and_crop
from data_management.megadb import megadb_utils
import path_utils  # from ai4eutils
import sas_blob_utils  # from ai4eutils


def main(queried_images_json_path: str,
         output_dir: str,
         detector_version: str,
         detector_output_cache_base_dir: str,
         run_detector: bool,
         resume_file_path: Optional[str],
         cropped_images_dir: Optional[str],
         save_full_images: bool,
         square_crops: bool,
         check_crops_valid: bool,
         confidence_threshold: float,
         images_dir: Optional[str],
         threads: int) -> None:
    """
    Args:
        queried_images_json_path: str, path to output of json_validator.py
        detector_version: str, detector version string, e.g., '4.1',
            see {batch_detection_api_url}/supported_model_versions,
            determines the subfolder of detector_output_cache_base_dir in
            which to find and save detector outputs
        detector_output_cache_base_dir: str, path to local directory
            where detector outputs are cached, 1 JSON file per dataset
        cropped_images_dir: str, path to local directory for saving crops of
            bounding boxes
        run_detector: bool, whether to run Batch Detection API, or to skip
            running the detector entirely
        output_dir: str, path to directory to save outputs, see module docstring
        save_full_images: bool, whether to save downloaded images to images_dir,
            images_dir must be given if save_full_images=True
        square_crops: bool, whether to crop bounding boxes as squares
        check_crops_valid: bool, whether to load each crop to ensure the file is
            valid (i.e., not truncated)
        confidence_threshold: float, only crop bounding boxes above this value
        images_dir: optional str, path to local directory where images are saved
        threads: int, number of threads to use for downloading images
        resume_file_path: optional str, path to save JSON file with list of info
            dicts on running tasks, or to resume from running tasks, only used
            if run_detector=True
    """

    # This dictionary will get written out at the end of this process; store
    # diagnostic variables here
    log: dict[str, Any] = {}

    # error checking
    assert 0 <= confidence_threshold <= 1
    if save_full_images:
        assert images_dir is not None
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
            print(f'Created images_dir at {images_dir}')

    with open(queried_images_json_path, 'r') as f:
        js = json.load(f)
    detector_output_cache_dir = os.path.join(
        detector_output_cache_base_dir, f'v{detector_version}')
    if not os.path.exists(detector_output_cache_dir):
        os.makedirs(detector_output_cache_dir)
        print(f'Created directory at {detector_output_cache_dir}')
    images_without_ground_truth_bbox = [k for k in js if 'bbox' not in js[k]]
    images_to_detect, detection_cache, categories = filter_detected_images(
        potential_images_to_detect=images_without_ground_truth_bbox,
        detector_output_cache_dir=detector_output_cache_dir)
    print(f'{len(images_to_detect)} images not in detection cache')

    if run_detector:
        log['images_submitted_for_detection'] = images_to_detect

        assert resume_file_path is not None
        assert not os.path.isdir(resume_file_path)
        batch_detection_api_url = os.environ['BATCH_DETECTION_API_URL']

        if os.path.exists(resume_file_path):
            tasks_by_dataset = resume_tasks(
                resume_file_path,
                batch_detection_api_url=batch_detection_api_url)
        else:
            task_lists_dir = os.path.join(output_dir, 'batchapi_tasklists')
            tasks_by_dataset = submit_batch_detection_api(
                images_to_detect=images_to_detect,
                task_lists_dir=task_lists_dir,
                detector_version=detector_version,
                account=os.environ['CLASSIFICATION_BLOB_STORAGE_ACCOUNT'],
                container=os.environ['CLASSIFICATION_BLOB_CONTAINER'],
                sas_token=os.environ['CLASSIFICATION_BLOB_CONTAINER_WRITE_SAS'],
                caller=os.environ['DETECTION_API_CALLER'],
                batch_detection_api_url=batch_detection_api_url,
                resume_file_path=resume_file_path)

        wait_for_tasks(tasks_by_dataset, detector_output_cache_dir,
                       output_dir=output_dir)

        # refresh detection cache
        print('Refreshing detection cache...')
        images_to_detect, detection_cache, categories = filter_detected_images(
            potential_images_to_detect=images_without_ground_truth_bbox,
            detector_output_cache_dir=detector_output_cache_dir)
        print(f'{len(images_to_detect)} images not in detection cache')

    log['images_missing_detections'] = images_to_detect

    if cropped_images_dir is not None:

        images_failed_dload_crop, num_downloads, num_crops = download_and_crop(
            queried_images_json=js,
            detection_cache=detection_cache,
            detection_categories=categories,
            detector_version=detector_version,
            cropped_images_dir=cropped_images_dir,
            confidence_threshold=confidence_threshold,
            save_full_images=save_full_images,
            square_crops=square_crops,
            check_crops_valid=check_crops_valid,
            images_dir=images_dir,
            threads=threads,
            images_missing_detections=images_to_detect)
        log['images_failed_download_or_crop'] = images_failed_dload_crop
        log['num_new_downloads'] = num_downloads
        log['num_new_crops'] = num_crops

    print(f'{len(images_to_detect)} images with missing detections.')
    if cropped_images_dir is not None:
        print(f'{len(images_failed_dload_crop)} images failed to download or '
              'crop.')

    # save log of bad images
    date = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., '20200722_110816'
    log_path = os.path.join(output_dir, f'detect_and_crop_log_{date}.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=1)


def load_detection_cache(detector_output_cache_dir: str,
                         datasets: Collection[str]) -> tuple[
                             dict[str, dict[str, dict[str, Any]]],
                             dict[str, str]
                         ]:
    """Loads detection cache for a given dataset. Returns empty dictionaries
    if the cache does not exist.

    Args:
        detector_output_cache_dir: str, path to local directory where detector
            outputs are cached, 1 JSON file per dataset
        datasets: list of str, names of datasets

    Returns:
        detection_cache: dict, maps dataset name to dict, which maps
            image file to corresponding entry in 'images' list from the
            Batch Detection API output. detection_cache[ds] is an empty dict
            if no cached detections were found for the given dataset ds.
        detection_categories: dict, maps str category ID to str category name
    """
    # cache of Detector outputs: dataset name => {img_path => detection_dict}
    detection_cache = {}
    detection_categories: dict[str, str] = {}

    pbar = tqdm(datasets)
    for ds in pbar:
        pbar.set_description(f'Loading dataset {ds} into detection cache')
        cache_path = os.path.join(detector_output_cache_dir, f'{ds}.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                js = json.load(f)
            detection_cache[ds] = {img['file']: img for img in js['images']}
            if len(detection_categories) == 0:
                detection_categories = js['detection_categories']
            assert detection_categories == js['detection_categories']
        else:
            tqdm.write(f'No detection cache found for dataset {ds}')
            detection_cache[ds] = {}
    return detection_cache, detection_categories


def filter_detected_images(
        potential_images_to_detect: Iterable[str],
        detector_output_cache_dir: str
        ) -> tuple[list[str],
                   dict[str, dict[str, dict[str, Any]]],
                   dict[str, str]]:
    """Checks image paths against cached Detector outputs, and prepares
    the SAS URIs for each image not in the cache.

    Args:
        potential_images_to_detect: list of str, paths to images that do not
            have ground truth bounding boxes, each path has format
            <dataset-name>/<img-filename>, where <img-filename> is the blob name
        detector_output_cache_dir: str, path to local directory where detector
            outputs are cached, 1 JSON file per dataset

    Returns:
        images_to_detect: list of str, paths to images not in the detector
            output cache, with the format <dataset-name>/<img-filename>
        detection_cache: dict, maps str dataset name to dict,
            detection_cache[dataset_name] is the 'detections' list from the
            Batch Detection API output
        detection_categories: dict, maps str category ID to str category name,
            empty dict if no cached detections are found
    """
    datasets = set(img_path[:img_path.find('/')]
                   for img_path in potential_images_to_detect)
    detection_cache, detection_categories = load_detection_cache(
        detector_output_cache_dir, datasets)

    images_to_detect = []
    for img_path in potential_images_to_detect:
        # img_path: <dataset-name>/<img-filename>
        ds, img_file = img_path.split('/', maxsplit=1)
        if img_file not in detection_cache[ds]:
            images_to_detect.append(img_path)

    return images_to_detect, detection_cache, detection_categories


def split_images_list_by_dataset(images_to_detect: Iterable[str]
                                 ) -> dict[str, list[str]]:
    """
    Args:
        images_to_detect: list of str, image paths with the format
            <dataset-name>/<image-filename>

    Returns: dict, maps dataset name to a list of image paths
    """
    images_by_dataset: dict[str, list[str]] = {}
    for img_path in images_to_detect:
        dataset = img_path[:img_path.find('/')]
        if dataset not in images_by_dataset:
            images_by_dataset[dataset] = []
        images_by_dataset[dataset].append(img_path)
    return images_by_dataset


def submit_batch_detection_api(images_to_detect: Iterable[str],
                               task_lists_dir: str,
                               detector_version: str,
                               account: str,
                               container: str,
                               sas_token: str,
                               caller: str,
                               batch_detection_api_url: str,
                               resume_file_path: str
                               ) -> dict[str, list[Task]]:
    """
    Args:
        images_to_detect: list of str, list of str, image paths with the format
            <dataset-name>/<image-filename>
        task_lists_dir: str, path to local directory for saving JSON files
            each containing a list of image URLs corresponding to an API task
        detector_version: str, MegaDetector version string, e.g., '4.1',
            see {batch_detection_api_url}/supported_model_versions
        account: str, Azure Storage account name
        container: str, Azure Blob Storage container name, where the task lists
            will be uploaded
        sas_token: str, SAS token with write permissions for the container
        caller: str, allow-listed caller
        batch_detection_api_url: str, URL to batch detection API
        resume_file_path: str, path to save resume file

    Returns: dict, maps str dataset name to list of Task objects
    """
    filtered_images_to_detect = [
        x for x in images_to_detect if path_utils.is_image_file(x)]
    not_images = set(images_to_detect) - set(filtered_images_to_detect)
    if len(not_images) == 0:
        print('Good! All image files have valid file extensions.')
    else:
        print(f'Skipping {len(not_images)} files with non-image extensions:')
        pprint.pprint(sorted(not_images))
    images_to_detect = filtered_images_to_detect

    datasets_table = megadb_utils.MegadbUtils().get_datasets_table()

    images_by_dataset = split_images_list_by_dataset(images_to_detect)
    tasks_by_dataset = {}
    for dataset, image_paths in images_by_dataset.items():
        # get SAS URL for images container
        images_sas_token = datasets_table[dataset]['container_sas_key']
        if images_sas_token[0] == '?':
            images_sas_token = images_sas_token[1:]
        images_container_url = sas_blob_utils.build_azure_storage_uri(
            account=datasets_table[dataset]['storage_account'],
            container=datasets_table[dataset]['container'],
            sas_token=images_sas_token)

        # strip image paths of dataset name
        image_blob_names = [path[path.find('/') + 1:] for path in image_paths]

        tasks_by_dataset[dataset] = submit_batch_detection_api_by_dataset(
            dataset=dataset,
            image_blob_names=image_blob_names,
            images_container_url=images_container_url,
            task_lists_dir=task_lists_dir,
            detector_version=detector_version,
            account=account, container=container, sas_token=sas_token,
            caller=caller, batch_detection_api_url=batch_detection_api_url)

    # save list of dataset names and task IDs for resuming
    resume_json = [
        {
            'dataset': dataset,
            'task_name': task.name,
            'task_id': task.id,
            'local_images_list_path': task.local_images_list_path
        }
        for dataset in tasks_by_dataset
        for task in tasks_by_dataset[dataset]
    ]
    with open(resume_file_path, 'w') as f:
        json.dump(resume_json, f, indent=1)
    return tasks_by_dataset


def submit_batch_detection_api_by_dataset(
        dataset: str,
        image_blob_names: Sequence[str],
        images_container_url: str,
        task_lists_dir: str,
        detector_version: str,
        account: str,
        container: str,
        sas_token: str,
        caller: str,
        batch_detection_api_url: str
        ) -> list[Task]:
    """
    Args:
        dataset: str, name of dataset
        image_blob_names: list of str, image blob names from the same dataset
        images_container_url: str, URL to blob storage container where images
            from this dataset are stored, including SAS token with read
            permissions if container is not public
        **see submit_batch_detection_api() for description of other args

    Returns: list of Task objects
    """
    os.makedirs(task_lists_dir, exist_ok=True)

    date = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., '20200722_110816'
    task_list_base_filename = f'task_list_{dataset}_{date}.json'

    task_list_paths, _ = divide_list_into_tasks(
        file_list=image_blob_names,
        save_path=os.path.join(task_lists_dir, task_list_base_filename))

    # complete task name: 'detect_for_classifier_caltech_20200722_110816_task01'
    task_name_template = 'detect_for_classifier_{dataset}_{date}_task{n:>02d}'
    tasks: list[Task] = []
    for i, task_list_path in enumerate(task_list_paths):
        task = Task(
            name=task_name_template.format(dataset=dataset, date=date, n=i),
            images_list_path=task_list_path, api_url=batch_detection_api_url)
        task.upload_images_list(
            account=account, container=container, sas_token=sas_token)
        task.generate_api_request(
            caller=caller,
            input_container_url=images_container_url,
            model_version=detector_version)
        print(f'Submitting task for: {task_list_path}')
        task.submit()
        print(f'- task ID: {task.id}')
        tasks.append(task)

        # HACK! Sleep for 10s between task submissions in the hopes that it
        # decreases the chance of backend JSON "database" corruption
        time.sleep(10)
    return tasks


def resume_tasks(resume_file_path: str, batch_detection_api_url: str
                 ) -> dict[str, list[Task]]:
    """
    Args:
        resume_file_path: str, path to resume file with list of info dicts on
            running tasks
        batch_detection_api_url: str, URL to batch detection API

    Returns: dict, maps str dataset name to list of Task objects
    """
    with open(resume_file_path, 'r') as f:
        resume_json = json.load(f)

    tasks_by_dataset: dict[str, list[Task]] = {}
    for info_dict in resume_json:
        dataset = info_dict['dataset']
        if dataset not in tasks_by_dataset:
            tasks_by_dataset[dataset] = []
        task = Task(name=info_dict['task_name'],
                    task_id=info_dict['task_id'],
                    images_list_path=info_dict['local_images_list_path'],
                    validate=False,
                    api_url=batch_detection_api_url)
        tasks_by_dataset[dataset].append(task)
    return tasks_by_dataset


def wait_for_tasks(tasks_by_dataset: Mapping[str, Iterable[Task]],
                   detector_output_cache_dir: str,
                   output_dir: Optional[str] = None,
                   poll_interval: int = 120) -> None:
    """Waits for the Batch Detection API tasks to finish running.

    For jobs that finish successfully, merges the output with cached detector
    outputs.

    Args:
        tasks_by_dataset: dict, maps str dataset name to list of Task objects
        detector_output_cache_dir: str, path to local directory where detector
            outputs are cached, 1 JSON file per dataset, directory must
            already exist
        output_dir: optional str, task status responses for completed tasks are
            saved to <output_dir>/batchapi_response/{task_id}.json
        poll_interval: int, # of seconds between pinging the task status API
    """
    remaining_tasks: list[tuple[str, Task]] = [
        (dataset, task) for dataset, tasks in tasks_by_dataset.items()
        for task in tasks]

    progbar = tqdm(total=len(remaining_tasks))
    while True:
        new_remaining_tasks = []
        for dataset, task in remaining_tasks:
            try:
                task.check_status()
            except (BatchAPIResponseError, requests.HTTPError) as e:
                exception_type = type(e).__name__
                tqdm.write(f'Error in checking status of task {task.id}: '
                           f'({exception_type}) {e}')
                tqdm.write(f'Skipping task {task.id}.')
                continue

            # task still running => continue
            if task.status == TaskStatus.RUNNING:
                new_remaining_tasks.append((dataset, task))
                continue

            progbar.update(1)
            tqdm.write(f'Task {task.id} stopped with status {task.status}')

            if task.status in [TaskStatus.PROBLEM, TaskStatus.FAILED]:
                tqdm.write('API response:')
                tqdm.write(str(task.response))
                continue

            # task finished successfully, save response to disk
            assert task.status == TaskStatus.COMPLETED
            if output_dir is not None:
                save_dir = os.path.join(output_dir, 'batchapi_response')
                if not os.path.exists(save_dir):
                    tqdm.write(f'Creating API output dir: {save_dir}')
                    os.makedirs(save_dir)
                with open(os.path.join(save_dir, f'{task.id}.json'), 'w') as f:
                    json.dump(task.response, f, indent=1)
            message = task.response['Status']['message']
            num_failed_shards = message['num_failed_shards']
            if num_failed_shards != 0:
                tqdm.write(f'Task {task.id} completed with {num_failed_shards} '
                           'failed shards.')

            detections_url = message['output_file_urls']['detections']
            if task.id not in detections_url:
                tqdm.write('Invalid detections URL in response. Skipping task.')
                continue

            detections = requests.get(detections_url).json()
            msg = cache_detections(
                detections=detections, dataset=dataset,
                detector_output_cache_dir=detector_output_cache_dir)
            tqdm.write(msg)

        remaining_tasks = new_remaining_tasks
        if len(remaining_tasks) == 0:
            break
        tqdm.write(f'Sleeping for {poll_interval} seconds...')
        time.sleep(poll_interval)

    progbar.close()


def download_and_crop(
        queried_images_json: Mapping[str, Mapping[str, Any]],
        detection_cache: Mapping[str, Mapping[str, Mapping[str, Any]]],
        detection_categories: Mapping[str, str],
        detector_version: str,
        cropped_images_dir: str,
        confidence_threshold: float,
        save_full_images: bool,
        square_crops: bool,
        check_crops_valid: bool,
        images_dir: Optional[str] = None,
        threads: int = 1,
        images_missing_detections: Optional[Iterable[str]] = None
        ) -> tuple[list[str], int, int]:
    """
    Saves crops to a file with the same name as the original image with an
    additional suffix appended, starting with 3 underscores:
    - if image has ground truth bboxes: "___cropXX.jpg", where "XX" indicates
        the bounding box index
    - if image has bboxes from MegaDetector: "___cropXX_mdvY.Y.jpg", where
        "Y.Y" indicates the MegaDetector version
    See module docstring for more info and examples.

    Note: this function is very similar to the "download_and_crop()" function in
        crop_detections.py. The main difference is that this function uses
        MegaDB to look up Azure Storage container information for images based
        on the dataset, whereas the crop_detections.py version has no concept
        of a "dataset" and "ground-truth" bounding boxes from MegaDB.

    Args:
        queried_images_json: dict, represents JSON output of json_validator.py,
            all images in queried_images_json are assumed to have either ground
            truth or cached detected bounding boxes unless
            images_missing_detections is given
        detection_cache: dict, dataset_name => {img_path => detection_dict}
        detector_version: str, detector version string, e.g., '4.1'
        cropped_images_dir: str, path to folder where cropped images are saved
        confidence_threshold: float, only crop bounding boxes above this value
        save_full_images: bool, whether to save downloaded images to images_dir,
            images_dir must be given and must exist if save_full_images=True
        square_crops: bool, whether to crop bounding boxes as squares
        check_crops_valid: bool, whether to load each crop to ensure the file is
            valid (i.e., not truncated)
        images_dir: optional str, path to folder where full images are saved
        threads: int, number of threads to use for downloading images
        images_missing_detections: optional list of str, image files to skip
            because they have no ground truth or cached detected bounding boxes

    Returns: list of str, images with bounding boxes that failed to download or
        crop properly
    """
    # error checking before we download and crop any images
    valid_img_paths = set(queried_images_json.keys())
    if images_missing_detections is not None:
        valid_img_paths -= set(images_missing_detections)
    for img_path in valid_img_paths:
        info_dict = queried_images_json[img_path]
        ds, img_file = img_path.split('/', maxsplit=1)
        assert ds == info_dict['dataset']

        if 'bbox' in info_dict:  # ground-truth bounding boxes
            pass
        elif img_file in detection_cache[ds]:  # detected bounding boxes
            bbox_dicts = detection_cache[ds][img_file]['detections']
            assert all('conf' in bbox_dict for bbox_dict in bbox_dicts)
            # convert from category ID to category name
            for d in bbox_dicts:
                d['category'] = detection_categories[d['category']]
        else:
            raise ValueError(f'{img_path} has no ground truth bounding boxes '
                             'and was not found in the detection cache. Please '
                             'include it in images_missing_detections.')

    # we need the datasets table for getting SAS keys
    datasets_table = megadb_utils.MegadbUtils().get_datasets_table()
    container_clients = {}  # dataset name => ContainerClient

    pool = futures.ThreadPoolExecutor(max_workers=threads)
    future_to_img_path = {}
    images_failed_download = []

    print(f'Getting bbox info for {len(valid_img_paths)} images...')
    for img_path in tqdm(sorted(valid_img_paths)):
        # we already did all error checking above, so we don't do any here
        info_dict = queried_images_json[img_path]
        ds, img_file = img_path.split('/', maxsplit=1)

        # get ContainerClient
        if ds not in container_clients:
            sas_token = datasets_table[ds]['container_sas_key']
            if sas_token[0] == '?':
                sas_token = sas_token[1:]
            url = sas_blob_utils.build_azure_storage_uri(
                account=datasets_table[ds]['storage_account'],
                container=datasets_table[ds]['container'],
                sas_token=sas_token)
            container_clients[ds] = ContainerClient.from_container_url(url)
        container_client = container_clients[ds]

        # get bounding boxes
        # we must include the dataset <ds> in <crop_path_template> because
        #    '{img_path}' actually gets populated with <img_file> in
        #    load_and_crop()
        is_ground_truth = ('bbox' in info_dict)
        if is_ground_truth:  # ground-truth bounding boxes
            bbox_dicts = info_dict['bbox']
            crop_path_template = os.path.join(
                cropped_images_dir, ds, '{img_path}___crop{n:>02d}.jpg')
        else:  # detected bounding boxes
            bbox_dicts = detection_cache[ds][img_file]['detections']
            crop_path_template = os.path.join(
                cropped_images_dir, ds,
                '{img_path}___crop{n:>02d}_' + f'mdv{detector_version}.jpg')

        ds_dir = None if images_dir is None else os.path.join(images_dir, ds)

        # get the image, either from disk or from Blob Storage
        future = pool.submit(
            load_and_crop, img_file, ds_dir, container_client, bbox_dicts,
            confidence_threshold, crop_path_template, save_full_images,
            square_crops, check_crops_valid)
        future_to_img_path[future] = img_path

    total = len(future_to_img_path)
    total_downloads = 0
    total_new_crops = 0
    print(f'Reading/downloading {total} images and cropping...')
    for future in tqdm(futures.as_completed(future_to_img_path), total=total):
        img_path = future_to_img_path[future]
        try:
            did_download, num_new_crops = future.result()
            total_downloads += did_download
            total_new_crops += num_new_crops
        except Exception as e:  # pylint: disable=broad-except
            exception_type = type(e).__name__
            tqdm.write(f'{img_path} - generated {exception_type}: {e}')
            images_failed_download.append(img_path)

    pool.shutdown()
    for container_client in container_clients.values():
        # inelegant way to close the container_clients
        with container_client:
            pass

    print(f'Downloaded {total_downloads} images.')
    print(f'Made {total_new_crops} new crops.')
    return images_failed_download, total_downloads, total_new_crops


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Detects and crops images.')
    parser.add_argument(
        'queried_images_json',
        help='path to JSON file mapping image paths and classification info')
    parser.add_argument(
        'output_dir',
        help='path to directory to save log file. If --run-detector, then '
             'task lists and status responses are also saved here.')
    parser.add_argument(
        '-c', '--detector-output-cache-dir', required=True,
        help='(required) path to directory where detector outputs are cached')
    parser.add_argument(
        '-v', '--detector-version', required=True,
        help='(required) detector version string, e.g., "4.1"')
    parser.add_argument(
        '-d', '--run-detector', action='store_true',
        help='Run the Batch Detection API. If not given, skips running the '
             'detector (and only use ground truth and cached bounding boxes).')
    parser.add_argument(
        '-r', '--resume-file',
        help='path to save JSON file with list of info dicts on running tasks, '
             'or to resume from running tasks. Only used if --run-detector is '
             'set. Each dict has keys '
             '["dataset", "task_id", "task_name", "local_images_list_path", '
             '"remote_images_list_url"]')
    parser.add_argument(
        '-p', '--cropped-images-dir',
        help='path to local directory for saving crops of bounding boxes. No '
             'images are downloaded or cropped if this argument is not given.')
    parser.add_argument(
        '--save-full-images', action='store_true',
        help='if downloading an image, save the full image to --images-dir, '
             'only used if <cropped_images_dir> is not None')
    parser.add_argument(
        '--square-crops', action='store_true',
        help='crop bounding boxes as squares, '
             'only used if <cropped_images_dir> is not None')
    parser.add_argument(
        '--check-crops-valid', action='store_true',
        help='load each crop to ensure file is valid (i.e., not truncated), '
             'only used if <cropped_images_dir> is not None')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.0,
        help='confidence threshold above which to crop bounding boxes, '
             'only used if <cropped_images_dir> is not None')
    parser.add_argument(
        '-i', '--images-dir',
        help='path to local directory where images are saved, '
             'only used if <cropped_images_dir> is not None')
    parser.add_argument(
        '-n', '--threads', type=int, default=1,
        help='number of threads to use for downloading images, '
             'only used if <cropped_images_dir> is not None')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(queried_images_json_path=args.queried_images_json,
         output_dir=args.output_dir,
         detector_version=args.detector_version,
         detector_output_cache_base_dir=args.detector_output_cache_dir,
         run_detector=args.run_detector,
         resume_file_path=args.resume_file,
         cropped_images_dir=args.cropped_images_dir,
         save_full_images=args.save_full_images,
         square_crops=args.square_crops,
         check_crops_valid=args.check_crops_valid,
         confidence_threshold=args.threshold,
         images_dir=args.images_dir,
         threads=args.threads)

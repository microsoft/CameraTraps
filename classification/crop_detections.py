r"""
Given a detections JSON file from MegaDetector, crops the bounding boxes above
a certain confidence threshold.

This script takes as input a detections JSON file, usually the output of
detection/run_tf_detector_batch.py or the output of the Batch API in the
"Batch processing API output format".

See https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing.

The script can crop images that are either available locally or that need to be
downloaded from an Azure Blob Storage container.

We assume that no image contains over 100 bounding boxes, and we always save
crops as RGB .jpg files for consistency. For each image, each bounding box is
cropped and saved to a file with a suffix "___cropXX_mdvY.Y.jpg" added to the
filename as the original image. "XX" ranges from "00" to "99" and "Y.Y"
ndicates the MegaDetector version. Based on the given confidence threshold, we
may skip saving certain bounding box crops, but we still increment the bounding
box number for skipped boxes.

Example cropped image path (with MegaDetector bbox)
    "path/to/image.jpg___crop00_mdv4.1.jpg"

By default, the images are cropped exactly per the given bounding box
coordinates. However, if square crops are desired, pass the --square-crops
flag. This will always generate a square crop whose size is the larger of the
bounding box width or height. In the case that the square crop boundaries exceed
the original image size, the crop is padded with 0s.

This script outputs a log file to
    <output_dir>/crop_detections_log_{timestamp}.json
which contains images that failed to download and crop properly.

Example command:

python crop_detections.py \
    detections.json \
    /path/to/crops \
    --images-dir /path/to/images \
    --container-url "https://account.blob.core.windows.net/container?sastoken" \
    --detector-version "4.1" \
    --threshold 0.8 \
    --save-full-images --square-crops \
    --threads 50 \
    --logdir "."
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from concurrent import futures
from datetime import datetime
import io
import json
import os
from typing import Any, BinaryIO, Optional

from azure.storage.blob import ContainerClient
from PIL import Image, ImageOps
from tqdm import tqdm


def main(detections_json_path: str,
         cropped_images_dir: str,
         images_dir: Optional[str],
         container_url: Optional[str],
         detector_version: Optional[str],
         save_full_images: bool,
         square_crops: bool,
         check_crops_valid: bool,
         confidence_threshold: float,
         threads: int,
         logdir: str) -> None:
    """
    Args:
        detections_json_path: str, path to detections JSON file
        cropped_images_dir: str, path to local directory for saving crops of
            bounding boxes
        images_dir: optional str, path to local directory where images are saved
        container_url: optional str, URL (with SAS token, if necessary) of Azure
            Blob Storage container to download images from, if images are not
            all already locally available in <images_dir>
        detector_version: str, detector version string, e.g., '4.1',
            see {batch_detection_api_url}/supported_model_versions
        save_full_images: bool, whether to save downloaded images to images_dir,
            images_dir must be given if save_full_images=True
        square_crops: bool, whether to crop bounding boxes as squares
        check_crops_valid: bool, whether to load each crop to ensure the file is
            valid (i.e., not truncated)
        confidence_threshold: float, only crop bounding boxes above this value
        threads: int, number of threads to use for downloading images
        logdir: str, path to directory to save log file
    """
    # error checking
    assert 0 <= confidence_threshold <= 1, \
            'Invalid confidence threshold {}'.format(confidence_threshold)
    if save_full_images:
        assert images_dir is not None, \
            'save_full_images specified but no images_dir provided'
        if not os.path.exists(images_dir):
            os.makedirs(images_dir, exist_ok=True)
            print(f'Created images_dir at {images_dir}')

    # load detections JSON
    with open(detections_json_path, 'r') as f:
        js = json.load(f)
    detections = {img['file']: img for img in js['images']}
    detection_categories = js['detection_categories']

    # get detector version
    if 'info' in js and 'detector' in js['info']:
        api_det_version = js['info']['detector'] # .rsplit('v', maxsplit=1)[1]
        if detector_version is not None:
            assert api_det_version == detector_version,\
            '.json file specifies a detector version of {}, but the caller has specified {}'.format(
            api_det_version,detector_version)
        else:
            detector_version = api_det_version
    if detector_version is None:
        detector_version = 'unknown'

    # convert from category ID to category name
    images_missing_detections = []

    # copy keys to modify dict in-place
    for img_path in list(detections.keys()):
        info_dict = detections[img_path]
        if 'detections' not in info_dict or info_dict['detections'] is None:
            del detections[img_path]
            images_missing_detections.append(img_path)
            continue
        for d in info_dict['detections']:
            if d['category'] not in detection_categories:
                print('Warning: ignoring detection with category {} for image {}'.format(
                    d['category'],img_path))                
                # This will be removed later when we filter for animals
                d['category'] = 'unsupported'
            else:
                d['category'] = detection_categories[d['category']]

    images_failed_dload_crop, num_downloads, num_crops = download_and_crop(
        detections=detections,
        cropped_images_dir=cropped_images_dir,
        images_dir=images_dir,
        container_url=container_url,
        detector_version=detector_version,
        confidence_threshold=confidence_threshold,
        save_full_images=save_full_images,
        square_crops=square_crops,
        check_crops_valid=check_crops_valid,
        threads=threads)
    print(f'{len(images_failed_dload_crop)} images failed to download or crop.')

    # save log of bad images
    log = {
        'images_missing_detections': images_missing_detections,
        'images_failed_download_or_crop': images_failed_dload_crop,
        'num_new_downloads': num_downloads,
        'num_new_crops': num_crops
    }
    os.makedirs(logdir, exist_ok=True)
    date = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., '20200722_110816'
    log_path = os.path.join(logdir, f'crop_detections_log_{date}.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=1)


def download_and_crop(
        detections: Mapping[str, Mapping[str, Any]],
        cropped_images_dir: str,
        images_dir: Optional[str],
        container_url: Optional[str],
        detector_version: str,
        confidence_threshold: float,
        save_full_images: bool,
        square_crops: bool,
        check_crops_valid: bool,
        threads: int = 1
        ) -> tuple[list[str], int, int]:
    """
    Saves crops to a file with the same name as the original image with an
    additional suffix appended, starting with 3 underscores:
    - if image has ground truth bboxes: "___cropXX.jpg", where "XX" indicates
        the bounding box index
    - if image has bboxes from MegaDetector: "___cropXX_mdvY.Y.jpg", where
        "Y.Y" indicates the MegaDetector version
    See module docstring for more info and examples.

    Args:
        detections: dict, maps image paths to info dict
            {
                "detections": [{
                    "category": "animal",  # must be name, not "1" or "2"
                    "conf": 0.926,
                    "bbox": [0.0, 0.2762, 0.1539, 0.2825],
                }],
                "is_ground_truth": True  # whether bboxes are ground truth
            }
        cropped_images_dir: str, path to folder where cropped images are saved
        images_dir: optional str, path to folder where full images are saved
        container_url: optional str, URL (with SAS token, if necessary) of Azure
            Blob Storage container to download images from, if images are not
            all already locally available in <images_dir>
        detector_version: str, detector version string, e.g., '4.1'
        confidence_threshold: float, only crop bounding boxes above this value
        save_full_images: bool, whether to save downloaded images to images_dir,
            images_dir must be given and must exist if save_full_images=True
        square_crops: bool, whether to crop bounding boxes as squares
        check_crops_valid: bool, whether to load each crop to ensure the file is
            valid (i.e., not truncated)
        threads: int, number of threads to use for downloading images

    Returns:
        images_failed_download: list of str, images with bounding boxes that
            failed to download or crop properly
        total_downloads: int, number of images downloaded
        total_new_crops: int, number of new crops saved to cropped_images_dir
    """
    # True for ground truth, False for MegaDetector
    # always save as .jpg for consistency
    crop_path_template = {
        True: os.path.join(cropped_images_dir, '{img_path}___crop{n:>02d}.jpg'),
        False: os.path.join(
            cropped_images_dir,
            '{img_path}___crop{n:>02d}_' + f'{detector_version}.jpg')
    }

    pool = futures.ThreadPoolExecutor(max_workers=threads)
    future_to_img_path = {}
    images_failed_download = []

    container_client = None
    if container_url is not None:
        container_client = ContainerClient.from_container_url(container_url)

    print(f'Getting bbox info for {len(detections)} images...')
    for img_path in tqdm(sorted(detections.keys())):
        # we already did all error checking above, so we don't do any here
        info_dict = detections[img_path]
        bbox_dicts = info_dict['detections']
        is_ground_truth = info_dict.get('is_ground_truth', False)

        # get the image, either from disk or from Blob Storage
        future = pool.submit(
            load_and_crop, img_path, images_dir, container_client, bbox_dicts,
            confidence_threshold, crop_path_template[is_ground_truth],
            save_full_images, square_crops, check_crops_valid)
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
    if container_client is not None:
        # inelegant way to close the container_client
        with container_client:
            pass

    print(f'Downloaded {total_downloads} images.')
    print(f'Made {total_new_crops} new crops.')
    return images_failed_download, total_downloads, total_new_crops


def load_local_image(img_path: str |  BinaryIO) -> Optional[Image.Image]:
    """Attempts to load an image from a local path."""
    try:
        with Image.open(img_path) as img:
            img.load()
        return img
    except OSError as e:  # PIL.UnidentifiedImageError is a subclass of OSError
        exception_type = type(e).__name__
        tqdm.write(f'Unable to load {img_path}. {exception_type}: {e}.')
    return None


def load_and_crop(img_path: str,
                  images_dir: Optional[str],
                  container_client: Optional[ContainerClient],
                  bbox_dicts: Iterable[Mapping[str, Any]],
                  confidence_threshold: float,
                  crop_path_template: str,
                  save_full_image: bool,
                  square_crops: bool,
                  check_crops_valid: bool) -> tuple[bool, int]:
    """Given an image and a list of bounding boxes, checks if the crops already
    exist. If not, loads the image locally or Azure Blob Storage, then crops it.

    local image path: <images_dir>/<img_path>
    Azure storage: <img_path> as the blob name inside the container

    An image is only downloaded from Azure Blob Storage if it does not already
    exist locally and if it has at least 1 bounding box with confidence greater
    than the confidence threshold.

    Args:
        img_path: str, image path
        images_dir: optional str, path to local directory of images, and where
            full images are saved if save_full_images=True
        container_client: optional ContainerClient, this function does not
            use container_client in any context manager
        bbox_dicts: list of dicts, each dict contains info on a bounding box
        confidence_threshold: float, only crop bounding boxes above this value
        crop_path_template: str, contains placeholders {img_path} and {n}
        save_full_images: bool, whether to save downloaded images to images_dir,
            images_dir must be given and must exist if save_full_images=True
        square_crops: bool, whether to crop bounding boxes as squares
        check_crops_valid: bool, whether to load each crop to ensure the file is
            valid (i.e., not truncated)

    Returns:
        did_download: bool, whether image was downloaded from Azure Blob Storage
        num_new_crops: int, number of new crops successfully saved
    """
    did_download = False
    num_new_crops = 0

    # crop_path => normalized bbox coordinates [xmin, ymin, width, height]
    bboxes_tocrop: dict[str, list[float]] = {}
    for i, bbox_dict in enumerate(bbox_dicts):
        # only ground-truth bboxes do not have a "confidence" value
        if 'conf' in bbox_dict and bbox_dict['conf'] < confidence_threshold:
            continue
        if bbox_dict['category'] != 'animal':
            continue
        crop_path = crop_path_template.format(img_path=img_path, n=i)
        if not os.path.exists(crop_path) or (
                check_crops_valid and load_local_image(crop_path) is None):
            bboxes_tocrop[crop_path] = bbox_dict['bbox']
    if len(bboxes_tocrop) == 0:
        return did_download, num_new_crops

    img = None

    # try loading image from local directory
    if images_dir is not None:
        full_img_path = os.path.join(images_dir, img_path)
        debug_path = full_img_path
        if os.path.exists(full_img_path):
            img = load_local_image(full_img_path)

    # try to download image from Blob Storage
    if img is None and container_client is not None:
        debug_path = img_path
        with io.BytesIO() as stream:
            container_client.download_blob(img_path).readinto(stream)
            stream.seek(0)

            if save_full_image:
                os.makedirs(os.path.dirname(full_img_path), exist_ok=True)
                with open(full_img_path, 'wb') as f:
                    f.write(stream.read())
                stream.seek(0)

            img = load_local_image(stream)
        did_download = True

    assert img is not None, 'image "{}" failed to load or download properly'.format(
        debug_path)
    
    if img.mode != 'RGB':
        img = img.convert(mode='RGB')  # always save as RGB for consistency

    # crop the image
    for crop_path, bbox in bboxes_tocrop.items():
        num_new_crops += save_crop(
            img, bbox_norm=bbox, square_crop=square_crops, save=crop_path)
    return did_download, num_new_crops


def save_crop(img: Image.Image, bbox_norm: Sequence[float], square_crop: bool,
              save: str) -> bool:
    """Crops an image and saves the crop to file.

    Args:
        img: PIL.Image.Image object, already loaded
        bbox_norm: list or tuple of float, [xmin, ymin, width, height] all in
            normalized coordinates
        square_crop: bool, whether to crop bounding boxes as a square
        save: str, path to save cropped image

    Returns: bool, True if a crop was saved, False otherwise
    """
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)

    if square_crop:
        # expand box width or height to be square, but limit to img size
        box_size = max(box_w, box_h)
        xmin = max(0, min(
            xmin - int((box_size - box_w) / 2),
            img_w - box_w))
        ymin = max(0, min(
            ymin - int((box_size - box_h) / 2),
            img_h - box_h))
        box_w = min(img_w, box_size)
        box_h = min(img_h, box_size)

    if box_w == 0 or box_h == 0:
        tqdm.write(f'Skipping size-0 crop (w={box_w}, h={box_h}) at {save}')
        return False

    # Image.crop() takes box=[left, upper, right, lower]
    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])

    if square_crop and (box_w != box_h):
        # pad to square using 0s
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

    os.makedirs(os.path.dirname(save), exist_ok=True)
    crop.save(save)
    return True


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Crop detections from MegaDetector.')
    parser.add_argument(
        'detections_json',
        help='path to detections JSON file')
    parser.add_argument(
        'cropped_images_dir',
        help='path to local directory for saving crops of bounding boxes')
    parser.add_argument(
        '-i', '--images-dir',
        help='path to directory where full images are already available, '
             'or where images will be written if --save-full-images is set')
    parser.add_argument(
        '-c', '--container-url',
        help='URL (including SAS token, if necessary) of Azure Blob Storage '
             'container to download images from, if images are not all already '
             'locally available in <images_dir>')
    parser.add_argument(
        '-v', '--detector-version',
        help='detector version string, e.g., "4.1", used if detector version '
             'cannot be inferred from detections JSON')
    parser.add_argument(
        '--save-full-images', action='store_true',
        help='forces downloading of full images to --images-dir')
    parser.add_argument(
        '--square-crops', action='store_true',
        help='crop bounding boxes as squares')
    parser.add_argument(
        '--check-crops-valid', action='store_true',
        help='load each crop to ensure file is valid (i.e., not truncated)')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.0,
        help='confidence threshold above which to crop bounding boxes')
    parser.add_argument(
        '-n', '--threads', type=int, default=1,
        help='number of threads to use for downloading and cropping images')
    parser.add_argument(
        '--logdir', default='.',
        help='path to directory to save log file')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(detections_json_path=args.detections_json,
         cropped_images_dir=args.cropped_images_dir,
         images_dir=args.images_dir,
         container_url=args.container_url,
         detector_version=args.detector_version,
         save_full_images=args.save_full_images,
         square_crops=args.square_crops,
         check_crops_valid=args.check_crops_valid,
         confidence_threshold=args.threshold,
         threads=args.threads,
         logdir=args.logdir)

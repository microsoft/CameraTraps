"""
Render images with bounding boxes annotated on them to a folder, based on a
detector output result file (json). The original images can be local or in
Azure Blob Storage.
"""

#%% Imports

import argparse
import json
import os
import random
import sys
from typing import Any, List, Optional

from tqdm import tqdm

from data_management.annotations.annotation_constants import (
    detector_bbox_category_id_to_name)  # here id is int
from visualization import visualization_utils as vis_utils


#%% Constants

# convert category ID from int to str
DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}


#%% Main function

def visualize_detector_output(detector_output_path: str,
                              out_dir: str,
                              images_dir: str,
                              is_azure: bool = False,
                              confidence: float = 0.8,
                              sample: int = -1,
                              output_image_width: int = 700,
                              random_seed: Optional[int] = None) -> List[str]:
    """Draw bounding boxes on images given the output of the detector.

    Args:
        detector_output_path: str, path to detector output json file
        out_dir: str, path to directory for saving annotated images
        images_dir: str, path to local images dir, or a SAS URL to an Azure Blob
            Storage container
        is_azure: bool, whether images_dir points to an Azure URL
        confidence: float, threshold above which annotations will be rendered
        sample: int, maximum number of images to annotate, -1 for all
        output_image_width: int, width in pixels to resize images for display,
            set to -1 to use original image width
        random_seed: int, for deterministic image sampling when sample != -1

    Returns: list of str, paths to annotated images
    """
    # arguments error checking
    assert confidence > 0 and confidence < 1, (
        f'Confidence threshold {confidence} is invalid, must be in (0, 1).')

    assert os.path.exists(detector_output_path), (
        f'Detector output file does not exist at {detector_output_path}.')

    if is_azure:
        # we don't import sas_blob_utils at the top of this file in order to
        # accommodate the MegaDetector Colab notebook which does not have
        # the azure-storage-blob package installed
        import sas_blob_utils
    else:
        assert os.path.isdir(images_dir)

    os.makedirs(out_dir, exist_ok=True)

    #%% Load detector output

    with open(detector_output_path) as f:
        detector_output = json.load(f)
    assert 'images' in detector_output, (
        'Detector output file should be a json with an "images" field.')
    images = detector_output['images']

    detector_label_map = DEFAULT_DETECTOR_LABEL_MAP
    if 'detection_categories' in detector_output:
        print('detection_categories provided')
        detector_label_map = detector_output['detection_categories']

    num_images = len(images)
    print(f'Detector output file contains {num_images} entries.')

    if sample > 0:
        assert num_images >= sample, (
            f'Sample size {sample} greater than number of entries '
            f'({num_images}) in detector result.')

        if random_seed is not None:
            images = sorted(images, key=lambda x: x['file'])
            random.seed(random_seed)

        random.shuffle(images)
        images = sorted(images[:sample], key=lambda x: x['file'])
        print(f'Sampled {len(images)} entries from the detector output file.')


    #%% Load images, annotate them and save

    print('Starting to annotate the images...')
    num_saved = 0
    annotated_img_paths = []
    image_obj: Any  # str for local images, BytesIO for Azure images

    for entry in tqdm(images):
        image_id = entry['file']

        if 'failure' in entry:
            print(f'Skipping {image_id}, failure: "{entry["failure"]}"')
            continue

        # max_conf = entry['max_detection_conf']

        if is_azure:
            blob_uri = sas_blob_utils.build_blob_uri(
                container_uri=images_dir, blob_name=image_id)
            if not sas_blob_utils.check_blob_exists(blob_uri):
                container = sas_blob_utils.get_container_from_uri(images_dir)
                print(f'Image {image_id} not found in blob container '
                      f'{container}; skipped.')
                continue
            image_obj, _ = sas_blob_utils.download_blob_to_stream(blob_uri)
        else:
            image_obj = os.path.join(images_dir, image_id)
            if not os.path.exists(image_obj):
                print(f'Image {image_id} not found in images_dir; skipped.')
                continue

        # resize is for displaying them more quickly
        image = vis_utils.resize_image(
            vis_utils.open_image(image_obj), output_image_width)

        vis_utils.render_detection_bounding_boxes(
            entry['detections'], image, label_map=detector_label_map,
            confidence_threshold=confidence)

        for char in ['/', '\\', ':']:
            image_id = image_id.replace(char, '~')
        annotated_img_path = os.path.join(out_dir, f'anno_{image_id}')
        annotated_img_paths.append(annotated_img_path)
        image.save(annotated_img_path)
        num_saved += 1

        if is_azure:
            image_obj.close()  # BytesIO object

    print(f'Rendered detection results on {num_saved} images, '
          f'saved to {out_dir}.')

    return annotated_img_paths


#%% Command-line driver

def main() -> None:
    """Main function."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Annotate the bounding boxes predicted by a detector above '
                    'some confidence threshold, and save the annotated images.')
    parser.add_argument(
        'detector_output_path', type=str,
        help='Path to json output file of the detector')
    parser.add_argument(
        'out_dir', type=str,
        help='Path to directory where the annotated images will be saved. '
             'The directory will be created if it does not exist.')
    parser.add_argument(
        '-c', '--confidence', type=float, default=0.8,
        help='Value between 0 and 1, indicating the confidence threshold '
             'above which to visualize bounding boxes')
    parser.add_argument(
        '-i', '--images_dir', type=str, default=None,
        help='Path to a local directory or a SAS URL (in double quotes) to an '
             'Azure blob storage container where images are stored. This '
             'serves as the root directory for image paths in '
             'detector_output_path. If an Azure URL, pass the -a/--is-azure '
             'flag. You can use Azure Storage Explorer to obtain a SAS URL.')
    parser.add_argument(
        '-a', '--is-azure', action='store_true',
        help='Flag that indidcates images_dir is an Azure blob storage '
             'container URL.')
    parser.add_argument(
        '-n', '--sample', type=int, default=-1,
        help='Number of images to be annotated and rendered. Set to -1 '
             '(default) to annotate all images in the detector output file. '
             'There may be fewer images if some are not found in images_dir.')
    parser.add_argument(
        '-w', '--output_image_width', type=int, default=700,
        help='Integer, desired width in pixels of the output annotated images. '
             'Use -1 to not resize. Default: 700.')
    parser.add_argument(
        '-r', '--random_seed', type=int, default=None,
        help='Integer, for deterministic order of image sampling')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    visualize_detector_output(
        detector_output_path=args.detector_output_path,
        out_dir=args.out_dir,
        confidence=args.confidence,
        images_dir=args.images_dir,
        is_azure=args.is_azure,
        sample=args.sample,
        output_image_width=args.output_image_width,
        random_seed=args.random_seed)


if __name__ == '__main__':
    main()

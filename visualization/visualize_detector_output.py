#####
#
# visualize_detector_output.py
#
# Render images with bounding boxes annotated on them to a folder, based on a detector output result
# file (CSV). The original images can be local or in Azure Blob Storage.
#
#####

#%% Imports

import argparse
import io
import json
import os
from urllib import parse

import pandas as pd
from azure.storage.blob import BlockBlobService
from tqdm import tqdm

import visualization_utils as vis_utils


#%% Settings and user-supplied arguments

viz_size = (675, 450)  # width by height, in pixels


parser = argparse.ArgumentParser(description=('Annotate the bounding boxes predicted by a detector '
                                              'above some confidence threshold, and save the annotated images.'))

parser.add_argument('detector_output_path', type=str,
                    help='path to the csv output file of the detector',
                    default='RequestID_all_output.csv')

parser.add_argument('out_dir', type=str,
                    help='path to a directory where the annotated images will be saved. Created if does not already exist')

parser.add_argument('-c', '--confidence', type=float,
                    help=('a value between 0 and 1, indicating the confidence threshold above which to visualize '
                          'bounding boxes'),
                    default=0.8)

parser.add_argument('-i', '--images_dir', type=str,
                    help=('path to a local directory where the images are stored. This needs to be the root '
                          'directory for image paths used in the detector_output_path'),
                    default=None)

parser.add_argument('-s', '--sas_url', type=str,
                    help=('SAS URL with list and read permissions to an Azure blob storage container where the '
                          'images are stored. You can use Azure Storage Explorer to obtain a SAS URL'),
                    default=None)

parser.add_argument('-n', '--sample', type=int,
                    help=('an integer specifying how many images should be annotated and rendered. Default (-1) is all '
                          'images that have detector result. There may result in fewer images if some are not '
                          'found in images_dir'),
                    default=-1)

args = parser.parse_args()

assert args.confidence < 1.0 and args.confidence > 0.0, \
    'The confidence threshold {} supplied is not valid; choose a threshold between 0 and 1.'.format(args.confidence)

assert os.path.exists(args.detector_output_path), \
    'Detector output file does not exist at {}'.format(args.detector_output_path)

assert args.images_dir or args.sas_url, \
    ('One of images_dir (original images in a local directory) or sas_url (images in the cloud) is required.')

if args.images_dir and args.sas_url:
    print('Both local images_dir and remote sas_url are supplied. Using local images as originals.')

images_local = True if args.images_dir is not None else False

os.makedirs(args.out_dir, exist_ok=True)


#%% Helper functions

def get_sas_key_from_uri(sas_uri):
    """Get the query part of the SAS token that contains permissions, access times and
    signature.

    Args:
        sas_uri: Azure blob storage SAS token

    Returns: Query part of the SAS token.
    """
    url_parts = parse.urlsplit(sas_uri)
    return url_parts.query

def get_service_from_uri(sas_uri):
    return BlockBlobService(
        account_name=get_account_from_uri(sas_uri),
        sas_token=get_sas_key_from_uri(sas_uri))

def get_account_from_uri(sas_uri):
    url_parts = parse.urlsplit(sas_uri)
    loc = url_parts.netloc
    return loc.split('.')[0]

def get_container_from_uri(sas_uri):
    url_parts = parse.urlsplit(sas_uri)

    raw_path = url_parts.path[1:]
    container = raw_path.split('/')[0]

    return container


#%% Load detector output

df = pd.read_csv(args.detector_output_path)

assert len(df.shape) == 2 and df.shape[1] == 3, 'Detector output file should be a csv with 3 columns.'

num_rows = df.shape[0]
print('Detector output file contains {} entries.'.format(num_rows))

if args.sample > 0:
    assert num_rows >= args.sample, \
        'Sample size {} specified greater than number of entries in detector result.'.format(args.sample)

    df = df.sample(args.sample)
    print('Sampled {} entries from the detector output file.'.format(df.shape[0]))


#%% Load images, annotate them and save

if not images_local:
    blob_service = get_service_from_uri(args.sas_url)
    container_name = get_container_from_uri(args.sas_url)

print('Starting to annotate the images...')
num_saved = 0
for i_row, row in tqdm(df.iterrows()):
    image_id = row[0]
    max_conf = float(row[1])
    boxes_and_scores = json.loads(row[2])

    if images_local:
        image_obj = os.path.join(args.images_dir, image_id)
        if not os.path.exists(image_obj):
            print('Image {} is not found at images_dir; skipped.'.format(image_id))
            continue
    else:
        if not blob_service.exists(container_name, blob_name=image_id):
            print('Image {} is not found in the blob container {}; skipped.'.format(image_id, container_name))
            continue

        image_obj = io.BytesIO()
        _ = blob_service.get_blob_to_stream(container_name, image_id, image_obj)

    image = vis_utils.open_image(image_obj).resize(viz_size)  # resize is to display them more quickly
    vis_utils.render_detection_bounding_boxes(boxes_and_scores, image, confidence_threshold=args.confidence)

    annotated_img_name = image_id.replace('/', '~')
    annotated_img_path = os.path.join(args.out_dir, annotated_img_name)
    image.save(annotated_img_path)
    num_saved += 1

print('Rendered detection results on {} images, saved to {}.'.format(num_saved, args.out_dir))

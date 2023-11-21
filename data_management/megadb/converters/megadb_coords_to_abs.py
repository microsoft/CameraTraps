"""
A prerequisite step to using megadb_to_cct.py

Takes in a list of sequence objects from MegaDB, and download the images to read their dimensions
if that image has a bbox label. For those images, add `id`, `width`, and `height` attributes.
"""

import argparse
import io
import json
import os
import time
import uuid
from functools import partial
from multiprocessing.pool import ThreadPool as workerpool

import humanfriendly
from PIL import Image
from tqdm import tqdm

from ct_utils import write_json
from data_management.megadb.megadb_utils import MegadbUtils


def round_to_int(f):
    return int(round(f))

def _get_image_dims(storage_client, path_prefix, image_obj):
    file_name = image_obj['file']
    file_path_on_blob = os.path.join(path_prefix, file_name)

    try:
        stream = io.BytesIO()
        storage_client.download_blob(file_path_on_blob).readinto(stream)
        image = Image.open(stream)
        image_height = image.height
        image_width = image.width
        stream.close()

        image_obj['width'] = image_width
        image_obj['height'] = image_height

        for b in image_obj['bbox']:
            coords = b['bbox']
            x = max(round_to_int(coords[0] * image_width), 0)
            y = max(round_to_int(coords[1] * image_height), 0)
            box_w = min(round_to_int(coords[2] * image_width), image_width)
            box_h = min(round_to_int(coords[3] * image_height), image_height)

            b['bbox'] = [x, y, box_w, box_h]

        return image_obj

    except Exception as e:
        print('Exception getting width and height for {}. Exception: {}'.format(file_name, e))
        return image_obj


def get_image_dims(mega_db_seqs, dataset_name, datasets_table, n_cores):

    images_to_get_dims_for = []

    for seq in tqdm(mega_db_seqs):
        assert 'seq_id' in seq and 'images' in seq
        for i in seq['images']:
            assert 'file' in i

        for im in seq['images']:
            if 'bbox' in im and len(im['bbox']) > 1:
                if 'id' not in im:
                    im['id'] = str(uuid.uuid1())
                images_to_get_dims_for.append(im)

    print('Getting the dimensions for {} images'.format(len(images_to_get_dims_for)))

    storage_client = MegadbUtils.get_storage_client(datasets_table, dataset_name)
    path_prefix = datasets_table[dataset_name]['path_prefix']

    if n_cores:
        print('Using threads to download images')
        pool = workerpool(n_cores)
        updated_im_objects = pool.map(partial(_get_image_dims, storage_client, path_prefix),
                                       images_to_get_dims_for)
        print('pool.map has returned')
    else:
        print('Downloading images sequentially')
        updated_im_objects = []
        for image_obj in tqdm(images_to_get_dims_for):
            updated_im_objects.append(get_image_dims(storage_client, path_prefix, image_obj))
    print('Successfully updated {} images.'.format(len(updated_im_objects)))
    updated_im_objects = {i['id']:i for i in updated_im_objects}

    # update the sequences
    print('Updating the sequence objects...')
    for seq in tqdm(mega_db_seqs):
        updated_images = []
        for im in seq['images']:
            if 'bbox' in im and im['id'] in updated_im_objects:
                updated_images.append(updated_im_objects[im['id']])
            else:
                updated_images.append(im)
        seq['images'] = updated_images

    return mega_db_seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_name',
        help='The name of the dataset; only entries from this dataset will be used')
    parser.add_argument(
        'mega_db_seqs',
        help='A json containing a list of sequence objects')
    parser.add_argument(
        'out_file',
        help='Path to store the resulting json to input to megadb_to_cct.py')
    parser.add_argument(
        '--ncores',
        type=int,
        default=None,
        help='Number of cores to use when downloading images to read their dimensions')
    args = parser.parse_args()

    assert len(args.dataset_name) > 0, 'dataset_name cannot be an empty string'
    assert os.path.exists(args.mega_db_seqs), 'File at mega_db path does not exist'
    assert args.out_file.endswith('.json'), 'out_cct_db path needs to end in .json'
    assert args.out_file != args.mega_db_seqs
    assert 'COSMOS_ENDPOINT' in os.environ and 'COSMOS_KEY' in os.environ

    print('Loading entries...')
    with open(args.mega_db_seqs) as f:
        mega_db_entries = json.load(f)
    print('Number of entries in the mega_db: {}'.format(len(mega_db_entries)))

    megadb_utils = MegadbUtils()
    datasets_table = megadb_utils.get_datasets_table()

    start_time = time.time()

    updated_seqs = get_image_dims(mega_db_entries, args.dataset_name, datasets_table, args.ncores)
    write_json(args.out_file, updated_seqs)

    elapsed = time.time() - start_time
    print('Time elapsed: {}'.format(humanfriendly.format_timespan(elapsed)))


if __name__ == '__main__':
    main()

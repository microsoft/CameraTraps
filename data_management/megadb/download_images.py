"""
Download images multi-threaded.

Expect the input JSON to be a list of dicts, where the dicts have fields
- download_id (usually dataset_name+uuid, uuid doesn't have to be from the database)
- file (from images)
- dataset and possibly location (from sequences)
- new_entry if setting the --only_new_images flag

Example entry:
 {
  "bbox": [
   {
    "category": "animal",
    "bbox": [ 0.3876, 0.3667, 0.1046, 0.22 ]
   }
  ],
  "file": "animals/0467/0962.jpg",
  "dataset": "wcs",
  "location": "3282",
  "download_id": "wcs+8a066ce6-3f14-11ea-aae7-9801a7a664ab",
  "new_entry": true
 }

The environment variables COSMOS_ENDPOINT and COSMOS_KEY need to be set, to access the Datasets table.

The downloading threads may get stuck (request timeouts) and stop downloading. Re-start a couple of times until all are downloaded.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing.pool import ThreadPool
from urllib.parse import quote

import humanfriendly
import requests

from data_management.megadb.megadb_utils import MegadbUtils


def download_file(url, local_path):
    try:
        r = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(r.content)
    except Exception as e:
        print('download_file exception: {}'.format(e))
        sys.exit(1)

def construct_url(item, datasets_table):
    try:
        encoded_file_name = quote(item['file'])
        dataset_name = item['dataset']

        storage_account = datasets_table[dataset_name]['storage_account']
        container = datasets_table[dataset_name]['container']
        container_sas_key = datasets_table[dataset_name]['container_sas_key']
        path_prefix = datasets_table[dataset_name]['path_prefix']

        path_to_blob = path_prefix + '/' + encoded_file_name if len(path_prefix) > 0 else encoded_file_name

        # currently only supports one endpoint
        url = 'https://{}.blob.core.windows.net/{}/{}{}'.format(storage_account, container, path_to_blob, container_sas_key)
        return url
    except Exception as e:
        print('construct_url exception: {}'.format(e))


def main():
    parser = argparse.ArgumentParser(
        description='Script to download image files'
    )
    parser.add_argument(
        'file_list'
    )
    parser.add_argument(
        'store_dir',
        help='Path to a directory to store the downloaded files'
    )
    parser.add_argument(
        '--single_thread',
        action='store_true'
    )
    parser.add_argument(
        '--only_new_images',
        action='store_true'
    )

    args = parser.parse_args()

    os.makedirs(args.store_dir, exist_ok=True)

    megadb_utils = MegadbUtils()
    datasets_table = megadb_utils.get_datasets_table()
    print('Obtained the datasets table. Loading the file_list now...')

    with open(args.file_list) as f:
        file_list = json.load(f)

    existing = os.listdir(args.store_dir)
    existing = set([i.split('.jpg')[0] for i in existing])


    file_list_to_download = [i for i in file_list if i['download_id'] not in existing]

    if args.only_new_images:
        print('Only going to download new images.')
        file_list_to_download = [i for i in file_list_to_download if 'new_entry' in i]

    # if need to re-download a dataset's images in case of corruption
    # file_list_to_download = [i for i in file_list_to_download if i['dataset'] == 'rspb_gola']

    print('file_list has {} items, still need to download {} items'.format(
        len(file_list),
        len(file_list_to_download)
    ))

    urls = [construct_url(item, datasets_table) for item in file_list_to_download]
    local_paths = [os.path.join(args.store_dir, '{}.jpg'.format(item['download_id'])) for item in file_list_to_download]

    origin_and_dest = zip(urls, local_paths)

    start_time = time.time()
    if args.single_thread:
        print('Starting to download, single threaded...')
        for url, local_path in origin_and_dest:
            download_file(url, local_path)
    else:
        print('Starting to download, using ThreadPool...')
        pool = ThreadPool()
        list(pool.starmap(download_file, origin_and_dest))

    elapsed = time.time() - start_time
    print('Time spent on download: {}'.format(humanfriendly.format_timespan(elapsed)))


if __name__ == '__main__':
    main()

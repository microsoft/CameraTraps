r"""
Download images multi-threaded. Accepts either JSON or text file as input.

1) JSON file format: either the output of a MegaDB query or the Batch Detection
    API. It should be a list of entry dicts, where each entry has fields
    - <save_key> (defaults to 'download_id'): str, usually dataset_name+uuid,
        uuid doesn't have to be from the database
    - 'file': str, either <blob_name> if 'dataset' is in entry, or
        <dataset>/<blob_name> if 'dataset' is not in entry
    - 'dataset': str, if 'file' is <blob_name>
    - 'new_entry': bool, if setting the --only-new-images flag

    Example entry:
    {
        "bbox": [{
            "category": "animal",
            "bbox": [ 0.3876, 0.3667, 0.1046, 0.22 ]
        }],
        "file": "animals/0467/0962.jpg",
        "dataset": "wcs",
        "location": "3282",
        "download_id": "wcs+8a066ce6-3f14-11ea-aae7-9801a7a664ab",
        "new_entry": true
    }

2) Text file format: one line per file, each line is <dataset_name>/<blob_name>.

The environment variables COSMOS_ENDPOINT and COSMOS_KEY need to be set to
access the Datasets table.

The downloading threads may get stuck (request timeouts) and stop downloading.
Re-start a couple of times until all are downloaded.

Example usage for downloading images queried from MegaDB:
    python download_images.py json file_list.json /save/images/to/dir \
        --save-key "download_id" --threads 50 --only-new-images

Example usage for downloading images from a Batch Detection API output:
    python download_images.py json detections.json /save/images/to/dir \
        --save-key "files" --key "images" --threads 50

Example usage for downloading images from a text file:
    python download_images.py txt file_list.txt /save/images/to/dir --threads 50
"""
import argparse
from concurrent import futures
from datetime import datetime
import json
import os
import shutil
from typing import Any, Container, Dict, Mapping, Optional, Tuple

import requests
from tqdm import tqdm

from data_management.megadb.megadb_utils import MegadbUtils
from sas_blob_utils import build_azure_storage_uri  # ai4eutils


def download_file(url: str, filename: str, store_dir: str,
                  check_existing_dir: Optional[str] = None) -> None:
    """Saves a file to <store_dir>/<filename>. If the file already exists at
    <check_existing_dir>/<filename>, then creates a hard-link. Otherwise,
    attempts to download from <url>.
    """
    dst = os.path.normpath(os.path.join(store_dir, filename))
    dirname = os.path.dirname(dst)
    os.makedirs(dirname, exist_ok=True)

    if check_existing_dir is not None:
        src = os.path.normpath(os.path.join(check_existing_dir, filename))
        if os.path.exists(src):
            try:
                os.link(src, dst)
                return
            except OSError:
                try:
                    shutil.copy(src, dst)
                    return
                except OSError:
                    pass

    assert not os.path.exists(dst)
    response = requests.get(url)
    response.raise_for_status()
    with open(dst, 'wb') as f:
        f.write(response.content)


def construct_url(img_path: str, datasets_table: Mapping[str, Any],
                  dataset_name: Optional[str] = None) -> str:
    """Builds Azure SAS storage URL.

    Args:
        img_path: str, either <dataset_name>/<blob> (set dataset_name=None)
            or <img_file> without the 'path_prefix' from dataset_table
        datasets_table: dict, from MegaDB
        dataset_name: optional str

    Returns: str, URL with SAS token
    """
    if dataset_name is None:
        dataset_name, blob = img_path.split('/', maxsplit=1)
    else:
        blob = img_path
        path_prefix = datasets_table[dataset_name].get('path_prefix', '')
        if len(path_prefix) > 0:
            blob = path_prefix + '/' + blob

    sas_token = datasets_table[dataset_name]['container_sas_key']
    if sas_token[0] == '?':
        sas_token = sas_token[1:]

    url = build_azure_storage_uri(
        account=datasets_table[dataset_name]['storage_account'],
        container=datasets_table[dataset_name]['container'],
        blob=blob, sas_token=sas_token)
    return url


def process_json(file_list_path: str,
                 save_key: Optional[str],
                 json_key: Optional[str],
                 only_new_images: bool,
                 existing: Container[str],
                 datasets_table: Mapping[str, Any]
                 ) -> Tuple[Dict[str, str], int]:
    """Processes JSON file.

    Args:
        file_list_path: str, path to JSON file
        existing: set of str, paths (in local OS format)

    Returns:
        filename_to_url: dict, maps filename (str) to url (str)
        count: int, number of entries in original JSON file
    """
    with open(file_list_path) as f:
        file_list = json.load(f)
    if json_key is not None:
        file_list = file_list[json_key]

    count = len(file_list)

    entries_to_download = {}  # filename -> entry
    for entry in file_list:
        if only_new_images and 'new_entry' not in entry:
            continue

        # build filename
        _, ext = os.path.splitext(entry['file'])
        filename = entry[save_key]
        if not filename.endswith(ext):
            filename += ext

        if os.path.normpath(filename) in existing:
            continue
        assert filename not in entries_to_download
        entries_to_download[filename] = entry

    # if need to re-download a dataset's images in case of corruption
    # entries_to_download = {
    #     filename: entry for filename, entry in entries_to_download.items()
    #     if entry['dataset'] == DATASET
    # }

    filename_to_url = {}
    for filename, entry in entries_to_download.items():
        try:
            url = construct_url(img_path=entry['file'],
                                datasets_table=datasets_table,
                                dataset_name=entry.get('dataset', None))
            filename_to_url[filename] = url
        except Exception as e:  # pylint: disable=broad-except
            exception_type = type(e).__name__
            print(f'construct_url() generated {exception_type}: {e}')

    return filename_to_url, count


def process_txt(file_list_path: str,
                existing: Container[str],
                datasets_table: Mapping[str, Any]
                ) -> Tuple[Dict[str, str], int]:
    """Processes text file.

    Args:
        file_list_path: str, path to text file, each line in file is a filename
        existing: set of str, paths (in local OS format)

    Returns:
        filename_to_url: dict, maps filename (str) to url (str)
        count: int, number of entries in original text file
    """
    filename_to_url = {}
    count = 0
    with open(file_list_path, 'r') as f:
        for filename in f:
            count += 1
            filename = filename.rstrip()  # remove newline
            if os.path.normpath(filename) in existing or len(filename) == 0:
                continue

            try:
                url = construct_url(img_path=filename,
                                    datasets_table=datasets_table)
                filename_to_url[filename] = url
            except Exception as e:  # pylint: disable=broad-except
                exception_type = type(e).__name__
                print(f'construct_url() generated {exception_type}: {e}')
    return filename_to_url, count


def main(filetype: str,
         file_list_path: str,
         store_dir: str,
         save_key: Optional[str],
         json_key: Optional[str],
         only_new_images: bool,
         threads: int,
         check_existing_dir: Optional[str]) -> None:
    # input validation
    assert filetype in ['json', 'txt']
    if check_existing_dir is not None:
        assert os.path.isdir(check_existing_dir)
        assert check_existing_dir != store_dir

    if os.path.exists(store_dir):
        assert os.path.isdir(store_dir)
        print('Searching for existing files')
        # existing files, with paths relative to <store_dir>
        existing = set(
            os.path.relpath(os.path.join(dirpath, f), store_dir)
            for dirpath, _, filenames in os.walk(store_dir) for f in filenames)
    else:
        print('Creating directory at:', store_dir)
        os.makedirs(store_dir)
        existing = set()

    print('Loading datasets table from MegaDB')
    datasets_table = MegadbUtils().get_datasets_table()

    # parse JSON or TXT file
    print('Processing file list')
    if filetype == 'json':
        filename_to_url, count = process_json(
            file_list_path, save_key, json_key, only_new_images,
            existing=existing, datasets_table=datasets_table)
    else:
        filename_to_url, count = process_txt(
            file_list_path, existing=existing, datasets_table=datasets_table)

    print(f'file_list has {count} items, still need to download '
          f'{len(filename_to_url)} items')

    print('Submitting URLs to download')
    pool = futures.ThreadPoolExecutor(max_workers=threads)
    future_to_filename = {}
    for filename, url in tqdm(filename_to_url.items()):
        future = pool.submit(download_file, url, filename, store_dir,
                             check_existing_dir)
        future_to_filename[future] = filename

    print('Fetching results')
    total = len(future_to_filename)
    failed_filenames = []
    for future in tqdm(futures.as_completed(future_to_filename), total=total):
        filename = future_to_filename[future]
        try:
            future.result()
        except Exception as e:  # pylint: disable=broad-except
            exception_type = type(e).__name__
            tqdm.write(f'{filename} - generated {exception_type}: {e}')
            failed_filenames.append(filename)

    if len(failed_filenames) > 0:
        print(f'{len(failed_filenames)} failed to download. Writing log...')
        date = datetime.now().strftime('%Y%m%d_%H%M%S')  # ex: '20200722_110816'
        with open(f'download_images_failed_{date}.json', 'w') as f:
            json.dump(failed_filenames, f, indent=1)
    else:
        print('Success!')


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Script to download image files')
    parser.add_argument(
        'type', choices=['json', 'txt'],
        help='whether the input file is in JSON format, or a text file list')
    parser.add_argument(
        'file_list_json',
        help='path to JSON file containing list of image entries')
    parser.add_argument(
        'store_dir',
        help='Path to a directory to store the downloaded files')
    parser.add_argument(
        '-s', '--save-key', default='download_id',
        help='(type=json) key inside each image entry corresponding to image '
             'save path')
    parser.add_argument(
        '-k', '--key',
        help='(type=json) optional key, if the list of image entries '
             'corresponds to a particular key inside file_list_json')
    parser.add_argument(
        '--only-new-images', action='store_true',
        help='(type=json) only download images whose entries contain '
             '{"new_entry": true}')
    parser.add_argument(
        '-t', '--threads', type=int, default=1,
        help='number of threads to download files')
    parser.add_argument(
        '--check-existing-dir',
        help='path to directory that may already contain desired image, in '
             'which case a hard link is created to that image')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(filetype=args.type,
         file_list_path=args.file_list_json,
         store_dir=args.store_dir,
         save_key=args.save_key,
         json_key=args.key,
         threads=args.threads,
         only_new_images=args.only_new_images,
         check_existing_dir=args.check_existing_dir)

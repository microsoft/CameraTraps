"""
Example usage:
    python analyze_failed_images.py failed.json \
        -a ACCOUNT -c CONTAINER -s SAS_TOKEN
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent import futures
import json
from pprint import pprint
import threading
from typing import Any, Optional

from PIL import Image, ImageFile
import requests
from tqdm import tqdm

from data_management.megadb.megadb_utils import MegadbUtils
import path_utils  # from ai4eutils
import sas_blob_utils  # from ai4eutils


ImageFile.LOAD_TRUNCATED_IMAGES = False


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze a list of images that failed to download or crop.')
    parser.add_argument(
        'failed_images', metavar='URL_OR_PATH',
        help='URL or path to text or JSON file containing list of image paths')
    parser.add_argument(
        '-k', '--json-keys', nargs='*',
        help='list of keys in JSON file containing image paths')
    parser.add_argument(
        '-a', '--account',
        help='name of Azure Blob Storage account. If not given, then image '
             'paths are assumed to start with the dataset name, so we can look '
             'up the account from MegaDB.')
    parser.add_argument(
        '-c', '--container',
        help='name of Azure Blob Storage container. If not given, then image '
             'paths are assumed to start with the dataset name, so we can look '
             'up the container from MegaDB.')
    parser.add_argument(
        '-s', '--sas-token',
        help='optional SAS token (without leading "?") if the container is not '
             'publicly accessible. If account and container not given, then '
             'image paths are assumed to start with the dataset name, so we '
             'can look up the SAS Token from MegaDB.')
    return parser.parse_args()


def check_image_condition(img_path: str,
                          truncated_images_lock: threading.Lock,
                          account: Optional[str] = None,
                          container: Optional[str] = None,
                          sas_token: Optional[str] = None,
                          datasets_table: Optional[Mapping[str, Any]] = None
                          ) -> tuple[str, str]:
    """
    Args:
        img_path: str, either <blob_name> if datasets_table is None, or
            <dataset>/<blob_name> if datasets_table is given
        account: str, name of Azure Blob Storage account
        container: str, name of Azure Blob Storage container
        sas_token: str, optional SAS token (without leading '?') if the
            container is not publicly accessible
        datasets_table: dict, maps dataset name to dict of information

    Returns: (img_file, status) tuple, where status is one of
        'nonexistant': blob does not exist in the container
        'non_image': img_file does not have valid file extension
        'good': image exists and is able to be opened without setting
            ImageFile.LOAD_TRUNCATED_IMAGES=True
        'truncated': image exists but can only be opened by setting
            ImageFile.LOAD_TRUNCATED_IMAGES=True
        'bad': image exists, but cannot be opened even when setting
            ImageFile.LOAD_TRUNCATED_IMAGES=True
    """
    if (account is None) or (container is None) or (datasets_table is not None):
        assert account is None
        assert container is None
        assert sas_token is None
        assert datasets_table is not None

        dataset, img_file = img_path.split('/', maxsplit=1)
        account = datasets_table[dataset]['storage_account']
        container = datasets_table[dataset]['container']
        sas_token = datasets_table[dataset]['container_sas_key']
        if sas_token[0] == '?':  # strip leading '?' from SAS token
            sas_token = sas_token[1:]
    else:
        img_file = img_path

    if not path_utils.is_image_file(img_file):
        return img_file, 'non_image'

    blob_url = sas_blob_utils.build_azure_storage_uri(
        account=account, container=container, sas_token=sas_token,
        blob=img_file)
    blob_exists = sas_blob_utils.check_blob_exists(blob_url)
    if not blob_exists:
        return img_file, 'nonexistant'

    stream, _ = sas_blob_utils.download_blob_to_stream(blob_url)
    stream.seek(0)
    try:
        with truncated_images_lock:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            with Image.open(stream) as img:
                img.load()
        return img_file, 'good'
    except OSError as e:  # PIL.UnidentifiedImageError is a subclass of OSError
        try:
            stream.seek(0)
            with truncated_images_lock:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                with Image.open(stream) as img:
                    img.load()
            return img_file, 'truncated'
        except Exception as e:  # pylint: disable=broad-except
            exception_type = type(e).__name__
            tqdm.write(f'Unable to load {img_file}. {exception_type}: {e}.')
            return img_file, 'bad'


def analyze_images(url_or_path: str, json_keys: Optional[Sequence[str]] = None,
                   account: Optional[str] = None,
                   container: Optional[str] = None,
                   sas_token: Optional[str] = None) -> None:
    """
    Args:
        url_or_path: str, URL or local path to a file containing a list
            of image paths. Each image path is either <blob_name> if account and
            container are given, or <dataset>/<blob_name> if account and
            container are None. File can either be a list of image paths, or a
            JSON file containing image paths.
        json_keys: optional list of str, only relevant if url_or_path is a JSON
            file. If json_keys=None, then the JSON file at url_or_path is
            assumed to be a JSON list of image paths. If json_keys is not None,
            then the JSON file should be a dict, whose values corresponding to
            json_keys are lists of image paths.
        account: str, name of Azure Blob Storage account
        container: str, name of Azure Blob Storage container
        sas_token: str, optional SAS token (without leading '?') if the
            container is not publicly accessible
    """
    datasets_table = None
    if (account is None) or (container is None):
        assert account is None
        assert container is None
        assert sas_token is None
        datasets_table = MegadbUtils().get_datasets_table()

    is_json = ('.json' in url_or_path)
    if url_or_path.startswith(('http://', 'https://')):
        r = requests.get(url_or_path)
        if is_json:
            img_paths = r.json()
        else:
            img_paths = r.text.splitlines()
    else:
        with open(url_or_path, 'r') as f:
            if is_json:
                img_paths = json.load(f)
            else:
                img_paths = f.readlines()

    if is_json and json_keys is not None:
        img_paths_json = img_paths
        img_paths = []
        for k in json_keys:
            img_paths += img_paths_json[k]

    mapping: dict[str, list[str]] = {
        status: []
        for status in ['good', 'nonexistant', 'non_image', 'truncated', 'bad']
    }

    pool = futures.ThreadPoolExecutor(max_workers=100)

    # lock before changing ImageFile.LOAD_TRUNCATED_IMAGES
    truncated_images_lock = threading.Lock()

    futures_list = []
    for img_path in tqdm(img_paths):
        future = pool.submit(
            check_image_condition, img_path, truncated_images_lock, account,
            container, sas_token, datasets_table)
        futures_list.append(future)

    total = len(futures_list)
    for future in tqdm(futures.as_completed(futures_list), total=total):
        img_file, status = future.result()
        mapping[status].append(img_file)

    for status, img_list in mapping.items():
        print(f'{status}: {len(img_list)}')
        pprint(sorted(img_list))


if __name__ == '__main__':
    args = _parse_args()
    analyze_images(url_or_path=args.failed_images, json_keys=args.json_keys,
                   account=args.account, container=args.container,
                   sas_token=args.sas_token)

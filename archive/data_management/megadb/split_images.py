"""
For a directory of images downloaded using e.g. download_images.py, split
them into train/val/test folders based on location. The location information needs
to be present in the file_list of entries. All images without location
information will go into the training folder. This allows inference to be performed
on individual images instead of only on tfrecords.

The environment variables COSMOS_ENDPOINT and COSMOS_KEY need to be set.
"""

import argparse
import json
import os
import time
from collections import defaultdict
from shutil import move

import humanfriendly
from tqdm import tqdm

from data_management.megadb.megadb_utils import Splits, MegadbUtils


def look_up_split(splits_table, entry):
    dataset_name = entry['dataset']

    if 'location' not in entry:
        return Splits.TRAIN
    else:
        split_lists = splits_table[dataset_name]
        if entry['location'] in split_lists[Splits.TRAIN]:  # checks membership in a set
            return Splits.TRAIN
        elif entry['location'] in split_lists[Splits.VAL]:
            return Splits.VAL
        elif entry['location'] in split_lists[Splits.TEST]:
            return Splits.TEST
        else:
            if dataset_name not in ['nacti', 'wcs']:
                print('Entry file {} in dataset {} has location not in predefined splits; moving to train'.format(
                    entry['file'], dataset_name
                ))
            return Splits.TRAIN


def main():
    parser = argparse.ArgumentParser(
        description='Script to split downloaded image files into train/val/test folders'
    )
    parser.add_argument(
        'file_list'
    )
    parser.add_argument(
        '--origin_dir',
        required=True,
        help='Path to a directory storing the downloaded files'
    )
    parser.add_argument(
        '--dest_dir',
        required=True,
        help='Path to a directory where the train/val/test folders are or will be created'
    )

    args = parser.parse_args()

    megadb_utils = MegadbUtils()
    splits_table = megadb_utils.get_splits_table()
    print('Obtained the splits table.')

    assert os.path.exists(args.file_list)
    assert os.path.exists(args.origin_dir)

    os.makedirs(args.dest_dir, exist_ok=True)

    dest_folders = {
        Splits.TRAIN: os.path.join(args.dest_dir, Splits.TRAIN),
        Splits.VAL: os.path.join(args.dest_dir, Splits.VAL),
        Splits.TEST: os.path.join(args.dest_dir, Splits.TEST)
    }

    for folder in dest_folders:
        os.makedirs(dest_folders[folder], exist_ok=True)

    print('Loading file_list...')
    with open(args.file_list) as f:
        file_list = json.load(f)

    counter = defaultdict(lambda: defaultdict(int))
    count = 0

    start_time = time.time()
    for entry in tqdm(file_list):
        which_split = look_up_split(splits_table, entry)
        counter[entry['dataset']][which_split] += 1

        ext = os.path.splitext(entry['file'])[1]
        download_id = entry['download_id'] + ext
        origin_path = os.path.join(args.origin_dir, download_id)

        dest_path = os.path.join(args.dest_dir, which_split, download_id)
        if os.path.exists(dest_path):
            continue # already moved in a previous run of the script

        if not os.path.exists(origin_path):
            print('Image not found in origin dir at {}'.format(origin_path))
            continue

        dest = move(origin_path, dest_path)

        count += 1
        if count % 50000 == 0:
            print(counter)
            print()

    elapsed = time.time() - start_time
    print('Time spent on moving files: {}'.format(humanfriendly.format_timespan(elapsed)))
    print(counter)
    with open('mdv5_splits_tally.json', 'w') as f:
        json.dump(counter, f, indent=1)


if __name__ == '__main__':
    main()

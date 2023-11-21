"""
Creates a classification dataset CSV with a corresponding JSON file determining
the train/val/test split.

This script takes as input a "queried images" JSON file whose keys are paths to
images and values are dictionaries containing information relevant for training
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

We assume that the tuple (dataset, location) identifies a unique location. In
other words, we assume that no two datasets have overlapping locations. This
probably isn't 100% true, but it's pretty much the best we can do in terms of
avoiding overlapping locations between the train/val/test splits.

This script outputs 3 files to <output_dir>:

1) classification_ds.csv, contains columns:
    - 'path': str, path to cropped images
    - 'dataset': str, name of dataset
    - 'location': str, location that image was taken, as saved in MegaDB
    - 'dataset_class': str, original class assigned to image, as saved in MegaDB
    - 'confidence': float, confidence that this crop is of an actual animal,
        1.0 if the crop is a "ground truth bounding box" (i.e., from MegaDB),
        <= 1.0 if the bounding box was detected by MegaDetector
    - 'label': str, comma-separated list of label(s) assigned to this crop for
        the sake of classification

2) label_index.json: maps integer to label name
    - keys are string representations of Python integers (JSON requires keys to
        be strings), numbered from 0 to num_labels-1
    - values are strings, label names

3) splits.json: serialization of a Python dict that maps each split
    ['train', 'val', 'test'] to a list of length-2 lists, where each inner list
    is [<dataset>, <location>]

Example usage:
    python create_classification_dataset.py \
        run_idfg2 \
        --queried-images-json run_idfg2/queried_images.json \
        --cropped-images-dir /ssd/crops_sq \
        -d $HOME/classifier-training/mdcache -v "4.1" -t 0.8
"""
from __future__ import annotations

import argparse
from collections.abc import Container, MutableMapping
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from classification import detect_and_crop


DATASET_FILENAME = 'classification_ds.csv'
LABEL_INDEX_FILENAME = 'label_index.json'
SPLITS_FILENAME = 'splits.json'


def main(output_dir: str,
         mode: list[str],
         match_test: Optional[list[str]],
         queried_images_json_path: Optional[str],
         cropped_images_dir: Optional[str],
         detector_version: Optional[str],
         detector_output_cache_base_dir: Optional[str],
         confidence_threshold: Optional[float],
         min_locs: Optional[int],
         val_frac: Optional[float],
         test_frac: Optional[float],
         splits_method: Optional[str],
         label_spec_json_path: Optional[str]) -> None:
    """Main function."""
    # input validation
    assert set(mode) <= {'csv', 'splits'}
    if label_spec_json_path is not None:
        assert splits_method == 'smallest_first'

    test_set_locs = None  # set of (dataset, location) tuples
    test_set_df = None
    if match_test is not None:
        match_test_csv_path, match_test_splits_path = match_test
        match_df = pd.read_csv(match_test_csv_path, index_col=False,
                               float_precision='high')
        with open(match_test_splits_path, 'r') as f:
            match_splits = json.load(f)
        test_set_locs = set((loc[0], loc[1]) for loc in match_splits['test'])
        ds_locs = pd.Series(zip(match_df['dataset'], match_df['location']))
        test_set_df = match_df[ds_locs.isin(test_set_locs)]

    dataset_path = os.path.join(output_dir, DATASET_FILENAME)

    if 'csv' in mode:
        assert queried_images_json_path is not None
        assert cropped_images_dir is not None
        assert detector_version is not None
        assert detector_output_cache_base_dir is not None
        assert confidence_threshold is not None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f'Created {output_dir}')

        df, log = create_classification_csv(
            queried_images_json_path=queried_images_json_path,
            detector_output_cache_base_dir=detector_output_cache_base_dir,
            detector_version=detector_version,
            cropped_images_dir=cropped_images_dir,
            confidence_threshold=confidence_threshold,
            min_locs=min_locs,
            append_df=test_set_df,
            exclude_locs=test_set_locs)
        print('Saving classification dataset CSV')
        df.to_csv(dataset_path, index=False)
        for msg, img_list in log.items():
            print(f'{msg}:', len(img_list))

        # create label index JSON
        labels = df['label']
        if any(labels.str.contains(',')):
            print('multi-label!')
            labels = labels.map(lambda x: x.split(',')).explode()
            # look into sklearn.preprocessing.MultiLabelBinarizer
        label_names = sorted(labels.unique())
        with open(os.path.join(output_dir, LABEL_INDEX_FILENAME), 'w') as f:
            # Note: JSON always saves keys as strings!
            json.dump(dict(enumerate(label_names)), f, indent=1)

    if 'splits' in mode:
        assert splits_method is not None
        assert val_frac is not None
        assert (match_test is None) != (test_frac is None)
        if test_frac is None:
            test_frac = 0

        print(f'Creating splits using "{splits_method}" method...')
        df = pd.read_csv(dataset_path, index_col=False, float_precision='high')

        if splits_method == 'random':
            split_to_locs = create_splits_random(
                df, val_frac, test_frac, test_split=test_set_locs)
        else:
            split_to_locs = create_splits_smallest_label_first(
                df, val_frac, test_frac, test_split=test_set_locs,
                label_spec_json_path=label_spec_json_path)
        with open(os.path.join(output_dir, SPLITS_FILENAME), 'w') as f:
            json.dump(split_to_locs, f, indent=1)


def create_classification_csv(
        queried_images_json_path: str,
        detector_output_cache_base_dir: str,
        detector_version: str,
        cropped_images_dir: str,
        confidence_threshold: float,
        min_locs: Optional[int] = None,
        append_df: Optional[pd.DataFrame] = None,
        exclude_locs: Optional[Container[tuple[str, str]]] = None
        ) -> tuple[pd.DataFrame, dict[str, list]]:
    """Creates a classification dataset.

    The classification dataset is a pd.DataFrame with columns:
    - path: str, <dataset>/<crop-filename>
    - dataset: str, name of camera trap dataset
    - location: str, location of image, provided by the camera trap dataset
    - dataset_class: image class, as provided by the camera trap dataset
    - confidence: float, confidence of bounding box, 1 if ground truth
    - label: str, comma-separated list of classification labels

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
        confidence_threshold: float, only crop bounding boxes above this value
        min_locs: optional int, minimum # of locations that each label must
            have in order to be included
        append_df: optional pd.DataFrame, existing DataFrame that is appended to
            the classification CSV
        exclude_locs: optional set of (dataset, location) tuples, crops from
            these locations are excluded (does not affect append_df)

    Returns:
        df: pd.DataFrame, the classification dataset
        log: dict, with the following keys
            'images missing detections': list of str, images without ground
                truth bboxes and not in detection cache
            'images without confident detections': list of str, images in
                detection cache with no bboxes above the confidence threshold
            'missing crops': list of tuple (img_path, i), where i is the
                i-th crop index
    """
    assert 0 <= confidence_threshold <= 1

    columns = [
        'path', 'dataset', 'location', 'dataset_class', 'confidence', 'label']
    if append_df is not None:
        assert (append_df.columns == columns).all()

    with open(queried_images_json_path, 'r') as f:
        js = json.load(f)

    print('loading detection cache...', end='')
    detector_output_cache_dir = os.path.join(
        detector_output_cache_base_dir, f'v{detector_version}')
    datasets = set(img_path[:img_path.find('/')] for img_path in js)
    detection_cache, cat_id_to_name = detect_and_crop.load_detection_cache(
        detector_output_cache_dir=detector_output_cache_dir, datasets=datasets)
    print('done!')

    missing_detections = []  # no cached detections or ground truth bboxes
    images_no_confident_detections = []  # cached detections contain 0 bboxes
    images_missing_crop = []  # tuples: (img_path, crop_index)
    all_rows = []

    # True for ground truth, False for MegaDetector
    # always save as .jpg for consistency
    crop_path_template = {
        True: '{img_path}___crop{n:>02d}.jpg',
        False: '{img_path}___crop{n:>02d}_' + f'mdv{detector_version}.jpg'
    }

    for img_path, img_info in tqdm(js.items()):
        ds, img_file = img_path.split('/', maxsplit=1)

        # get bounding boxes
        if 'bbox' in img_info:  # ground-truth bounding boxes
            bbox_dicts = img_info['bbox']
            is_ground_truth = True
        else:  # get bounding boxes from detector cache
            if img_file in detection_cache[ds]:
                bbox_dicts = detection_cache[ds][img_file]['detections']
                # convert from category ID to category name
                for d in bbox_dicts:
                    d['category'] = cat_id_to_name[d['category']]
            else:
                missing_detections.append(img_path)
                continue
            is_ground_truth = False

        # check if crops are already downloaded, and ignore bboxes below the
        # confidence threshold
        rows = []
        for i, bbox_dict in enumerate(bbox_dicts):
            conf = 1.0 if is_ground_truth else bbox_dict['conf']
            if conf < confidence_threshold:
                continue
            if bbox_dict['category'] != 'animal':
                tqdm.write(f'Bbox {i} of {img_path} is non-animal. Skipping.')
                continue
            crop_path = crop_path_template[is_ground_truth].format(
                img_path=img_path, n=i)
            full_crop_path = os.path.join(cropped_images_dir, crop_path)
            if not os.path.exists(full_crop_path):
                images_missing_crop.append((img_path, i))
            else:
                # assign all images without location info to 'unknown_location'
                img_loc = img_info.get('location', 'unknown_location')
                row = [crop_path, ds, img_loc, img_info['class'],
                       conf, ','.join(img_info['label'])]
                rows.append(row)
        if len(rows) == 0:
            images_no_confident_detections.append(img_path)
            continue
        all_rows += rows

    df = pd.DataFrame(data=all_rows, columns=columns)

    # remove images from labels that have fewer than min_locs locations
    if min_locs is not None:
        nlocs_per_label = df.groupby('label').apply(
            lambda xdf: len(xdf[['dataset', 'location']].drop_duplicates()))
        valid_labels_mask = (nlocs_per_label >= min_locs)
        valid_labels = nlocs_per_label.index[valid_labels_mask]
        invalid_labels = nlocs_per_label.index[~valid_labels_mask]
        orig = len(df)
        df = df[df['label'].isin(valid_labels)]
        print(f'Excluding {orig - len(df)} crops from {len(invalid_labels)} '
              'labels:', invalid_labels.tolist())

    if exclude_locs is not None:
        mask = ~pd.Series(zip(df['dataset'], df['location'])).isin(exclude_locs)
        print(f'Excluding {(~mask).sum()} crops from CSV')
        df = df[mask]
    if append_df is not None:
        print(f'Appending {len(append_df)} rows to CSV')
        df = df.append(append_df)

    log = {
        'images missing detections': missing_detections,
        'images without confident detections': images_no_confident_detections,
        'missing crops': images_missing_crop
    }
    return df, log


def create_splits_random(df: pd.DataFrame, val_frac: float,
                         test_frac: float = 0.,
                         test_split: Optional[set[tuple[str, str]]] = None,
                         ) -> dict[str, list[tuple[str, str]]]:
    """
    Args:
        df: pd.DataFrame, contains columns ['dataset', 'location', 'label']
            each row is a single image
            assumes each image is assigned exactly 1 label
        val_frac: float, desired fraction of dataset to use for val set
        test_frac: float, desired fraction of dataset to use for test set,
            must be 0 if test_split is given
        test_split: optional set of (dataset, location) tuples to use as test
            split

    Returns: dict, keys are ['train', 'val', 'test'], values are lists of locs,
        where each loc is a tuple (dataset, location)
    """
    if test_split is not None:
        assert test_frac == 0
    train_frac = 1. - val_frac - test_frac
    targets = {'train': train_frac, 'val': val_frac, 'test': test_frac}

    # merge dataset and location into a single string '<dataset>/<location>'
    df['dataset_location'] = df['dataset'] + '/' + df['location']

    # create DataFrame of counts. rows = locations, columns = labels
    loc_label_counts = (df.groupby(['label', 'dataset_location']).size()
                        .unstack('label', fill_value=0))
    num_locs = len(loc_label_counts)

    # label_count: label => number of examples
    # loc_count: label => number of locs containing that label
    label_count = loc_label_counts.sum()
    loc_count = (loc_label_counts > 0).sum()

    best_score = np.inf  # lower is better
    best_splits = None
    for _ in tqdm(range(10_000)):

        # generate a new split
        num_train = int(num_locs * (train_frac + np.random.uniform(-.03, .03)))
        if test_frac > 0:
            num_val = int(num_locs * (val_frac + np.random.uniform(-.03, .03)))
        else:
            num_val = num_locs - num_train
        permuted_locs = loc_label_counts.index[np.random.permutation(num_locs)]
        split_to_locs = {'train': permuted_locs[:num_train],
                         'val': permuted_locs[num_train:num_train + num_val]}
        if test_frac > 0:
            split_to_locs['test'] = permuted_locs[num_train + num_val:]

        # score the split
        score = 0.
        for split, locs in split_to_locs.items():
            split_df = loc_label_counts.loc[locs]
            target = targets[split]

            # SSE for # of images per label (with 2x weight)
            crop_frac = split_df.sum() / label_count
            score += 2 * ((crop_frac - target) ** 2).sum()

            # SSE for # of locs per label
            loc_frac = (split_df > 0).sum() / loc_count
            score += ((loc_frac - target) ** 2).sum()

        if score < best_score:
            tqdm.write(f'New lowest score: {score}')
            best_score = score
            best_splits = split_to_locs

    assert best_splits is not None
    split_to_locs = {
        s: sorted(locs.map(lambda x: tuple(x.split('/', maxsplit=1))))
        for s, locs in best_splits.items()
    }
    if test_split is not None:
        split_to_locs['test'] = test_split
    return split_to_locs


def create_splits_smallest_label_first(
        df: pd.DataFrame,
        val_frac: float,
        test_frac: float = 0.,
        label_spec_json_path: Optional[str] = None,
        test_split: Optional[set[tuple[str, str]]] = None,
        ) -> dict[str, list[tuple[str, str]]]:
    """
    Args:
        df: pd.DataFrame, contains columns ['dataset', 'location', 'label']
            each row is a single image
            assumes each image is assigned exactly 1 label
        val_frac: float, desired fraction of dataset to use for val set
        test_frac: float, desired fraction of dataset to use for test set,
            must be 0 if test_split is given
        label_spec_json_path: optional str, path to label spec JSON
        test_split: optional set of (dataset, location) tuples to use as test
            split

    Returns: dict, keys are ['train', 'val', 'test'], values are lists of locs,
        where each loc is a tuple (dataset, location)
    """
    # label => list of datasets to prioritize for test and validation sets
    prioritize = {}
    if label_spec_json_path is not None:
        with open(label_spec_json_path, 'r') as f:
            label_spec_js = json.load(f)
        for label, label_spec in label_spec_js.items():
            if 'prioritize' in label_spec:
                datasets = []
                for level in label_spec['prioritize']:
                    datasets += level
                prioritize[label] = datasets

    # merge dataset and location into a tuple (dataset, location)
    df['dataset_location'] = list(zip(df['dataset'], df['location']))
    loc_to_label_sizes = df.groupby(['dataset_location', 'label']).size()

    seen_locs = set()
    split_to_locs: dict[str, list[tuple[str, str]]] = dict(
        train=[], val=[], test=[])
    label_sizes_by_split = {
        label: dict(train=0, val=0, test=0)
        for label in df['label'].unique()
    }
    if test_split is not None:
        assert test_frac == 0
        split_to_locs['test'] = list(test_split)
        seen_locs.update(test_split)

    def add_loc_to_split(loc: tuple[str, str], split: str) -> None:
        split_to_locs[split].append(loc)
        for label, label_size in loc_to_label_sizes[loc].items():
            label_sizes_by_split[label][split] += label_size

    # sorted smallest to largest
    ordered_labels = df.groupby('label').size().sort_values()
    for label, label_size in tqdm(ordered_labels.items()):

        split_sizes = label_sizes_by_split[label]
        test_thresh = test_frac * label_size
        val_thresh = val_frac * label_size

        mask = df['label'] == label
        ordered_locs = sort_locs_by_size(
            loc_to_size=df[mask].groupby('dataset_location').size().to_dict(),
            prioritize=prioritize.get(label, None))
        ordered_locs = [loc for loc in ordered_labels if loc not in seen_locs]

        for loc in ordered_locs:
            seen_locs.add(loc)
            # greedily add to test set until it has >= 15% of images
            if split_sizes['test'] < test_thresh:
                split = 'test'
            elif split_sizes['val'] < val_thresh:
                split = 'val'
            else:
                split = 'train'
            add_loc_to_split(loc, split)
        seen_locs.update(ordered_locs)

    # sort the resulting locs
    split_to_locs = {s: sorted(locs) for s, locs in split_to_locs.items()}
    return split_to_locs


def sort_locs_by_size(loc_to_size: MutableMapping[tuple[str, str], int],
                      prioritize: Optional[Container[str]] = None
                      ) -> list[tuple[str, str]]:
    """Sorts locations by size, optionally prioritizing locations from certain
    datasets first.

    Args:
        loc_to_size: dict, maps each (dataset, location) tuple to its size,
            modified in-place
        prioritize: optional list of str, datasets to prioritize

    Returns: list of (dataset, location) tuples, ordered from smallest size to
        largest. Locations from prioritized datasets come first.
    """
    result = []
    if prioritize is not None:
        # modify loc_to_size in place, so copy its keys before iterating
        prioritized_loc_to_size = {
            loc: loc_to_size.pop(loc) for loc in list(loc_to_size.keys())
            if loc[0] in prioritize
        }
        result = sort_locs_by_size(prioritized_loc_to_size)

    result += sorted(loc_to_size, key=loc_to_size.__getitem__)
    return result


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Creates classification dataset.')

    # arguments relevant to both creating the dataset CSV and splits.json
    parser.add_argument(
        'output_dir',
        help='path to directory where the 3 output files should be '
             'saved: 1) dataset CSV, 2) label index JSON, 3) splits JSON')
    parser.add_argument(
        '--mode', nargs='+', choices=['csv', 'splits'],
        default=['csv', 'splits'],
        help='whether to generate only a CSV, only a splits.json file (based '
             'on an existing classification_ds.csv), or both')
    parser.add_argument(
        '--match-test', nargs=2, metavar=('CLASSIFICATION_CSV', 'SPLITS_JSON'),
        help='path to an existing classification CSV and path to an existing '
             'splits JSON file from which to match test set')

    # arguments only relevant for creating the dataset CSV
    csv_group = parser.add_argument_group(
        'arguments for creating classification CSV')
    csv_group.add_argument(
        '-q', '--queried-images-json',
        help='path to JSON file containing image paths and classification info')
    csv_group.add_argument(
        '-c', '--cropped-images-dir',
        help='path to local directory for saving crops of bounding boxes')
    csv_group.add_argument(
        '-d', '--detector-output-cache-dir',
        help='(required) path to directory where detector outputs are cached')
    csv_group.add_argument(
        '-v', '--detector-version',
        help='(required) detector version string, e.g., "4.1"')
    csv_group.add_argument(
        '-t', '--threshold', type=float, default=0.8,
        help='confidence threshold above which to crop bounding boxes')
    csv_group.add_argument(
        '--min-locs', type=int,
        help='minimum number of locations that each label must have in order '
             'to be included (does not apply to match-test-splits)')

    # arguments only relevant for creating the splits JSON
    splits_group = parser.add_argument_group(
        'arguments for creating train/val/test splits')
    splits_group.add_argument(
        '--val-frac', type=float,
        help='(required) fraction of data to use for validation split')
    splits_group.add_argument(
        '--test-frac', type=float,
        help='fraction of data to use for test split, must be provided if '
             '--match-test is not given')
    splits_group.add_argument(
        '--method', choices=['random', 'smallest_first'], default='random',
        help='"random": randomly tries up to 10,000 different train/val/test '
             'splits and chooses the one that best meets the scoring criteria, '
             'does not support --label-spec. '
             '"smallest_first": greedily divides locations into splits '
             'starting with the smallest class first. Supports --label-spec.')
    splits_group.add_argument(
        '--label-spec',
        help='optional path to label specification JSON file, if specifying '
             'dataset priority. Requires --method=smallest_first.')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(output_dir=args.output_dir,
         mode=args.mode,
         match_test=args.match_test,
         queried_images_json_path=args.queried_images_json,
         cropped_images_dir=args.cropped_images_dir,
         detector_version=args.detector_version,
         detector_output_cache_base_dir=args.detector_output_cache_dir,
         confidence_threshold=args.threshold,
         min_locs=args.min_locs,
         val_frac=args.val_frac,
         test_frac=args.test_frac,
         splits_method=args.method,
         label_spec_json_path=args.label_spec)

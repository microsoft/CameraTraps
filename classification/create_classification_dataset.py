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
probably isn't 100% true, but it's probably the best we can do in terms of
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
        be strings), numbered from 0 to num_labels
    - values are strings, label names

2) splits.json: serialization of a Python dict that maps each split
    ['train', 'val', 'test'] to a list of length-2 lists, where each inner list
    is [<dataset>, <location>]

Example usage:
    python create_classification_dataset.py \
        run_idfg2/queried_images.json /ssd/crops_sq \
        -c $HOME/classifier-training/mdcache -v "4.1" -t 0.8 \
        -d run_idfg2/classifcation_ds.csv -s run_idfg2/splits.json
"""
import argparse
import json
import os
from typing import Container, Dict, List, Mapping, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from classification import detect_and_crop


def main(queried_images_json_path: str,
         detector_version: str,
         detector_output_cache_base_dir: str,
         cropped_images_dir: str,
         output_dir: str,
         confidence_threshold: float,
         label_spec_json_path: Optional[str] = None,
         match_test_csv_path: Optional[str] = None,
         match_test_splits_path: Optional[str] = None
         ) -> None:
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
        output_dir: str, path to directory to save dataset CSV, splits JSON, and
            label index JSON
        confidence_threshold: float, only crop bounding boxes above this value
        label_spec_json_path: optional str, path to label spec JSON
        match_test_csv_path: optional str, path to existing classification CSV
        match_test_splits_path: optional str, path to existing splits JSON
    """
    # input validation
    assert 0 <= confidence_threshold <= 1
    if (match_test_csv_path is None) != (match_test_splits_path is None):
        raise ValueError('both match_test_csv_path and match_test_splits_path '
                         'must be given together')

    exclude_locs = None  # set of (dataset, location) tuples
    append_df = None
    if match_test_splits_path is not None:
        with open(match_test_splits_path, 'r') as f:
            match_splits = json.load(f)
        test_set_locs = set(tuple(loc) for loc in match_splits['test'])
        match_df = pd.read_csv(match_test_csv_path, index_col=False,
                               float_precision='high')
        ds_locs = pd.Series(zip(match_df['dataset'], match_df['location']))
        test_set_df = match_df[ds_locs.isin(test_set_locs)]
        exclude_locs = test_set_locs
        append_df = test_set_df

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created {output_dir}')
    csv_save_path = os.path.join(output_dir, 'classification_ds.csv')
    result = create_crops_csv(queried_images_json_path,
                              detector_version,
                              detector_output_cache_base_dir,
                              cropped_images_dir,
                              confidence_threshold,
                              csv_save_path,
                              append_df=append_df,
                              exclude_locs=exclude_locs)

    missing_detections, images_no_confident_detections, missing_crops = result
    print('Images missing detections:', len(missing_detections))
    print('Images without confident detections:',
          len(images_no_confident_detections))
    print('Missing crops:', len(missing_crops))

    crops_df = pd.read_csv(
        csv_save_path, index_col=False, float_precision='high')

    # create label index JSON
    labels = crops_df['label']
    if any(crops_df['label'].str.contains(',')):
        print('multi-label!')
        labels = labels.map(lambda x: x.split(',')).explode()
        # look into sklearn.preprocessing.MultiLabelBinarizer
    label_names = sorted(labels.unique())
    with open(os.path.join(output_dir, 'label_index.json'), 'w') as f:
        # Note: JSON always saves keys as strings!
        json.dump(dict(enumerate(label_names)), f, indent=1)

    prioritize = None
    if label_spec_json_path is not None:
        with open(label_spec_json_path, 'r') as f:
            label_spec_js = json.load(f)
        prioritize = {}
        for label, label_spec in label_spec_js.items():
            if 'prioritize' in label_spec:
                datasets = []
                for level in label_spec['prioritize']:
                    datasets.extend(level)
                prioritize[label] = datasets

    print('Creating splits...')
    split_to_locs = create_splits(crops_df, prioritize=prioritize)
    with open(os.path.join(output_dir, 'splits.json'), 'w') as f:
        json.dump(split_to_locs, f, indent=1)


def create_crops_csv(queried_images_json_path: str,
                     detector_version: str,
                     detector_output_cache_base_dir: str,
                     cropped_images_dir: str,
                     confidence_threshold: float,
                     csv_save_path: str,
                     append_df: Optional[pd.DataFrame] = None,
                     exclude_locs: Optional[Container[Tuple[str, str]]] = None
                     ) -> Tuple[List[str], List[str], List[Tuple[str, int]]]:
    """Creates a classification dataset CSV.

    The classification dataset CSV contains the columns
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
        csv_save_path: str, path to save dataset csv
        append_df: optional pd.DataFrame, existing DataFrame that is appended to
            the classification CSV
        exclude_locs: optional set of (dataset, location) tuples, crops from
            these locations are excluded (does not affect append_df)

    Returns:
        missing_detections: list of str, images without ground truth
            bboxes and not in detection cache
        images_no_confident_detections: list of str, images in detection cache
            with no bboxes above the confidence threshold
        images_missing_crop: list of tuple (img_path, i), where i is the i-th
            crop index
    """
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
    # always save as JPG for consistency
    crop_path_template = {
        True: '{img_path_root}_crop{n:>02d}.jpg',
        False: '{img_path_root}_mdv{v}_crop{n:>02d}.jpg'
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
            img_path_root = os.path.splitext(img_path)[0]
            crop_path = crop_path_template[is_ground_truth].format(
                img_path_root=img_path_root, v=detector_version, n=i)
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
        all_rows.extend(rows)

    df = pd.DataFrame(data=all_rows, columns=columns)
    if exclude_locs is not None:
        mask = ~pd.Series(zip(df['dataset'], df['location'])).isin(exclude_locs)
        print(f'Excluding {(~mask).sum()} crops from CSV')
        df = df[mask]
    if append_df is not None:
        print(f'Appending {len(append_df)} rows to CSV')
        df = df.append(append_df)

    print('Saving classification dataset CSV...', end='')
    df.to_csv(csv_save_path, index=False)
    print('done!')

    return (missing_detections,
            images_no_confident_detections,
            images_missing_crop)


def sort_locs_by_size(loc_to_size: Mapping[Tuple[str, str], int],
                      prioritize: Optional[Container[str]] = None
                      ) -> List[Tuple[str, str]]:
    """Given a dict mapping each (dataset, location) tuple to its size, returns
    a list of (dataset, location) tuples, ordered from smallest size to largest.
    If a list of datasets to prioritize is given, then locations from those
    datasets come first.
    """
    result = []
    if prioritize is not None:
        prioritized_loc_to_size = {}
        remaining_loc_to_size = {}
        for loc, size in loc_to_size.items():
            if loc[0] in prioritize:
                prioritized_loc_to_size[loc] = size
            else:
                remaining_loc_to_size[loc] = size
        loc_to_size = remaining_loc_to_size
        result.extend(sort_locs_by_size(prioritized_loc_to_size))

    result.extend(sorted(loc_to_size, key=loc_to_size.__getitem__))
    return result


def create_splits(df: pd.DataFrame,
                  prioritize: Optional[Mapping[str, Container[str]]] = None
                  ) -> Dict[str, List[Tuple[str, str]]]:
    """
    Args:
        df: pd.DataFrame, contains columns ['dataset', 'location', 'label']
            each row is a single image
            assumes each image is assigned exactly 1 label
        prioritize: optional dict, label => list of datasets to prioritize
            for inclusion in the test and validation sets

    Returns: dict, keys are ['train', 'val', 'test'], values are lists of locs,
        where each loc is a tuple (dataset, location)
    """
    # merge dataset and location into a tuple (dataset, location)
    df['dataset_location'] = list(zip(df['dataset'], df['location']))

    loc_to_label_sizes = df.groupby(['dataset_location', 'label']).size()

    seen_locs = set()
    split_to_locs: Dict[str, List[Tuple[str, str]]] = dict(
        train=[], val=[], test=[])
    label_sizes_by_split = {
        label: dict(train=0, val=0, test=0)
        for label in df['label'].unique()
    }

    def add_loc_to_split(loc: Tuple[str, str], split: str) -> None:
        split_to_locs[split].append(loc)
        for label, label_size in loc_to_label_sizes[loc].items():
            label_sizes_by_split[label][split] += label_size

    # sorted smallest to largest
    ordered_labels = df.groupby('label').size().sort_values()
    for label, label_size in tqdm(ordered_labels.items()):

        mask = df['label'] == label
        ordered_locs = sort_locs_by_size(
            loc_to_size=df[mask].groupby('dataset_location').size().to_dict(),
            prioritize=prioritize)

        for loc in ordered_locs:
            if loc in seen_locs:
                continue
            seen_locs.add(loc)

            # greedily add to test set until it has >= 15% of images
            if label_sizes_by_split[label]['test'] < 0.15 * label_size:
                split = 'test'
            elif label_sizes_by_split[label]['val'] < 0.15 * label_size:
                split = 'val'
            else:
                split = 'train'
            add_loc_to_split(loc, split)

    # sort the resulting locs
    split_to_locs = {
        split: sorted(locs) for split, locs in split_to_locs.items()
    }
    return split_to_locs


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Creates classification dataset.')
    parser.add_argument(
        'queried_images_json',
        help='path to JSON file containing image paths and classification info')
    parser.add_argument(
        'cropped_images_dir',
        help='path to local directory for saving crops of bounding boxes')
    parser.add_argument(
        '-c', '--detector-output-cache-dir', required=True,
        help='(required) path to directory where detector outputs are cached')
    parser.add_argument(
        '-v', '--detector-version', required=True,
        help='(required) detector version string, e.g., "4.1"')
    parser.add_argument(
        '-o', '--output-dir', required=True,
        help='(required) path to directory where the 3 output files should be '
             'saved: 1) dataset CSV, 2) label index JSON, 3) splits JSON')
    parser.add_argument(
        '-t', '--confidence-threshold', type=float, default=0.8,
        help='confidence threshold above which to crop bounding boxes')
    parser.add_argument(
        '--label-spec',
        help='optional path to label specification JSON file, if specifying '
             'dataset priority')
    parser.add_argument(
        '--match-test-csv',
        help='path to an existing classification CSV from which to match '
             'val/test crops (requires --match-test-splits)')
    parser.add_argument(
        '--match-test-splits',
        help='path to existing split JSON file from which to match val/test '
             'locations (requires --match-test-csv)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(queried_images_json_path=args.queried_images_json,
         detector_version=args.detector_version,
         detector_output_cache_base_dir=args.detector_output_cache_dir,
         cropped_images_dir=args.cropped_images_dir,
         output_dir=args.output_dir,
         confidence_threshold=args.confidence_threshold,
         label_spec_json_path=args.label_spec,
         match_test_csv_path=args.match_test_csv,
         match_test_splits_path=args.match_test_splits)

r"""
Merges classification results with Batch Detection API outputs.

This script takes 2 main files as input:
1) Either a "dataset CSV" (output of create_classification_dataset.py) or a
    "classification results CSV" (output of evaluate_model.py). The CSV is
    expected to have columns listed below. The 'label' and [label names] columns
    are optional, but at least one of them must be provided.
    - 'path': str, path to cropped image
        - if passing in a detections JSON, must match
            <img_file>___cropXX_mdvY.Y.jpg
        - if passing in a queried images JSON, must match
            <dataset>/<img_file>___cropXX_mdvY.Y.jpg or
            <dataset>/<img_file>___cropXX.jpg
    - 'label': str, label assigned to this crop
    - [label names]: float, confidence in each label
2) Either a "detections JSON" (output of MegaDetector) or a "queried images
    JSON" (output of json_validatory.py).

If the CSV contains [label names] columns (e.g., output of evaluate_model.py),
then each crop's "classifications" output will have one value per category.
Categories are sorted decreasing by confidence.
    "classifications": [
        ["3", 0.901],
        ["1", 0.071],
        ["4", 0.025],
        ["2", 0.003],
    ]

If the CSV only contains the 'label' column (e.g., output of
create_classification_dataset.py), then each crop's "classifications" output
will have only one value, with a confidence of 1.0. The label's classification
category ID is always greater than 1,000,000, to distinguish it from a predicted
category ID.
    "classifications": [
        ["1000004", 1.0]
    ]

If the CSV contains both [label names] and 'label' columns, then both the
predicted categories and label category will be included. By default, the
label-category is included last; if the --label-first flag is given, then the
label catgory is placed first in the results.
    "classifications": [
        ["1000004", 1.0],  # label put first if --label-first flag is given
        ["3", 0.901],  # all other results are sorted by confidence
        ["1", 0.071],
        ["4", 0.025],
        ["2", 0.003]
    ]

Example usage:
    python merge_classification_detection_output.py \
        BASE_LOGDIR/LOGDIR/outputs_test.csv.gz \
        BASE_LOGDIR/label_index.json \
        BASE_LOGDIR/queried_images.json \
        --classifier-name "efficientnet-b3-idfg-moredata" \
        --detector-output-cache-dir $HOME/classifier-training/mdcache \
        --detector-version "4.1" \
        --output-json BASE_LOGDIR/LOGDIR/classifier_results.json \
        --datasets idfg idfg_swwlf_2019
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import datetime
import json
import os
from typing import Any

import pandas as pd
from tqdm import tqdm

from ct_utils import truncate_float


def row_to_classification_list(row: Mapping[str, Any],
                               label_names: Sequence[str],
                               contains_preds: bool,
                               label_pos: str | None,
                               threshold: float,
                               relative_conf: bool = False
                               ) -> list[tuple[str, float]]:
    """Given a mapping from label name to output probability, returns a list of
    tuples, (str(label_id), prob), which can be serialized into the Batch API
    output format.

    The list of tuples is returned in sorted order by the predicted probability
    for each label.

    If 'label' is in row and label_pos is not None, then we add
    (label_id + 1_000_000, 1.) to the list. If label_pos='first', we put this at
    the front of the list. Otherwise, we put it at the end.
    """
    contains_label = ('label' in row)
    assert contains_label or contains_preds
    if relative_conf:
        assert contains_label and contains_preds

    result = []
    if contains_preds:
        result = [(str(i), row[label]) for i, label in enumerate(label_names)]
        if relative_conf:
            label_conf = row[row['label']]
            result = [(k, max(v - label_conf, 0)) for k, v in result]

        # filter out confidences below the threshold, and set precision to 4
        result = [
            (k, truncate_float(conf, precision=4))
            for k, conf in result if conf >= threshold
        ]

        # sort from highest to lowest probability
        result = sorted(result, key=lambda x: x[1], reverse=True)

    if contains_label and label_pos is not None:
        label = row['label']
        label_id = label_names.index(label)
        item = (str(label_id + 1_000_000), 1.)
        if label_pos == 'first':
            result = [item] + result
        else:
            result.append(item)
    return result


def main(classification_csv_path: str,
         label_names_json_path: str,
         output_json_path: str,
         classifier_name: str,
         threshold: float,
         datasets: Sequence[str] | None,
         detection_json_path: str | None,
         queried_images_json_path: str | None,
         detector_output_cache_base_dir: str | None,
         detector_version: str | None,
         samples_per_label: int | None,
         seed: int,
         label_pos: str | None,
         relative_conf: bool,
         typical_confidence_threshold: float) -> None:
    """Main function."""
    # input validation
    assert os.path.exists(classification_csv_path)
    assert os.path.exists(label_names_json_path)
    assert 0 <= threshold <= 1
    for x in [detection_json_path, queried_images_json_path]:
        if x is not None:
            assert os.path.exists(x)
    assert label_pos in [None, 'first', 'last']

    # load classification CSV
    print('Loading classification CSV...')
    df = pd.read_csv(classification_csv_path, float_precision='high',
                     index_col='path')
    if relative_conf or label_pos is not None:
        assert 'label' in df.columns

    # load label names
    with open(label_names_json_path, 'r') as f:
        idx_to_label = json.load(f)
    label_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
    if 'label' in df.columns:
        for i, label in enumerate(label_names):
            idx_to_label[str(i + 1_000_000)] = f'label: {label}'

    if queried_images_json_path is not None:
        assert detector_output_cache_base_dir is not None
        assert detector_version is not None
        detection_js = process_queried_images(
            df=df, queried_images_json_path=queried_images_json_path,
            detector_output_cache_base_dir=detector_output_cache_base_dir,
            detector_version=detector_version, datasets=datasets,
            samples_per_label=samples_per_label, seed=seed)
    elif detection_json_path is not None:
        with open(detection_json_path, 'r') as f:
            detection_js = json.load(f)
        images = {}
        for img in detection_js['images']:
            path = img['file']
            if datasets is None or path[:path.find('/')] in datasets:
                images[path] = img
        detection_js['images'] = images

    classification_time = datetime.date.fromtimestamp(
        os.path.getmtime(classification_csv_path))
    classifier_timestamp = classification_time.strftime('%Y-%m-%d %H:%M:%S')

    classification_js = combine_classification_with_detection(
        detection_js=detection_js, df=df, idx_to_label=idx_to_label,
        label_names=label_names, classifier_name=classifier_name,
        classifier_timestamp=classifier_timestamp, threshold=threshold,
        label_pos=label_pos, relative_conf=relative_conf,
        typical_confidence_threshold=typical_confidence_threshold)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(classification_js, f, indent=1)

    print('Wrote merged classification/detection results to {}'.format(output_json_path))


def process_queried_images(
         df: pd.DataFrame,
         queried_images_json_path: str,
         detector_output_cache_base_dir: str,
         detector_version: str,
         datasets: Sequence[str] | None = None,
         samples_per_label: int | None = None,
         seed: int = 123
         ) -> dict[str, Any]:
    """Creates a detection JSON object roughly in the Batch API detection
    format.

    Detections are either ground-truth (from the queried images JSON) or
    retrieved from the detector output cache. Only images corresponding to crop
    paths from the given pd.DataFrame are included in the detection JSON.

    Args:
        df: pd.DataFrame, either a "classification dataset CSV" or a
            "classification results CSV",  column 'path' has format
            <dataset>/<img_file>___cropXX[...].jpg
        queried_images_json_path: str, path to queried images JSON
        detector_output_cache_base_dir: str
        detector_version: str
        datasets: optional list of str, only crops from these datasets will be
            be included in the output, set to None to include all datasets
        samples_per_label: optional int, if not None, then randomly sample this
            many bounding boxes per label (each label must have at least this
            many examples)
        seed: int, used for random sampling if samples_per_label is not None

    Returns: dict, detections JSON file, except that the 'images' field is a
        dict (img_path => dict) instead of a list
    """
    # input validation
    assert os.path.exists(queried_images_json_path)
    detection_cache_dir = os.path.join(
        detector_output_cache_base_dir, f'v{detector_version}')
    assert os.path.isdir(detection_cache_dir)

    # extract dataset name from crop path so we can process 1 dataset at a time
    df['dataset'] = df.index.map(lambda x: x[:x.find('/')])
    unique_datasets = df['dataset'].unique()

    if datasets is not None:
        for ds in datasets:
            assert ds in unique_datasets
        df = df[df['dataset'].isin(datasets)]  # filter by dataset
    else:
        datasets = unique_datasets

    # randomly sample images for each class
    if samples_per_label is not None:
        print(f'Sampling {samples_per_label} bounding boxes per label')
        df = df.groupby('label').sample(samples_per_label, random_state=seed)

    # load queried images JSON, needed for ground-truth bbox info
    with open(queried_images_json_path, 'r') as f:
        queried_images_js = json.load(f)

    merged_js: dict[str, Any] = {
        'images': {},  # start as dict, will convert to list later
        'info': {}
    }
    images = merged_js['images']

    for ds in datasets:
        print('processing dataset:', ds)
        ds_df = df[df['dataset'] == ds]

        with open(os.path.join(detection_cache_dir, f'{ds}.json'), 'r') as f:
            detection_js = json.load(f)
        img_file_to_index = {
            im['file']: idx
            for idx, im in enumerate(detection_js['images'])
        }

        # compare info dicts
        class_info = merged_js['info']
        detection_info = detection_js['info']
        key = 'detector'
        if key not in class_info:
            class_info[key] = detection_info[key]
        assert class_info[key] == detection_info[key]

        # compare detection categories
        key = 'detection_categories'
        if key not in merged_js:
            merged_js[key] = detection_js[key]
        assert merged_js[key] == detection_js[key]
        cat_to_catid = {v: k for k, v in detection_js[key].items()}

        for crop_path in tqdm(ds_df.index):
            # crop_path: <dataset>/<img_file>___cropXX_mdvY.Y.jpg
            #            [----<img_path>----]       [-<suffix>--]
            img_path, suffix = crop_path.split('___crop')
            img_file = img_path[img_path.find('/') + 1:]

            # file has detection entry
            if '_mdv' in suffix and img_path not in images:
                img_idx = img_file_to_index[img_file]
                images[img_path] = detection_js['images'][img_idx]
                images[img_path]['file'] = img_path

            # bounding box is from ground truth
            elif img_path not in images:
                images[img_path] = {
                    'file': img_path,
                    'detections': [
                        {
                            'category': cat_to_catid[bbox_dict['category']],
                            'conf': 1.0,
                            'bbox': bbox_dict['bbox']
                        }
                        for bbox_dict in queried_images_js[img_path]['bbox']
                    ]
                }
    return merged_js


def combine_classification_with_detection(
        detection_js: dict[str, Any],
        df: pd.DataFrame,
        idx_to_label: Mapping[str, str],
        label_names: Sequence[str],
        classifier_name: str,
        classifier_timestamp: str,
        threshold: float,
        label_pos: str | None = None,
        relative_conf: bool = False,
        typical_confidence_threshold: float = None
        ) -> dict[str, Any]:
    """Adds classification information to a detection JSON. Classification
    information may include the true label and/or the predicted confidences
    of each label.

    Args:
        detection_js: dict, detections JSON file, except that the 'images'
            field is a dict (img_path => dict) instead of a list
        df: pd.DataFrame, classification results, indexed by crop path
        idx_to_label: dict, str(label_id) => label name, may also include
            str(label_id + 1e6) => 'label: {label_name}'
        label_names: list of str, label names
        classifier_name: str, name of classifier to include in output JSON
        classifier_timestamp: str, timestamp to include in output JSON
        threshold: float, for each crop, omit classification results for
            categories whose confidence is below this threshold
        label_pos: one of [None, 'first', 'last']
            None: do not include labels in the output JSON
            'first' / 'last': position in classification list to put the label
        relative_conf: bool, if True then for each class, outputs its relative
            confidence over the confidence of the true label, requires 'label'
            to be in CSV
        typical_confidence_threshold: float, useful default confidence
            threshold; not used directly, just passed along to the output file

    Returns: dict, detections JSON file updated with classification results
    """
    classification_metadata = {
        'classifier': classifier_name,
        'classification_completion_time': classifier_timestamp
    }

    if typical_confidence_threshold is not None:
        classification_metadata['classifier_metadata'] = \
        {'typical_classification_threshold':typical_confidence_threshold}

    detection_js['info'].update(classification_metadata)
    detection_js['classification_categories'] = idx_to_label

    contains_preds = (set(label_names) <= set(df.columns))
    if not contains_preds:
        print('CSV does not contain predictions. Outputting labels only.')

    images = detection_js['images']

    for crop_path in tqdm(df.index):
        # crop_path: <dataset>/<img_file>___cropXX_mdvY.Y.jpg
        #            [----<img_path>----]       [-<suffix>--]
        img_path, suffix = crop_path.split('___crop')
        crop_index = int(suffix[:2])

        detection_dict = images[img_path]['detections'][crop_index]
        detection_dict['classifications'] = row_to_classification_list(
            row=df.loc[crop_path], label_names=label_names,
            contains_preds=contains_preds, label_pos=label_pos,
            threshold=threshold, relative_conf=relative_conf)

    detection_js['images'] = list(images.values())
    return detection_js


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Merges classification results with Batch Detection API '
                    'outputs.')
    parser.add_argument(
        'classification_csv',
        help='path to classification CSV')
    parser.add_argument(
        'label_names_json',
        help='path to JSON file mapping label index to label name')
    parser.add_argument(
        '-o', '--output-json', required=True,
        help='(required) path to save output JSON with both detection and '
             'classification results')
    parser.add_argument(
        '-n', '--classifier-name', required=True,
        help='(required) name of classifier')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.1,
        help='Confidence threshold between 0 and 1. In the output file, omit '
             'classifier results on classes whose confidence is below this '
             'threshold.')
    parser.add_argument(
        '-d', '--datasets', nargs='*',
        help='optionally limit output to images from certain datasets. Assumes '
             'that image paths are given as <dataset>/<img_file>.')
    parser.add_argument(
        '--typical-confidence-threshold', type=float, default=None,
        help='useful default confidence threshold; not used directly, just '
             'passed along to the output file')

    detection_json_group = parser.add_argument_group(
        'arguments for passing in a detections JSON file')
    detection_json_group.add_argument(
        '-j', '--detection-json',
        help='path to detections JSON file')

    queried_images_group = parser.add_argument_group(
        'arguments for passing in a queried images JSON file')
    queried_images_group.add_argument(
        '-q', '--queried-images-json',
        help='path to queried images JSON file')
    queried_images_group.add_argument(
        '-c', '--detector-output-cache-dir',
        help='(required) path to directory where detector outputs are cached')
    queried_images_group.add_argument(
        '-v', '--detector-version',
        help='(required) detector version string, e.g., "4.1"')
    queried_images_group.add_argument(
        '-s', '--samples-per-label', type=int,
        help='randomly sample this many bounding boxes per label (each label '
             'must have at least this many examples)')
    queried_images_group.add_argument(
        '--seed', type=int, default=123,
        help='random seed, only used if --samples-per-label is given')
    queried_images_group.add_argument(
        '--label', choices=['first', 'last'], default=None,
        help='Whether to put the label first or last in the list of '
             'classifications. If this argument is omitted, then no labels are '
             'included in the output.')
    queried_images_group.add_argument(
        '--relative-conf', action='store_true',
        help='for each class, outputs its relative confidence over the '
             'confidence of the true label, requires "label" to be in CSV')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(classification_csv_path=args.classification_csv,
         label_names_json_path=args.label_names_json,
         output_json_path=args.output_json,
         classifier_name=args.classifier_name,
         threshold=args.threshold,
         datasets=args.datasets,
         detection_json_path=args.detection_json,
         queried_images_json_path=args.queried_images_json,
         detector_output_cache_base_dir=args.detector_output_cache_dir,
         detector_version=args.detector_version,
         samples_per_label=args.samples_per_label,
         seed=args.seed,
         label_pos=args.label,
         relative_conf=args.relative_conf,
         typical_confidence_threshold=args.typical_confidence_threshold)

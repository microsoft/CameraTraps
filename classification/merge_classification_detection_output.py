r"""
Merges classification results with Batch Detection API outputs.

Takes as input a CSV containing columns:
- 'path': str, path to cropped image, <dataset>/<crop-file-name>
- 'label': str, label assigned to this crop
- [label names]: float, confidence in each label

The 'label' and [label names] columns are optional, but at least one of them
must be proided.

If the CSV contains [label names] columns (e.g., output of evaluate_model.py),
then each crop's "classifications" output will have one value per category.
If the CSV also contains a 'label' column and the --label-first flag is given,
then the label category is placed first in the results. Other categories are
sorted decreasing by confidence.
    "classifications": [
        ["4", 0.025],  # label always first, even if classifier is incorrect
        ["3", 0.901],
        ["1", 0.071],
        ["2", 0.003]
    ]

If the CSV only contains the 'label' column (e.g., output of
create_classification_dataset.py), then each crop's "classifications" output
will have only one value, with a confidence of 1.0. Other categories are not
included.
    "classifications": [
        ["4", 1.0]
    ]

Example usage:
    python merge_classification_detection_output.py \
        run_idfg_moredata/20200814_084806/outputs_test.csv.gz \
        run_idfg_moredata/20200814_084806/label_index.json \
        run_idfg_moredata/queried_images.json \
        -n "efficientnet-b3-idfg-moredata" \
        -c $HOME/classifier-training/mdcache -v "4.1" \
        -o run_idfg_moredata/20200814_084806/classifier_results.json \
        -d idfg idfg_swwlf_2019
"""
import argparse
import datetime
import json
import os
from typing import List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm


def row_to_classification_list(row: Mapping[str, float],
                               label_names: Sequence[str],
                               contains_preds: bool,
                               label_first: bool,
                               relative_conf: bool = False
                               ) -> List[Tuple[str, float]]:
    """Given a mapping from label name to output probability, returns a list of
    tuples, (str(label_id), prob), which can be serialized into the Batch API
    output format.

    The list of tuples is returned in sorted order by the predicted probability
    for each label.

    If 'label' is in row, then we add (label_id + 1_000_000, 1.) to the list. If
    label_first=True, we put this at the front of the list. Otherwise, we put
    it at the end.
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
        # sort from highest to lowest probability
        result = sorted(result, key=lambda x: x[1], reverse=True)

    if contains_label:
        label = row['label']
        label_id = label_names.index(label)
        item = (str(label_id + 1_000_000), 1.)
        if label_first:
            result = [item] + result
        else:
            result.append(item)
    return result


def main(classification_csv_path: str,
         label_names_json_path: str,
         queried_images_json_path: str,
         classifier_name: str,
         detector_output_cache_base_dir: str,
         detector_version: str,
         output_json_path: str,
         datasets: Optional[Sequence[str]] = None,
         samples_per_label: Optional[int] = None,
         seed: int = 123,
         label_first: bool = False,
         relative_conf: bool = False
         ) -> None:
    """Main function."""
    # load classification CSV
    # extract dataset name from img_file so we can process 1 dataset at a time
    print('Loading classification CSV...')
    df = pd.read_csv(classification_csv_path, float_precision='high',
                     index_col=False)
    df['dataset'] = df['path'].str.split('/', n=1, expand=True)[0]
    df.set_index('path', inplace=True)

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

    classification_time = datetime.date.fromtimestamp(
        os.path.getmtime(classification_csv_path))
    classifier_timestamp = classification_time.strftime('%Y-%m-%d %H:%M:%S')

    with open(label_names_json_path, 'r') as f:
        idx_to_label = json.load(f)
    label_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
    if 'label' in df.columns:
        for i, label in enumerate(label_names):
            idx_to_label[str(i + 1_000_000)] = f'label: {label}'

    contains_preds = all(label_name in df.columns for label_name in label_names)
    if not contains_preds:
        print('CSV does not contain predictions. Outputting labels only.')

    with open(queried_images_json_path, 'r') as f:
        queried_images_js = json.load(f)
    img_root_to_full = {
        os.path.splitext(img_path)[0]: img_path
        for img_path in queried_images_js
    }

    detector_output_cache_dir = os.path.join(
        detector_output_cache_base_dir, f'v{detector_version}')

    classification_js = {
        'info': {
            'classifier': classifier_name,
            'classification_completion_time': classifier_timestamp,
            'format_version': "1.0"
        },
        'classification_categories': idx_to_label,
        'images': {}  # start as dict, will convert to list later
    }
    images = classification_js['images']

    for ds in datasets:
        print('processing dataset:', ds)
        ds_df = df[df['dataset'] == ds]

        detection_json_path = os.path.join(
            detector_output_cache_dir, f'{ds}.json')
        with open(detection_json_path, 'r') as f:
            detection_js = json.load(f)

        img_file_to_index = {
            im['file']: idx
            for idx, im in enumerate(detection_js['images'])
        }

        # compare info dicts
        class_info = classification_js['info']
        detection_info = detection_js['info']
        if 'detector' not in class_info:
            class_info['detector'] = detection_info['detector']
        assert class_info['detector'] == detection_info['detector']

        # compare detection categories
        key = 'detection_categories'
        if key not in classification_js:
            classification_js[key] = detection_js[key]
        assert classification_js[key] == detection_js[key]
        cat_to_catid = {v: k for k, v in detection_js[key].items()}

        for crop_path in tqdm(ds_df.index):
            # crop_path: <dataset>/<img_path_root>_<suffix>.jpg
            crop_index = int(crop_path[-6:-4])

            if '_mdv4.1' in crop_path:  # file has detection entry
                img_path_root = crop_path.split('_mdv4.1')[0]
                img_path = img_root_to_full[img_path_root]  # <dataset>/<img_file>

                if img_path not in images:
                    img_file = img_path[img_path.find('/') + 1:]
                    img_idx = img_file_to_index[img_file]
                    images[img_path] = detection_js['images'][img_idx]
                    images[img_path]['file'] = img_path

            else:  # bounding box is from ground truth
                img_path_root = crop_path.split('_crop')[0]
                img_path = img_root_to_full[img_path_root]  # <dataset>/<img_file>

                if img_path not in images:
                    images[img_path] = {
                        'file': img_path,
                        'max_detection_conf': 1.0,
                        'detections': []
                    }
                    for bbox_dict in queried_images_js[img_path]['bbox']:
                        catid = cat_to_catid[bbox_dict['category']]
                        images[img_path]['detections'].append({
                            'category': catid,
                            'conf': 1.0,
                            'bbox': bbox_dict['bbox']
                        })

            detection_dict = images[img_path]['detections'][crop_index]
            detection_dict['classifications'] = row_to_classification_list(
                row=ds_df.loc[crop_path], label_names=label_names,
                contains_preds=contains_preds, label_first=label_first,
                relative_conf=relative_conf)

    classification_js['images'] = list(images.values())

    with open(output_json_path, 'w') as f:
        json.dump(classification_js, f, indent=1)


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
        'queried_images_json',
        help='path to queried images JSON file')
    parser.add_argument(
        '-n', '--classifier-name', required=True,
        help='(required) name of classifier')
    parser.add_argument(
        '-c', '--detector-output-cache-dir', required=True,
        help='(required) path to directory where detector outputs are cached')
    parser.add_argument(
        '-v', '--detector-version', required=True,
        help='(required) detector version string, e.g., "4.1"')
    parser.add_argument(
        '-o', '--output-json', required=True,
        help='(required) path to save output JSON')
    parser.add_argument(
        '-d', '--datasets', nargs='*',
        help='optionally limit output to images from certain datasets')
    parser.add_argument(
        '-s', '--samples-per-label', type=int,
        help='randomly sample this many bounding boxes per label (each label '
             'must have at least this many examples)')
    parser.add_argument(
        '--seed', type=int, default=123,
        help='random seed, only used if --samples-per-label is given')
    parser.add_argument(
        '--label-first', action='store_true',
        help='if <classification_csv> contains a "label" column, always '
             'put the label first in the list of classifications')
    parser.add_argument(
        '--relative-conf', action='store_true',
        help='for each class, outputs its relative confidence over the '
             'confidence of the true label, requires "label" to be in CSV')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(classification_csv_path=args.classification_csv,
         label_names_json_path=args.label_names_json,
         queried_images_json_path=args.queried_images_json,
         classifier_name=args.classifier_name,
         detector_output_cache_base_dir=args.detector_output_cache_dir,
         detector_version=args.detector_version,
         output_json_path=args.output_json,
         datasets=args.datasets,
         samples_per_label=args.samples_per_label,
         seed=args.seed,
         label_first=args.label_first,
         relative_conf=args.relative_conf)

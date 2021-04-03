r"""
Identify images that may have been mislabeled.

A "mislabeled candidate" is defined as an image meeting both criteria:
- according to the ground-truth label, the model made an incorrect prediction
- the model's prediction confidence exceeds its confidence for the ground-truth
    label by at least <margin>

This script outputs for each dataset a text file containing the filenames of
mislabeled candidates, one per line. The text files are saved to:
    <logdir>/mislabeled_candidates_{split}_{dataset}.txt

To this list of files can then be passed to AzCopy to be downloaded:
    azcopy cp "http://<url_of_container>?<sas_token>" "/save/files/here" \
        --list-of-files "/path/to/mislabeled_candidates_{split}_{dataset}.txt"

To save the filename as <dataset_name>/<blob_name> (instead of just <blob_name>
by default), pass the --include-dataset-in-filename flag. Then, the images can
be downloaded with
    python data_management/megadb/download_images.py txt \
        "/path/to/mislabeled_candidates_{split}_{dataset}.txt" \
        /save/files/here \
        --threads 50

Assumes the following directory layout:
    <base_logdir>/
        label_index.json
        <logdir>/
            outputs_{split}.csv.gz

Example usage:
    python identify_mislabeled_candidates.py <base_logdir>/<logdir> \
        --margin 0.5 --splits val test
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
import json
import os

import pandas as pd
from tqdm import tqdm


def main(logdir: str, splits: Iterable[str], margin: float,
         include_dataset_in_filename: bool) -> None:
    # load files
    logdir = os.path.normpath(logdir)  # removes any trailing slash
    base_logdir = os.path.dirname(logdir)
    idx_to_label_json_path = os.path.join(base_logdir, 'label_index.json')
    with open(idx_to_label_json_path, 'r') as f:
        idx_to_label = json.load(f)
    label_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

    for split in splits:
        outputs_csv_path = os.path.join(logdir, f'outputs_{split}.csv.gz')
        candidates_df = get_candidates_df(outputs_csv_path, label_names, margin)

        # dataset => set of img_file
        candidate_image_files: defaultdict[str, set[str]] = defaultdict(set)

        for crop_path in tqdm(candidates_df['path']):
            # crop_path: <dataset>/<img_file>___cropXX_mdvY.Y.jpg
            #            [----<img_path>----]
            img_path = crop_path.split('___crop')[0]
            ds, img_file = img_path.split('/', maxsplit=1)
            if include_dataset_in_filename:
                candidate_image_files[ds].add(img_path)
            else:
                candidate_image_files[ds].add(img_file)

        for ds in sorted(candidate_image_files.keys()):
            img_files = candidate_image_files[ds]
            print(f'{ds} contains {len(img_files)} mislabeled candidates.')
            save_path = os.path.join(
                logdir, f'mislabeled_candidates_{split}_{ds}.txt')
            with open(save_path, 'w') as f:
                for img_file in sorted(img_files):
                    f.write(img_file + '\n')


def get_candidates_df(outputs_csv_path: str, label_names: Sequence[str],
                      margin: float) -> pd.DataFrame:
    """Returns a DataFrame containing crops only from mislabeled candidate
    images.
    """
    df = pd.read_csv(outputs_csv_path, float_precision='high')
    all_rows = range(len(df))
    df['pred'] = df[label_names].idxmax(axis=1)
    df['pred_conf'] = df.lookup(row_labels=all_rows, col_labels=df['pred'])
    df['label_conf'] = df.lookup(row_labels=all_rows, col_labels=df['label'])
    candidate_mask = df['pred_conf'] >= df['label_conf'] + margin
    candidates_df = df[candidate_mask].copy()
    return candidates_df


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Identify mislabeled candidate images.')
    parser.add_argument(
        'logdir',
        help='folder inside <base_logdir> containing `outputs_<split>.csv.gz`')
    parser.add_argument(
        '--margin', type=float, default=0.5,
        help='confidence margin to count as a mislabeled candidate')
    parser.add_argument(
        '--splits', nargs='+', choices=['train', 'val', 'test'],
        help='which splits to identify mislabeled candidates on')
    parser.add_argument(
        '-d', '--include-dataset-in-filename', action='store_true',
        help='save filename as <dataset_name>/<blob_name>')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(logdir=args.logdir, splits=args.splits, margin=args.margin,
         include_dataset_in_filename=args.include_dataset_in_filename)

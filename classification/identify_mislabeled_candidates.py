"""
Identify images that may have been mislabeled.

A "mislabeled candidate" is defined as an image meeting both criteria:
- according to the ground-truth label, the model made an incorrect prediction
- the model's prediction confidence exceeds its confidence for the ground-truth
    label by at least <margin>

This script outputs for each dataset a text file containing the file (blob)
names of mislabeled candidates, one per line. The text files are saved to:
    LOGDIR/mislabeled_candidates_{split}_{dataset}.txt

Assumes the following directory layout:
    base_logdir/
        queried_images.json
        label_index.json
        logdir/
            outputs_{split}.csv.json
"""
import argparse
import json
import os
from typing import Dict, Iterable, Set, Sequence

import pandas as pd
from tqdm import tqdm


def main(logdir: str, splits: Iterable[str], margin: float) -> None:
    # load files
    logdir = os.path.normpath(logdir)  # removes any trailing slash
    base_logdir = os.path.dirname(logdir)

    queried_images_json_path = os.path.join(base_logdir, 'queried_images.json')
    idx_to_label_json_path = os.path.join(base_logdir, 'label_index.json')

    with open(queried_images_json_path, 'r') as f:
        queried_images_js = json.load(f)

    with open(idx_to_label_json_path, 'r') as f:
        idx_to_label = json.load(f)
    label_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

    # map crop paths to list of image paths
    img_root_to_full = {
        os.path.splitext(img_path)[0]: img_path
        for img_path in queried_images_js
    }

    for split in splits:
        outputs_csv_path = os.path.join(logdir, f'outputs_{split}.csv.gz')
        candidates_df = get_candidates_df(outputs_csv_path, label_names, margin)

        # dataset => set of img_file
        candidate_image_files: Dict[str, Set[str]] = {}

        for crop_path in tqdm(candidates_df['img_file']):
            # crop_path: <dataset>/<img_path_root>_<suffix>.jpg
            if '_mdv4.1' in crop_path:  # file has detection entry
                img_path_root = crop_path.split('_mdv4.1')[0]
            else:  # bounding box is from ground truth
                img_path_root = crop_path.split('_crop')[0]

            img_path = img_root_to_full[img_path_root]  # <dataset>/<img_file>
            ds, img_file = img_path.split('/', maxsplit=1)

            if ds not in candidate_image_files:
                candidate_image_files[ds] = set()
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
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(logdir=args.logdir, splits=args.splits, margin=args.margin)

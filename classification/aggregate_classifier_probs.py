r"""Aggregate probabilities from a classifier's outputs according to a mapping
from the desired (target) categories to the classifier's categories.

Using the mapping, create a new version of the classifier output CSV with
probabilities summed within each target category. Also output a new
"index-to-name" JSON file which identifies the sequential order of the target
categories.

Example usage:

python aggregate_classifier_probs.py \
    classifier_output.csv.gz \
    --target-mapping target_to_classifier_labels.json \
    --output-csv classifier_output_remapped.csv.gz \
    --output-label-index label_index_remapped.json
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from tqdm import tqdm

def main(classifier_results_csv_path: str,
         target_mapping_json_path: str,
         output_csv_path: str,
         output_label_index_json_path: str) -> None:
    """Main function.

    Because the output CSV is often very large, we process it in chunks of 1000
    rows at a time.
    """
    chunked_df_iterator = pd.read_csv(
        classifier_results_csv_path, chunksize=1000, float_precision='high',
        index_col='path')

    with open(target_mapping_json_path, 'r') as f:
        target_mapping = json.load(f)
    target_names = sorted(target_mapping.keys())

    all_classifier_labels: set[str] = set()
    for classifier_labels in target_mapping.values():
        assert all_classifier_labels.isdisjoint(classifier_labels)
        all_classifier_labels.update(classifier_labels)

    for i, chunk_df in tqdm(enumerate(chunked_df_iterator)):
        if i == 0:
            assert set(chunk_df.columns) == all_classifier_labels
            header, mode = True, 'w'
        else:
            header, mode = False, 'a'

        agg_df = pd.DataFrame(
            data=0., index=chunk_df.index, columns=target_names)
        for target in target_names:
            classifier_labels = target_mapping[target]
            agg_df[target] = chunk_df[classifier_labels].sum(axis=1)

        agg_df.to_csv(output_csv_path, index=True, header=header, mode=mode)

    with open(output_label_index_json_path, 'w') as f:
        json.dump(dict(enumerate(target_names)), f, indent=1)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Aggregate classifier probabilities to target classes.')
    parser.add_argument(
        'classifier_results_csv',
        help='path to CSV with classifier probabilities')
    parser.add_argument(
        '-t', '--target-mapping', required=True,
        help='path to JSON file mapping target categories to classifier labels')
    parser.add_argument(
        '-o', '--output-csv', required=True,
        help='path to save output CSV with aggregated probabilities')
    parser.add_argument(
        '-i', '--output-label-index', required=True,
        help='path to save output label index JSON')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(classifier_results_csv_path=args.classifier_results_csv,
         target_mapping_json_path=args.target_mapping,
         output_csv_path=args.output_csv,
         output_label_index_json_path=args.output_label_index)

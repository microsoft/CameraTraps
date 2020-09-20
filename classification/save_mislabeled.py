r"""
Update the list of known mislabeled images in MegaDB.

List of known mislabeled images is stored in Azure Blob Storage.
- storage account: cameratrapsc
- container: classifier-training
- blob: megadb_mislabeled/{dataset}.csv, one file per dataset

Each file megadb_mislabeled/{dataset}.csv has two columns:
- 'file': str, blob name
- 'correct_class': optional str, correct dataset class
    - if empty, indicates that the existing class in MegaDB is inaccurate, but
      the correct class is unknown

This script assumes that the classifier-training container is mounted locally.

Takes as input a CSV file (output from Timelapse) with the following columns:
- 'File': str, <blob_basename>
- 'RelativePath': str, <dataset>\<blob_dirname>, note the use of '\' because of
    Windows
- 'mislabeled': str, values in ['true', 'false']
- 'correct_class': either empty or str
"""
import argparse
import os
import pathlib

import pandas as pd


def update_mislabeled_images(container_path: str, input_csv_path: str) -> None:
    """Main function."""
    df = pd.read_csv(input_csv_path, index_col=False)

    # error checking
    assert df['mislabeled'].dtype == bool

    # any row with 'correct_class' should be marked 'mislabeled'
    tmp = (df['correct_class'].notna() & df['mislabeled']).sum()
    assert df['correct_class'].notna().sum() == tmp

    # filter to only the mislabeled rows
    df = df[df['mislabeled']].copy()

    # convert '\' to '/'
    df['RelativePath'] = df['RelativePath'].map(
        lambda p: pathlib.PureWindowsPath(p).as_posix())
    df[['dataset', 'blob_dirname']] = df['RelativePath'].str.split(
        '/', n=1, expand=True)
    df['file'] = df['blob_dirname'] + '/' + df['File']

    for ds, ds_df in df.groupby('dataset'):
        sr_path = os.path.join(container_path, 'megadb_mislabeled', f'{ds}.csv')
        if os.path.exists(sr_path):
            old_sr = pd.read_csv(sr_path, index_col='file', squeeze=True)
        else:
            old_sr = pd.Series(index=pd.Index([], name='file'),
                               dtype='str', name='correct_class')

        ds_sr = ds_df.set_index('file', verify_integrity=True)['correct_class']

        # verify that overlapping indices are the same
        overlap_index = ds_sr.index.intersection(old_sr.index)
        assert ds_sr.loc[overlap_index].equals(old_sr.loc[overlap_index])

        # "add" any new mislabelings
        new_indices = ds_sr.index.difference(old_sr.index)
        new_sr = pd.concat([old_sr, ds_sr.loc[new_indices]],
                           verify_integrity=True)
        new_sr.sort_index(inplace=True)

        # write out results
        new_sr.to_csv(sr_path, index=True)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Merges classification results with Batch Detection API '
                    'outputs.')
    parser.add_argument(
        'container_path',
        help='path to locally-mounted classifier-training container')
    parser.add_argument(
        'input_csv',
        help='path to CSV file output by Timelapse')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    update_mislabeled_images(container_path=args.container_path,
                             input_csv_path=args.input_csv)

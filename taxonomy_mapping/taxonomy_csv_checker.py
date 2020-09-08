"""
Checks the taxonomy CSV file to make sure that for each row:
1) The 'taxonomy_level' column matches the lowest-level taxon level in the
    'taxonomy_string' column.
2) The 'scientific_name' column matches the scientific name from the
    lowest-level taxon level in the 'taxonomy_string' column.

Prints out any mismatches.
"""
import argparse

import pandas as pd


def check_taxonomy_csv(csv_path: str) -> None:
    """See module docstring."""
    taxonomy_df = pd.read_csv(csv_path)

    num_taxon_level_errors = 0
    num_scientific_name_errors = 0

    for i, row in taxonomy_df.iterrows():
        ds = row['dataset_name']
        ds_label = row['query']
        scientific_name = row['scientific_name']
        level = row['taxonomy_level']

        taxa_ancestry = row['taxonomy_string']
        _ = row['source']
        if pd.isna(taxa_ancestry):
            # taxonomy CSV rows without 'taxonomy_string' entries can only be
            # added to the JSON via the 'dataset_labels' key
            continue
        else:
            taxa_ancestry = eval(taxa_ancestry)  # pylint: disable=eval-used

        # each element in taxa_ancestry: id, level, scientific name, common name
        _, taxon_level, taxon_name, _ = taxa_ancestry[0]

        if level != taxon_level:
            print(f'row: {i}, {ds}, {ds_label}')
            print(f'- taxonomy_level column: {level}, '
                  f'level from taxonomy_string: {taxon_level}')
            print()
            num_taxon_level_errors += 1

        if scientific_name != taxon_name:
            print(f'row: {i}, {ds}, {ds_label}')
            print(f'- scientific_name column: {scientific_name}, '
                  f'name from taxonomy_string: {taxon_name}')
            print()
            num_scientific_name_errors += 1

    print('num taxon level errors:', num_taxon_level_errors)
    print('num scientific name errors:', num_scientific_name_errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'taxonomy_csv_path',
        help='path to taxonomy CSV file')
    args = parser.parse_args()

    check_taxonomy_csv(args.taxonomy_csv_path)

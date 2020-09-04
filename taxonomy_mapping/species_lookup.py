#######
#
# species_lookup.py
#
# Look up species names (common or scientific) in the GBIF and iNaturalist
# taxonomies.
#
# Run initialize_taxonomy_lookup() before calling any other function.
#######

#%% Constants and imports

import argparse
from collections import defaultdict
from itertools import compress
import os
import pickle
import shutil
from typing import Any, Dict, List, Mapping, Sequence, Set
import zipfile

import pandas as pd
import numpy as np
from tqdm import tqdm

import ai4e_web_utils

taxonomy_urls = {
    'GBIF': 'http://rs.gbif.org/datasets/backbone/backbone-current.zip',
    'iNaturalist': 'https://www.inaturalist.org/observations/inaturalist-dwca-with-taxa.zip'  # pylint: disable=line-too-long
}

files_to_unzip = {
    'GBIF': ['Taxon.tsv', 'VernacularName.tsv'],
    'iNaturalist': ['taxa.csv']
}

# As of 2020.05.12:
#
# GBIF: ~777MB zipped, ~1.6GB taxonomy
# iNat: ~2.2GB zipped, ~51MB taxonomy

taxonomy_download_dir = r'c:\temp\taxonomy'
os.makedirs(taxonomy_download_dir, exist_ok=True)
for taxonomy_name in taxonomy_urls:
    taxonomy_dir = os.path.join(taxonomy_download_dir, taxonomy_name)
    os.makedirs(taxonomy_dir, exist_ok=True)

serialized_structures_file = os.path.join(taxonomy_download_dir,
                                          'serialized_taxonomies.p')

# These are un-initialized globals that must be initialized by
# the initialize_taxonomy_lookup() function below.
inat_taxonomy: pd.DataFrame
gbif_taxonomy: pd.DataFrame
gbif_common_mapping: pd.DataFrame
inat_taxon_id_to_row: Dict[np.int64, int]
gbif_taxon_id_to_row: Dict[np.int64, int]
inat_taxon_id_to_vernacular: Dict[np.int64, Set[str]]
inat_vernacular_to_taxon_id: Dict[str, np.int64]
inat_taxon_id_to_scientific: Dict[np.int64, Set[str]]
inat_scientific_to_taxon_id: Dict[str, np.int64]
gbif_taxon_id_to_vernacular: Dict[np.int64, Set[str]]
gbif_vernacular_to_taxon_id: Dict[str, np.int64]
gbif_taxon_id_to_scientific: Dict[np.int64, Set[str]]
gbif_scientific_to_taxon_id: Dict[str, np.int64]


#%% Functions

# Initialization function

def initialize_taxonomy_lookup() -> None:
    """
    Initialize this module by doing the following:

    * Downloads and unzips the current GBIF and iNat taxonomies if necessary
      (only unzips what's necessary, but does not delete the original zipfiles)
    * Builds a bunch of dictionaries and tables to facilitate lookup
    * Serializes those tables via pickle
    * Skips all of the above if the serialized pickle file already exists
    """

    global inat_taxonomy,\
        gbif_taxonomy,\
        gbif_common_mapping,\
        inat_taxon_id_to_row,\
        gbif_taxon_id_to_row,\
        inat_taxon_id_to_vernacular,\
        inat_vernacular_to_taxon_id,\
        inat_taxon_id_to_scientific,\
        inat_scientific_to_taxon_id,\
        gbif_taxon_id_to_vernacular,\
        gbif_vernacular_to_taxon_id,\
        gbif_taxon_id_to_scientific,\
        gbif_scientific_to_taxon_id

    ## Load serialized taxonomy info if we've already saved it

    if os.path.isfile(serialized_structures_file):

        print(f'Reading taxonomy data from {serialized_structures_file}')

        with open(serialized_structures_file, 'rb') as f:
            structures_to_serialize = pickle.load(f)

        inat_taxonomy,\
        gbif_taxonomy,\
        gbif_common_mapping,\
        inat_taxon_id_to_row,\
        gbif_taxon_id_to_row,\
        inat_taxon_id_to_vernacular,\
        inat_vernacular_to_taxon_id,\
        inat_taxon_id_to_scientific,\
        inat_scientific_to_taxon_id,\
        gbif_taxon_id_to_vernacular,\
        gbif_vernacular_to_taxon_id,\
        gbif_taxon_id_to_scientific,\
        gbif_scientific_to_taxon_id = structures_to_serialize
        return


    ## If we don't have serialized taxonomy info, create it from scratch.

    # Download and unzip taxonomy files
    for taxonomy_name, zip_url in taxonomy_urls.items():

        need_to_download = False

        # Don't download the zipfile if we've already unzipped what we need
        for fn in files_to_unzip[taxonomy_name]:
            target_file = os.path.join(
                taxonomy_download_dir, taxonomy_name, fn)
            if not os.path.isfile(target_file):
                need_to_download = True
                break
        if not need_to_download:
            print(f'Bypassing download of {taxonomy_name}, all files available')
            continue

        zipfile_path = os.path.join(
            taxonomy_download_dir, zip_url.split('/')[-1])

        # Bypasses download if the file exists already
        ai4e_web_utils.download_url(
            zip_url, os.path.join(zipfile_path),
            progress_updater=ai4e_web_utils.DownloadProgressBar(),
            verbose=True)

        # Unzip the files we need
        files_we_need = files_to_unzip[taxonomy_name]

        with zipfile.ZipFile(zipfile_path, 'r') as zipH:

            for fn in files_we_need:
                print('Unzipping {}'.format(fn))
                target_file = os.path.join(
                    taxonomy_download_dir, taxonomy_name, fn)

                if os.path.isfile(target_file):
                    print(f'Bypassing unzip of {target_file}, file exists')
                else:
                    with zipH.open(fn) as zf, open(target_file, 'wb') as f:
                        shutil.copyfileobj(zf, f)

            # ...for each file that we need from this zipfile

        # Remove the zipfile
        # os.remove(zipfile_path)

    # ...for each taxonomy


    # Create dataframes from each of the taxonomy files, and the GBIF common
    # name file

    # Load iNat taxonomy
    inat_taxonomy = pd.read_csv(os.path.join(taxonomy_download_dir, 'iNaturalist', 'taxa.csv'))
    inat_taxonomy['scientificName'] = inat_taxonomy['scientificName'].fillna('').str.strip()
    inat_taxonomy['vernacularName'] = inat_taxonomy['vernacularName'].fillna('').str.strip()

    # Load GBIF taxonomy
    gbif_taxonomy = pd.read_csv(os.path.join(
        taxonomy_download_dir, 'GBIF', 'Taxon.tsv'), sep='\t')
    gbif_taxonomy['scientificName'] = gbif_taxonomy['scientificName'].fillna('').str.strip()
    gbif_taxonomy['canonicalName'] = gbif_taxonomy['canonicalName'].fillna('').str.strip()

    # Remove questionable rows from the GBIF taxonomy
    gbif_taxonomy = gbif_taxonomy[~gbif_taxonomy['taxonomicStatus'].isin(['doubtful', 'misapplied'])]
    gbif_taxonomy = gbif_taxonomy.reset_index()

    # Load GBIF vernacular name mapping
    gbif_common_mapping = pd.read_csv(os.path.join(
        taxonomy_download_dir, 'GBIF', 'VernacularName.tsv'), sep='\t')
    gbif_common_mapping['vernacularName'] = gbif_common_mapping['vernacularName'].fillna('').str.strip()

    # Only keep English mappings
    gbif_common_mapping = gbif_common_mapping.loc[gbif_common_mapping['language'] == 'en']
    gbif_common_mapping = gbif_common_mapping.reset_index()


    # Convert everything to lowercase

    def convert_df_to_lowercase(df):
        df = df.applymap(lambda s: s.lower() if isinstance(s, str) else s)
        return df

    inat_taxonomy = convert_df_to_lowercase(inat_taxonomy)
    gbif_taxonomy = convert_df_to_lowercase(gbif_taxonomy)
    gbif_common_mapping = convert_df_to_lowercase(gbif_common_mapping)


    # For each taxonomy table, create a mapping from taxon IDs to rows

    inat_taxon_id_to_row = {}
    gbif_taxon_id_to_row = {}

    print('Building iNat taxonID --> row table')
    for i_row, row in tqdm(inat_taxonomy.iterrows(), total=len(inat_taxonomy)):
        inat_taxon_id_to_row[row['taxonID']] = i_row

    print('Building GBIF taxonID --> row table')
    for i_row, row in tqdm(gbif_taxonomy.iterrows(), total=len(gbif_taxonomy)):
        gbif_taxon_id_to_row[row['taxonID']] = i_row


    # Create name mapping dictionaries

    inat_taxon_id_to_vernacular = defaultdict(set)
    inat_vernacular_to_taxon_id = defaultdict(set)
    inat_taxon_id_to_scientific = defaultdict(set)
    inat_scientific_to_taxon_id = defaultdict(set)

    gbif_taxon_id_to_vernacular = defaultdict(set)
    gbif_vernacular_to_taxon_id = defaultdict(set)
    gbif_taxon_id_to_scientific = defaultdict(set)
    gbif_scientific_to_taxon_id = defaultdict(set)


    # Build iNat dictionaries

    # row = inat_taxonomy.iloc[0]
    for i_row, row in tqdm(inat_taxonomy.iterrows(), total=len(inat_taxonomy)):

        taxon_id = row['taxonID']
        vernacular_name = row['vernacularName']
        scientific_name = row['scientificName']

        if len(vernacular_name) > 0:
            inat_taxon_id_to_vernacular[taxon_id].add(vernacular_name)
            inat_vernacular_to_taxon_id[vernacular_name].add(taxon_id)

        assert len(scientific_name) > 0
        inat_taxon_id_to_scientific[taxon_id].add(scientific_name)
        inat_scientific_to_taxon_id[scientific_name].add(taxon_id)


    # Build GBIF dictionaries

    for i_row, row in tqdm(gbif_taxonomy.iterrows(), total=len(gbif_taxonomy)):

        taxon_id = row['taxonID']

        # The canonical name is the Latin name; the "scientific name"
        # include the taxonomy name.
        #
        # http://globalnames.org/docs/glossary/

        scientific_name = row['canonicalName']

        # This only seems to happen for really esoteric species that aren't
        # likely to apply to our problems, but doing this for completeness.
        if len(scientific_name) == 0:
            scientific_name = row['scientificName']

        assert len(scientific_name) > 0
        gbif_taxon_id_to_scientific[taxon_id].add(scientific_name)
        gbif_scientific_to_taxon_id[scientific_name].add(taxon_id)

    for i_row, row in tqdm(gbif_common_mapping.iterrows(), total=len(gbif_common_mapping)):

        taxon_id = row['taxonID']

        # Don't include taxon IDs that were removed from the master table
        if taxon_id not in gbif_taxon_id_to_scientific:
            continue

        vernacular_name = row['vernacularName']

        assert len(vernacular_name) > 0
        gbif_taxon_id_to_vernacular[taxon_id].add(vernacular_name)
        gbif_vernacular_to_taxon_id[vernacular_name].add(taxon_id)


    # Save everything to file

    structures_to_serialize = [
        inat_taxonomy,
        gbif_taxonomy,
        gbif_common_mapping,
        inat_taxon_id_to_row,
        gbif_taxon_id_to_row,
        inat_taxon_id_to_vernacular,
        inat_vernacular_to_taxon_id,
        inat_taxon_id_to_scientific,
        inat_scientific_to_taxon_id,
        gbif_taxon_id_to_vernacular,
        gbif_vernacular_to_taxon_id,
        gbif_taxon_id_to_scientific,
        gbif_scientific_to_taxon_id
    ]

    print('Serializing...', end='')
    if not os.path.isfile(serialized_structures_file):
        with open(serialized_structures_file, 'wb') as p:
            pickle.dump(structures_to_serialize, p)
    print(' done')

# ...def initialize_taxonomy_lookup()


def get_scientific_name_from_row(r):
    """
    r: a dataframe that's really a row in one of our taxonomy tables
    """

    if 'canonicalName' in r and len(r['canonicalName']) > 0:
        scientific_name = r['canonicalName']
    else:
        scientific_name = r['scientificName']
    return scientific_name


def taxonomy_row_to_string(r):
    """
    r: a dataframe that's really a row in one of our taxonomy tables
    """

    if 'vernacularName' in r:
        common_string = ' (' + r['vernacularName'] + ')'
    else:
        common_string = ''
    scientific_name = get_scientific_name_from_row(r)

    return r['taxonRank'] + ' ' + scientific_name + common_string


def traverse_taxonomy(matching_rownums: Sequence[int],
                      taxon_id_to_row: Mapping[str, int],
                      taxon_id_to_vernacular: Mapping[str, Set[str]],
                      taxonomy: pd.DataFrame,
                      source_name: str,
                      query: str) -> List[Dict[str, Any]]:
    """
    Given a data frame that's a set of rows from one of our taxonomy tables,
    walks the taxonomy hierarchy from each row to put together a full taxonomy
    tree, then prunes redundant trees (e.g. if we had separate hits for a
    species and the genus that contains that species.)

    Returns a list of dicts:
    [
      {
        'source': 'inat' or 'gbif',
        'taxonomy': [(taxon_id, taxon_rank, scientific_name, [common names])]
      },
      ...
    ]
    """

    # list of dicts: {'source': source_name, 'taxonomy': match_details}
    matching_trees: List[Dict[str, Any]] = []

    # i_match = 0
    for i_match in matching_rownums:

        # list of (taxon_id, taxonRank, scientific name, [vernacular names])
        # corresponding to an exact match and its parents
        match_details = []
        current_row = taxonomy.iloc[i_match]

        # Walk taxonomy hierarchy
        while True:

            taxon_id = current_row['taxonID']
            vernacular_names = sorted(taxon_id_to_vernacular[taxon_id])  # sort for determinism, pylint: disable=line-too-long
            match_details.append((taxon_id, current_row['taxonRank'],
                                  get_scientific_name_from_row(current_row),
                                  vernacular_names))

            if np.isnan(current_row['parentNameUsageID']):
                break
            parent_taxon_id = current_row['parentNameUsageID'].astype('int64')
            if parent_taxon_id not in taxon_id_to_row:
                # This can happen because we remove questionable rows from the
                # GBIF taxonomy
                print(f'Warning: no row exists for parent_taxon_id {parent_taxon_id},' + \
                      f'child taxon_id: {taxon_id}, query: {query}')
                break
            i_parent_row = taxon_id_to_row[parent_taxon_id]
            current_row = taxonomy.iloc[i_parent_row]

            # The GBIF taxonomy contains unranked entries
            if current_row['taxonRank'] == 'unranked':
                break

        # ...while there is taxonomy left to walk

        matching_trees.append({'source': source_name,
                               'taxonomy': match_details})

    # ...for each match

    # Remove redundant matches
    b_valid_tree = [True] * len(matching_rownums)
    # i_tree_a = 0; tree_a = matching_trees[i_tree_a]
    for i_tree_a, tree_a in enumerate(matching_trees):

        tree_a_primary_taxon_id = tree_a['taxonomy'][0][0]

        # i_tree_b = 1; tree_b = matching_trees[i_tree_b]
        for i_tree_b, tree_b in enumerate(matching_trees):

            if i_tree_a == i_tree_b:
                continue

            # If tree a's primary taxon ID is inside tree b, discard tree a
            #
            # taxonomy_level_b = tree_b['taxonomy'][0]
            for taxonomy_level_b in tree_b['taxonomy']:
                if tree_a_primary_taxon_id == taxonomy_level_b[0]:
                    b_valid_tree[i_tree_a] = False
                    break

            # ...for each level in taxonomy B

        # ...for each tree (inner)

    # ...for each tree (outer)

    matching_trees = list(compress(matching_trees, b_valid_tree))
    return matching_trees

# ...def traverse_taxonomy()


def get_taxonomic_info(query: str) -> List[Dict[str, Any]]:
    """
    Main entry point: get taxonomic matches from both taxonomies for [query],
    which may be a scientific or common name.
    """
    query = query.strip().lower()
    # print("Finding taxonomy information for: {0}".format(query))

    inat_taxon_ids = set()
    if query in inat_scientific_to_taxon_id:
        inat_taxon_ids |= inat_scientific_to_taxon_id[query]
    if query in inat_vernacular_to_taxon_id:
        inat_taxon_ids |= inat_vernacular_to_taxon_id[query]

    # in GBIF, some queries hit for both common and scientific, make sure we end
    # up with unique inputs
    gbif_taxon_ids = set()
    if query in gbif_scientific_to_taxon_id:
        gbif_taxon_ids |= gbif_scientific_to_taxon_id[query]
    if query in gbif_vernacular_to_taxon_id:
        gbif_taxon_ids |= gbif_vernacular_to_taxon_id[query]

    # If the species is not found in either taxonomy, return None
    if (len(inat_taxon_ids) == 0) and (len(gbif_taxon_ids) == 0):
        return []

    # both GBIF and iNat have a 1-to-1 mapping between taxon_id and row number
    inat_row_indices = [inat_taxon_id_to_row[i] for i in inat_taxon_ids]
    gbif_row_indices = [gbif_taxon_id_to_row[i] for i in gbif_taxon_ids]

    # Walk both taxonomies
    inat_matching_trees = traverse_taxonomy(
        inat_row_indices, inat_taxon_id_to_row, inat_taxon_id_to_vernacular,
        inat_taxonomy, 'inat', query)
    gbif_matching_trees = traverse_taxonomy(
        gbif_row_indices, gbif_taxon_id_to_row, gbif_taxon_id_to_vernacular,
        gbif_taxonomy, 'gbif', query)

    return gbif_matching_trees + inat_matching_trees

# ...def get_taxonomic_info()


def print_taxonomy_matches(matches, verbose=False):
    """
    Console-friendly printing function to make nicely-indentend trees
    """

    # m = matches[0]
    for m in matches:

        source = m['source']

        # For example: [(9761484, 'species', 'anas platyrhynchos')]
        for i_taxonomy_level in range(0, len(m['taxonomy'])):
            taxonomy_level_info = m['taxonomy'][i_taxonomy_level]
            taxonomy_level = taxonomy_level_info[1]
            name = taxonomy_level_info[2]
            common = taxonomy_level_info[3]

            if i_taxonomy_level > 0:
                print('\t',end='')

            print('{} {} ({})'.format(taxonomy_level, name, common), end='')

            if i_taxonomy_level == 0:
                print(' ({})'.format(source))
            else:
                print('')

            if not verbose:
                break

        # ...for each taxonomy level

    # ...for each match

# ...def print_taxonomy_matches()


#%% Interactive drivers and debug

if False:

    #%% Initialization

    initialize_taxonomy_lookup()


    #%% Taxonomic lookup

    # query = 'lion'
    query = 'great blue heron'
    matches = get_taxonomic_info(query)
    # print(matches)

    print_taxonomy_matches(matches,verbose=True)
    
    # Print the taxonomy in the taxonomy spreadsheet format
    t = matches[1]['taxonomy']
    [(4956, 'species', 'ardea herodias', ['great blue heron']), (4950, 'genus', 'ardea', ['great herons']), (597395, 'subfamily', 'ardeinae', ['typical herons']), (4929, 'family', 'ardeidae', ['herons, egrets, and bitterns']), (67566, 'order', 'pelecaniformes', ['pelicans, herons, ibises, and allies']), (3, 'class', 'aves', ['birds']), (355675, 'subphylum', 'vertebrata', ['vertebrates']), (2, 'phylum', 'chordata', ['chordates']), (1, 'kingdom', 'animalia', ['animals']), (48460, 'stateofmatter', 'life', [])]
        
    print_taxonomy_matches(matches, verbose=True)


    #%% Directly access the taxonomy tables

    taxon_ids = gbif_vernacular_to_taxon_id['lion']
    for taxon_id in taxon_ids:
        i_row = gbif_taxon_id_to_row[taxon_id]
        print(taxonomy_row_to_string(gbif_taxonomy.iloc[i_row]))


#%% Command-line driver

def main():

    # Read command line inputs (absolute path)
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    args = parser.parse_args()
    input_file = args.input_file

    initialize_taxonomy_lookup()

    # Read the tokens from the input text file
    with open(input_file, 'r') as f:
        tokens = f.readlines()

    # Loop through each token and get scientific name
    for token in tokens:
        token = token.strip().lower()
        matches = get_taxonomic_info(token)
        print_taxonomy_matches(matches)

if __name__ == '__main__':
    main()

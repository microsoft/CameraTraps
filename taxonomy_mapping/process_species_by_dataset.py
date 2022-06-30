#
# process_species_by_dataset
#
# We generated a list of all the annotations in our universe; this script is
# used to (interactively) map them onto the GBIF and iNat taxonomies.  Don't
# try to run this script from top to bottom; it's used like a notebook, not like
# a script, since manual review steps are required.
#

#%% Imports

import os
import re
from typing import Any
import unicodedata

import pandas as pd
import numpy as np
from tqdm import tqdm

from taxonomy_mapping.species_lookup import (
    get_taxonomic_info, initialize_taxonomy_lookup, print_taxonomy_matches)
import taxonomy_mapping.retrieve_sample_image as retrieve_sample_image


# %autoreload 0
# %autoreload -species_lookup


#%% Constants

output_base = r"G:\git\agentmorrisprivate\camera-traps-taxonomy\taxonomy_archive"
xlsx_basename = 'species_by_dataset_2020_09_02_ic_ubc.xlsx'

# Input file
species_by_dataset_file = os.path.join(output_base, xlsx_basename)

# Output file after automatic remapping
output_file = species_by_dataset_file.replace('.xlsx', '.output.xlsx')

# File to which we manually copy that file and do all the manual review; this
# should never be programmatically written to
manual_review_xlsx = output_file.replace('.xlsx', '.manual.xlsx')

# The final output spreadsheet
output_xlsx = manual_review_xlsx.replace('.xlsx', '_remapped.xlsx')
output_csv = output_xlsx.replace('.xlsx', '.csv')

# HTML file generated to facilitate the identificaiton of egregious mismappings
html_output_file = os.path.join(output_base, 'mapping_previews.html')
download_images = True

master_table_file = r'C:\git\ai4edev\camera-traps-private\camera_trap_taxonomy_mapping.csv'


#%% Functions

def slugify(value: Any, allow_unicode: bool = False) -> str:
    """
    From:
    https://github.com/django/django/blob/master/django/utils/text.py

    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """

    value = str(value)
    value = unicodedata.normalize('NFKC', value)
    if not allow_unicode:
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower()).strip()
    return re.sub(r'[-\s]+', '-', value)


class TaxonomicMatch:

    def __init__(self, scientific_name, common_name, taxonomic_level, source,
                 taxonomy_string, match):
        self.scientific_name = scientific_name
        self.common_name = common_name
        self.taxonomic_level = taxonomic_level
        self.source = source
        self.taxonomy_string = taxonomy_string
        self.match = match

    def __repr__(self):
        return ('TaxonomicMatch('
            f'scientific_name={self.scientific_name}, '
            f'common_name={self.common_name}, '
            f'taxonomic_level={self.taxonomic_level}, '
            f'source={self.source}')

# Prefer iNat matches over GBIF matches
taxonomy_preference = 'inat'

def get_preferred_taxonomic_match(query: str) -> TaxonomicMatch:
    """
    Wrapper for species_lookup.py, but expressing a variety of heuristics and
    preferences that are specific to our scenario.
    """

    query = query.lower().strip().replace('_', ' ')

    # query = 'person'
    matches = get_taxonomic_info(query)

    # Do we have an iNat match?
    inat_matches = [m for m in matches if m['source'] == 'inat']
    gbif_matches = [m for m in matches if m['source'] == 'gbif']

    # print_taxonomy_matches(inat_matches, verbose=True)
    # print_taxonomy_matches(gbif_matches, verbose=True)

    scientific_name = ''
    common_name = ''
    taxonomic_level = ''
    match = ''
    source = ''
    taxonomy_string = ''

    n_inat_matches = len(inat_matches)
    n_gbif_matches = len(gbif_matches)
    
    selected_matches = None
    
    if n_inat_matches > 0 and taxonomy_preference == 'inat':
        selected_matches = 'inat'
    elif n_gbif_matches > 0:
        selected_matches = 'gbif'
        
    if selected_matches == 'inat':

        i_match = 0

        if len(inat_matches) > 1:
            # print('Warning: multiple iNat matches for {}'.format(query))

            # Prefer chordates... most of the names that aren't what we want
            # are esoteric insects, like a moth called "cheetah"
            #
            # If we can't find a chordate, just take the first match.
            #
            # i_test_match = 0
            for i_test_match, match in enumerate(inat_matches):
                found_vertebrate = False
                taxonomy = match['taxonomy']
                for taxonomy_level in taxonomy:
                    taxon_rank = taxonomy_level[1]
                    scientific_name = taxonomy_level[2]
                    if taxon_rank == 'phylum' and scientific_name == 'chordata':
                        i_match = i_test_match
                        found_vertebrate = True
                        break
                if found_vertebrate:
                    break

        match = inat_matches[i_match]['taxonomy']

        # This is (taxonID, taxonLevel, scientific, [list of common])
        lowest_level = match[0]
        taxonomic_level = lowest_level[1]
        scientific_name = lowest_level[2]
        assert len(scientific_name) > 0
        common_names = lowest_level[3]
        if len(common_names) > 1:
            # print(f'Warning: multiple iNat common names for {query}')
            # Default to returning the query
            if query in common_names:
                common_name = query
            else:
                common_name = common_names[0]
        elif len(common_names) > 0:
            common_name = common_names[0]

        # print(f'Matched iNat {query} to {scientific_name},{common_name}')
        source = 'inat'

    # ...if we had iNat matches

    # If we either prefer GBIF or didn't have iNat matches
    #
    # Code is deliberately redundant here; I'm expecting some subtleties in how
    # handle GBIF and iNat.
    elif selected_matches == 'gbif':

        i_match = 0

        if len(gbif_matches) > 1:
            # print('Warning: multiple GBIF matches for {}'.format(query))

            # Prefer chordates... most of the names that aren't what we want
            # are esoteric insects, like a moth called "cheetah"
            #
            # If we can't find a chordate, just take the first match.
            #
            # i_test_match = 0
            for i_test_match, match in enumerate(gbif_matches):
                found_vertebrate = False
                taxonomy = match['taxonomy']
                for taxonomy_level in taxonomy:
                    taxon_rank = taxonomy_level[1]
                    scientific_name = taxonomy_level[2]
                    if taxon_rank == 'phylum' and scientific_name == 'chordata':
                        i_match = i_test_match
                        found_vertebrate = True
                        break
                if found_vertebrate:
                    break

        match = gbif_matches[i_match]['taxonomy']

        # This is (taxonID, taxonLevel, scientific, [list of common])
        lowest_level = match[0]
        taxonomic_level = lowest_level[1]
        scientific_name = lowest_level[2]
        assert len(scientific_name) > 0

        common_names = lowest_level[3]
        if len(common_names) > 1:
            # print(f'Warning: multiple GBIF common names for {query}')
            # Default to returning the query
            if query in common_names:
                common_name = query
            else:
                common_name = common_names[0]
        elif len(common_names) > 0:
            common_name = common_names[0]

        source = 'gbif'

    # ...if we needed to look in the GBIF taxonomy

    taxonomy_string = str(match)

    return TaxonomicMatch(scientific_name, common_name, taxonomic_level, source,
                          taxonomy_string, match)

# ...def get_preferred_taxonomic_match()


#%% Initialization

initialize_taxonomy_lookup()


#%% Test single-query lookup

if False:
    #%%
    matches = get_taxonomic_info('equus quagga')
    print_taxonomy_matches(matches)
    #%%
    q = 'equus quagga'
    # q = "grevy's zebra"
    taxonomy_preference = 'inat'
    m = get_preferred_taxonomic_match(q)
    print(m.source)
    print(m.taxonomy_string)
    import clipboard
    clipboard.copy(m.taxonomy_string)


#%% Read the input data

df = pd.read_excel(species_by_dataset_file)


#%% Run all our taxonomic lookups

# i_row = 0; row = df.iloc[i_row]
# query = 'lion'

output_rows = []
for i_row, row in df.iterrows():

    dataset_name = row['dataset']
    query = row['species_label']

    taxonomic_match = get_preferred_taxonomic_match(query)

    def google_images_url(query: str) -> str:
        return f'https://www.google.com/search?tbm=isch&q={query}'

    scientific_url = ''
    if len(taxonomic_match.scientific_name) > 0:
        scientific_url = google_images_url(taxonomic_match.scientific_name)
    common_url = ''
    if len(taxonomic_match.common_name) > 0:
        common_url = google_images_url(taxonomic_match.common_name)
    query_url = google_images_url(query)

    output_row = {
        'dataset_name': dataset_name,
        'query': query,
        'taxonomy_level': taxonomic_match.taxonomic_level,
        'scientific_name': taxonomic_match.scientific_name,
        'common_name': taxonomic_match.common_name,
        'source': taxonomic_match.source,
        'is_typo': '',
        'setup': '',
        'notes': '',
        'non-global': '',
        'query_url': query_url,
        'scientific_url': scientific_url,
        'common_url': common_url,
        'taxonomy_string': taxonomic_match.taxonomy_string
    }
    output_rows.append(output_row)

# ...for each query

# Write to the excel file that we'll use for manual review
output_df = pd.DataFrame(data=output_rows, columns=[
    'dataset_name', 'query', 'taxonomy_level', 'scientific_name', 'common_name',
    'source', 'is_typo', 'setup', 'notes', 'non-global', 'query_url',
    'scientific_url', 'common_url', 'taxonomy_string'])
output_df.to_excel(output_file, index=None, header=True)


#%% Download preview images for everything we successfully mapped

# uncomment this to load saved output_file
# output_df = pd.read_excel(output_file, keep_default_na=False)

preview_base = os.path.join(output_base, 'preview_images')
os.makedirs(preview_base, exist_ok=True)

scientific_name_to_paths = {}

# i_row = 0; row = output_df.iloc[i_row]
for i_row, row in tqdm(output_df.iterrows(), total=len(output_df)):

    scientific_name = row.scientific_name

    assert isinstance(scientific_name, str)
    if len(scientific_name) == 0:
        continue
    if scientific_name in scientific_name_to_paths:
        continue

    image_paths = None
    preview_dir = os.path.join(preview_base, slugify(scientific_name))
    if os.path.isdir(preview_dir) and len(os.listdir(preview_dir)) > 0:
        print(f'Bypassing preview download for {preview_dir}')
        image_paths = os.listdir(preview_dir)
        image_paths = [os.path.join(preview_dir, p) for p in image_paths]
    elif download_images:
        print(f'Downloading images for {preview_dir}')
        os.makedirs(preview_dir, exist_ok=True)
        image_paths = retrieve_sample_image.download_images(
            scientific_name, output_directory=preview_dir, limit=4)
    if image_paths is not None:
        scientific_name_to_paths[scientific_name] = image_paths

# ...for each query


#%% Write HTML file with representative images to scan for obvious mis-mappings

with open(html_output_file, 'w') as f:
    f.write('<html><head></head><body>\n')

    # i_row = 0; row = output_df.iloc[i_row]
    for i_row, row in tqdm(output_df.iterrows(), total=len(output_df)):
        f.write('<p class="speciesinfo_p" style="font-weight:bold;font-size:130%">')
        common = row.common_name
        if len(common) == 0:
            common = 'no common name'
        f.write('{}: {} mapped to {} ({}) from {}</p>\n'.format(
            row.dataset_name, row.query, row.scientific_name, common,
            row.source))
        if row.scientific_name not in scientific_name_to_paths:
            f.write('<p class="content_p">no images available</p>')
        else:
            image_paths = scientific_name_to_paths[row.scientific_name]
            n_images = len(image_paths)
            image_paths = [os.path.relpath(p, output_base) for p in image_paths]
            image_width_percent = round(100 / n_images)
            f.write('<table class="image_table"><tr>\n')
            for image_path in image_paths:
                f.write('<td style="vertical-align:top;" width="{}%">'
                        '<img src="{}" style="display:block; width:100%; vertical-align:top; height:auto;">'
                        '</td>\n'.format(image_width_percent, image_path))
            f.write('</tr></table>\n')

    # ...for each row

    f.write('</body></html>\n')


#%% Look for redundancy with the master table

# Note: `master_table_file` is a CSV file that is the concatenation of the
#   manually-remapped files ("manual_remapped.xlsx"), which are the output of
#   this script run across from different groups of datasets. The concatenation
#   should be done manually. If `master_table_file` doesn't exist yet, skip this
#   code cell. Then, after going through the manual steps below, set the final
#   manually-remapped version to be the `master_table_file`.

def generate_query_id(dataset_name: str, query: str) -> str:
    return dataset_name + '|' + query

master_table = pd.read_csv(master_table_file)
master_table_dataset_queries = set()
for i_row, row in tqdm(master_table.iterrows(), total=len(master_table)):
    query_id = generate_query_id(row.dataset_name, row.query)
    master_table_dataset_queries.add(query_id)

for i_row, row in tqdm(output_df.iterrows(), total=len(output_df)):
    query_id = generate_query_id(row.dataset_name, row.query)
    if query_id in master_table_dataset_queries:
        print('Warning: query {} available in master table'.format(query_id))


#%% Manual review

# Copy the spreadsheet to another file; you're about to do a ton of manual
# review work and you don't want that programmatically overwrriten.
#
# See manual_review_xlsx above


#%% Read back the results of the manual review process

df = pd.read_excel(manual_review_xlsx)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.applymap(lambda s: s.lower().strip() if isinstance(s, str) else s)
    df = df.replace(np.nan, '', regex=True)
    return df

df = clean_df(df)


#%% Look for manual mapping errors

# Manually inspect df for typos in 'dataset_names' and 'taxonomy_level' columns
print('dataset names')
print(df['dataset_name'].unique())
print()
print('taxonomy levels:')
print(df['taxonomy_level'].unique())

# Identify rows where:
#
# - 'taxonomy_level' does not match level of 1st element in 'taxonomy_string'
# - 'scientific_name' does not match name of 1st element in 'taxonomy_string'
#
# ...both of which typically represent manual mapping errors.

# i_row = 0; row = df.iloc[i_row]
for i_row, row in df.iterrows():
    
    taxa_ancestry = row['taxonomy_string']
    
    # I'm not sure why both of these checks are necessary, best guess is that 
    # the Excel parser was reading blanks as na on one OS/Excel version and as ''
    # on another.
    if (isinstance(taxa_ancestry,str) and taxa_ancestry == '') or (pd.isna(taxa_ancestry)):
        continue

    # The taxonomy_string column is a .json-formatted string; expand it into
    # an object via eval()
    taxa_ancestry = eval(taxa_ancestry)  # pylint: disable=eval-used
    taxon_id, taxon_level, taxon_name, _ = taxa_ancestry[0]

    if row['taxonomy_level'] != taxon_level:
        print(f'row: {i_row}, {row["dataset_name"]}, {row["query"]}')
        print(f'- taxonomy_level column: {row["taxonomy_level"]}, '
              f'level from taxonomy_string: {taxon_level}')
        print()

    if row['scientific_name'] != taxon_name:
        print(f'row: {i_row}, {row["dataset_name"]}, {row["query"]}')
        print(f'- scientific_name column: {row["scientific_name"]}, '
              f'name from taxonomy_string: {taxon_name}')


#%% Find scientific names that were added manually, and match them to taxonomies

# i_row = 0; row = df.iloc[i_row]
for i_row, row in df.iterrows():

    assert row.source in ['manual', 'gbif', 'inat', '']

    if row.source == '':
        assert row.scientific_name == ''

    if False:
        if row.source == 'manual' and row.taxonomy_string != '':
            print('Warning: manual row with taxonomy string: {}.{}'.format(
                row.dataset_name, row.query))

    if row.source != '' and len(row.scientific_name) == 0:
        print('Unsourced row with no scientific name: {}.{}'.format(
            row.dataset_name, row.query))

    if row.source == 'manual':
        
        common_name = row.common_name
        if common_name == '':
            print('Warning: manual row with no common name: {}.{}'.format(
                row.dataset_name, row.query))

        scientific_name = row.scientific_name
        taxonomic_match = get_preferred_taxonomic_match(scientific_name)
        if taxonomic_match.source == '':
            print('Failed to match {}:{} ({})'.format(
                row.dataset_name, row.query, scientific_name))
        else:
            row.taxonomy_string = taxonomic_match.taxonomy_string
        if False:
            print('Matched {}.{} to {} ({}) via {}'.format(
                row.dataset_name, scientific_name,
                taxonomic_match.scientific_name,
                taxonomic_match.common_name,
                taxonomic_match.source))

# ...for each query


#%% Write out final version

df.to_excel(output_xlsx, index=None, header=True)
df.to_csv(output_csv, index=None, header=True)

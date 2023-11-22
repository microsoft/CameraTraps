#
# get_lila_category_list.py
#
# Generates a .json-formatted dictionary mapping each LILA dataset to all categories
# that exist for that dataset, with counts for the number of occurrences of each category 
# (the number of *annotations* for each category, not the number of *images*).
#
# Also loads the taxonomy mapping file, to include scientific names for each category.
#
# get_lila_category_counts counts the number of *images* for each category in each dataset.
#

#%% Constants and imports

import json
import os

from data_management.lila.lila_common import read_lila_metadata,\
    get_json_file_for_dataset, read_lila_taxonomy_mapping

# array to fill for output
category_list = []

# We'll write images, metadata downloads, and temporary files here
lila_local_base = os.path.expanduser('~/lila')

output_dir = os.path.join(lila_local_base,'lila_categories_list')
os.makedirs(output_dir,exist_ok=True)

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)

output_file = os.path.join(output_dir,'lila_dataset_to_categories.json')

# Created by get_lila_category_list.py... contains counts for each category
category_list_dir = os.path.join(lila_local_base,'lila_categories_list')
lila_dataset_to_categories_file = os.path.join(category_list_dir,'lila_dataset_to_categories.json')

assert os.path.isfile(lila_dataset_to_categories_file)


#%% Load category and taxonomy files

with open(lila_dataset_to_categories_file,'r') as f:
    lila_dataset_to_categories = json.load(f)

taxonomy_df = read_lila_taxonomy_mapping(metadata_dir)


#%% Map dataset names and category names to scientific names

ds_query_to_scientific_name = {}

unmapped_queries = set()

# i_row = 1; row = taxonomy_df.iloc[i_row]; row
for i_row,row in taxonomy_df.iterrows():
    
    ds_query = row['dataset_name'] + ':' + row['query']
    ds_query = ds_query.lower()
    
    if not isinstance(row['scientific_name'],str):
        unmapped_queries.add(ds_query)
        ds_query_to_scientific_name[ds_query] = 'unmapped'
        continue
        
    ds_query_to_scientific_name[ds_query] = row['scientific_name']
    
    
#%% Download and parse the metadata file

metadata_table = read_lila_metadata(metadata_dir)


#%% Download and extract metadata for the datasets we're interested in

for ds_name in metadata_table.keys():    
    metadata_table[ds_name]['json_filename'] = get_json_file_for_dataset(ds_name=ds_name,
                                                                         metadata_dir=metadata_dir,
                                                                         metadata_table=metadata_table)
    
#%% Get category names and counts for each dataset

from collections import defaultdict

dataset_to_categories = {}

# ds_name = 'NACTI'
for ds_name in metadata_table.keys():
    
    print('Finding categories in {}'.format(ds_name))
    
    json_filename = metadata_table[ds_name]['json_filename']
    sas_url = metadata_table[ds_name]['sas_url']
    
    base_url = sas_url.split('?')[0]    
    assert not base_url.endswith('/')
    
    # Open the metadata file    
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    # Collect list of categories and mappings to category name
    categories = data['categories']
    
    category_id_to_count = defaultdict(int)
    annotations = data['annotations']    
    
    # ann = annotations[0]
    for ann in annotations:
        category_id_to_count[ann['category_id']] = category_id_to_count[ann['category_id']] + 1
    
    # c = categories[0]
    for c in categories:
       count = category_id_to_count[c['id']] 
       if 'count' in c:
           assert 'bbox' in ds_name or c['count'] == count       
       c['count'] = count
       
       # Don't do taxonomy mapping for bbox data sets, which are sometimes just binary and are
       # always redundant with the class-level data sets.
       if 'bbox' in ds_name:
           c['scientific_name_from_taxonomy_mapping'] = None
       else:
           taxonomy_query_string = ds_name.lower().strip() + ':' + c['name'].lower()
           if taxonomy_query_string not in ds_query_to_scientific_name:
               print('No match for query string {}'.format(taxonomy_query_string))
               # As of right now, this is the only quirky case
               assert '#ref!' in taxonomy_query_string and 'wcs' in ds_name.lower()
               c['scientific_name_from_taxonomy_mapping'] = None
           else:
               sn = ds_query_to_scientific_name[taxonomy_query_string]
               assert sn is not None and len(sn) > 0
               c['scientific_name_from_taxonomy_mapping'] = sn
    
    dataset_to_categories[ds_name] = categories

# ...for each dataset


#%% Save dict

with open(output_file, 'w') as f:
    json.dump(dataset_to_categories,f,indent=2)


#%% Print the results

# ds_name = list(dataset_to_categories.keys())[0]
for ds_name in dataset_to_categories:
    
    print('\n** Category counts for {} **\n'.format(ds_name))
    
    categories = dataset_to_categories[ds_name]
    
    for c in categories:
        print('{} ({}): {}'.format(c['name'],c['scientific_name_from_taxonomy_mapping'],c['count']))
        
# ...for each dataset

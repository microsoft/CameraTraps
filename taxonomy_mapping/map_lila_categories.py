#
# Using the taxonomy .csv file, map all LILA datasets to the standard taxonomy 
#

#%% Constants and imports

import json
import os

import pandas as pd

# Created by get_lila_category_list.py... contains counts for each category
lila_dataset_to_categories_file = r"G:\temp\lila\lila_categories_list\lila_dataset_to_categories.json"
lila_taxonomy_file = r"G:\temp\lila\lila-taxonomy-mapping_release.22.07.03.1608.csv"

assert os.path.isfile(lila_dataset_to_categories_file)
assert os.path.isfile(lila_taxonomy_file)
    

#%% Load category and taxonomy files

with open(lila_dataset_to_categories_file,'r') as f:
    lila_dataset_to_categories = json.load(f)

taxonomy_df = pd.read_csv(lila_taxonomy_file)


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
    
    
#%% For each dataset, make sure we can map every category to the taxonomy

# dataset_name = list(lila_dataset_to_categories.keys())[0]
for _dataset_name in lila_dataset_to_categories.keys():
    
    if '_bbox' in _dataset_name:
        dataset_name = _dataset_name.replace('_bbox','')
    else:
        dataset_name = _dataset_name
    
    categories = lila_dataset_to_categories[dataset_name]
    
    # c = categories[0]
    for c in categories:
        ds_query = dataset_name + ':' + c['name']
        ds_query = ds_query.lower()
        
        if ds_query not in ds_query_to_scientific_name:
            print('Could not find mapping for {}'.format(ds_query))            
        else:
            scientific_name = ds_query_to_scientific_name[ds_query]


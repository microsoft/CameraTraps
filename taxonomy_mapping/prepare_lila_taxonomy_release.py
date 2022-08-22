#
# prepare_lila_taxonomy_release.py
#
# Given the private intermediate taxonomy mapping, prepare the public (release)
# taxonomy mapping file.
#

#%% Imports and constants

import os
import json
import pandas as pd

lila_taxonomy_file = os.path.expanduser('~/git/agentmorrisprivate/lila-taxonomy/lila-taxonomy-mapping.csv')
release_taxonomy_file = os.path.expanduser('~/lila/lila-taxonomy-mapping_release.22.08.22.0000.csv')

# Created by get_lila_category_list.py... contains counts for each category
lila_dataset_to_categories_file = os.path.expanduser('~/lila/lila_categories_list/lila_dataset_to_categories.json')

assert os.path.isfile(lila_dataset_to_categories_file)
assert os.path.isfile(lila_taxonomy_file)


#%% Find out which categories are actually used

df = pd.read_csv(lila_taxonomy_file)

with open(lila_dataset_to_categories_file,'r') as f:
    lila_dataset_to_categories = json.load(f)

used_category_mappings = []

# dataset_name = datasets_to_map[0]
for dataset_name in lila_dataset_to_categories.keys():
    
    ds_categories = lila_dataset_to_categories[dataset_name]
    for category in ds_categories:
        category_name = category['name'].lower()
        assert ':' not in category_name
        mapping_name = dataset_name + ':' + category_name
        used_category_mappings.append(mapping_name)

df['used'] = False

# i_row = 0; row = df.iloc[i_row]; row
for i_row,row in df.iterrows():
    ds_name = row['dataset_name']
    query = row['query']
    mapping_name = ds_name + ':' + query
    if mapping_name in used_category_mappings:
        df.loc[i_row,'used'] = True
    else:
        print('Dropping unused mapping {}'.format(mapping_name))

df = df[df.used]
df = df.drop('used',1)


#%% Generate the final output file

assert not os.path.isfile(release_taxonomy_file)

known_levels = ['stateofmatter',
                     'kingdom',
                     'phylum','subphylum',
                     'superclass','class','subclass','infraclass',
                     'superorder','order','parvorder','suborder','infraorder',
                     'zoosection',
                     'superfamily','family','subfamily','tribe',
                     'genus',
                     'species','subspecies','variety']

levels_to_include = ['kingdom',
                     'phylum','subphylum',
                     'superclass','class','subclass','infraclass',
                     'superorder','order','suborder','infraorder',
                     'superfamily','family','subfamily','tribe',
                     'genus',
                     'species','subspecies','variety']

levels_to_exclude = ['stateofmatter','zoosection','parvorder']

for s in levels_to_exclude:
    assert s not in levels_to_include
    
levels_used = set()

# i_row = 0; row = df.iloc[i_row]; row
for i_row,row in df.iterrows():
    
    if not isinstance(row['scientific_name'],str):
        assert not isinstance(row['taxonomy_string'],str)
        continue
    
    taxonomic_match = eval(row['taxonomy_string'])
    
    # match_at_level = taxonomic_match[0]
    for match_at_level in taxonomic_match:
        assert len(match_at_level) == 4
        levels_used.add(match_at_level[1])
        
levels_used = [s for s in levels_used if isinstance(s,str)]

for s in levels_used:
    assert s in levels_to_exclude or s in levels_to_include, 'Unrecognized level {}'.format(s)

for s in levels_to_include:
    assert s in levels_used
    
for s in levels_to_include:
    df[s] = ''
    
# i_row = 0; row = df.iloc[i_row]; row
for i_row,row in df.iterrows():
    
    if not isinstance(row['scientific_name'],str):
        assert not isinstance(row['taxonomy_string'],str)
        continue
    
    # E.g.: (43117, 'genus', 'lepus', ['hares and jackrabbits']
    taxonomic_match = eval(row['taxonomy_string'])
    
    for match_at_level in taxonomic_match:
        level = match_at_level[1]
        if level in levels_to_include:
            df.loc[i_row,level] = match_at_level[2]

df = df.drop('source',1)
df.to_csv(release_taxonomy_file,header=True,index=False)

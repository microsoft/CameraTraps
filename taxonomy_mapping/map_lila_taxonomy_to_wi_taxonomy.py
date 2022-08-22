#
# map_lila_taxonomy_to_wi_taxonomy.py
#
# Loads the LILA category mapping (in which taxonomy information comes from an iNat taxonomy snapshot)
# and tries to map each class to the Wildlife Insights taxonomy.
#

#%% Constants and imports

import json
import os

from tqdm import tqdm

from data_management.lila.lila_common import read_lila_taxonomy_mapping, \
    read_wildlife_insights_taxonomy_mapping

lila_local_base = os.path.expanduser('~/lila')

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)

# Created by get_lila_category_list.py... contains counts for each category
category_list_dir = os.path.join(lila_local_base,'lila_categories_list')
lila_dataset_to_categories_file = os.path.join(category_list_dir,'lila_dataset_to_categories.json')

lila_to_wi_supplementary_mapping_file = os.path.expanduser('~/git/CameraTraps/taxonomy_mapping/lila_to_wi_supplementary_mapping_file.csv')

assert os.path.isfile(lila_dataset_to_categories_file)
    

#%% Load category and taxonomy files

with open(lila_dataset_to_categories_file,'r') as f:
    lila_dataset_to_categories = json.load(f)

lila_taxonomy_df = read_lila_taxonomy_mapping(metadata_dir)

wi_taxonomy_df = read_wildlife_insights_taxonomy_mapping(metadata_dir)


#%% Pull everything out of pandas

lila_taxonomy = lila_taxonomy_df.to_dict('records')
wi_taxonomy = wi_taxonomy_df.to_dict('records')


#%% Cache WI taxonomy lookups

import numpy as np

def is_empty_wi_item(v):
    if isinstance(v,str):
        assert len(v) > 0
        return False
    else:
        assert isinstance(v,float) and np.isnan(v)
        return True
    
def taxonomy_items_equal(a,b):
    if isinstance(a,str) and (not isinstance(b,str)):
        return False
    if isinstance(b,str) and (not isinstance(a,str)):
        return False
    if (not isinstance(a,str)) or (not isinstance(b,str)):
        assert isinstance(a,float) and isinstance(b,float)
        return True
    return a == b
    
for taxon in wi_taxonomy:
    taxon['taxon_name'] = None

wi_taxon_name_to_taxon = {}

# This is just a handy lookup table that we'll use to debug mismatches
wi_common_name_to_taxon = {}

blank_taxon_name = 'blank'
blank_taxon = None

animal_taxon_name = 'animal'
animal_taxon = None

unknown_taxon_name = 'unknown'
unknown_taxon = None

ignore_taxa = set(['No CV Result','CV Needed','CV Failed'])

# taxon = wi_taxonomy[1000]; print(taxon)
for taxon in tqdm(wi_taxonomy):
    
    taxon_name = None
    
    assert taxon['taxonomyType'] == 'object' or taxon['taxonomyType'] == 'biological'

    if taxon['commonNameEnglish'] in ignore_taxa:
        continue

    if isinstance(taxon['commonNameEnglish'],str):
        
        wi_common_name_to_taxon[taxon['commonNameEnglish'].strip().lower()] = taxon

        special_taxon = False
        
        if taxon['commonNameEnglish'].strip().lower() == blank_taxon_name:
            blank_taxon = taxon                        
            special_taxon = True
        
        elif taxon['commonNameEnglish'].strip().lower() == animal_taxon_name:
            animal_taxon = taxon
            special_taxon = True
        
        elif taxon['commonNameEnglish'].strip().lower() == unknown_taxon_name:
            unknown_taxon = taxon
            special_taxon = True
        
        if special_taxon:            
            taxon_name = taxon['commonNameEnglish'].strip().lower()
            taxon['taxon_name'] = taxon_name
            wi_taxon_name_to_taxon[taxon_name] = taxon
            continue
        
    # Do we have a species name?
    if not is_empty_wi_item(taxon['species']):
        
        # If 'species' is populated, 'genus' should always be populated; one item currently breaks
        # this rule.
        if is_empty_wi_item(taxon['genus']):
            assert taxon['species'] == 'Mongoose Species' and taxon['family'] == 'Herpestidae'
            taxon['species'] = np.nan
            taxon['commonNameEnglish'] = 'Mongooses'
            taxon_name = taxon['family'].strip().lower()
        else:
            taxon_name = (taxon['genus'].strip() + ' ' + taxon['species'].strip()).strip().lower()
            assert not is_empty_wi_item(taxon['class']) and \
                not is_empty_wi_item(taxon['order']) and \
                not is_empty_wi_item(taxon['family'])
    
    elif not is_empty_wi_item(taxon['genus']):
        
        assert not is_empty_wi_item(taxon['class']) and \
            not is_empty_wi_item(taxon['order']) and \
            not is_empty_wi_item(taxon['family'])
        taxon_name = taxon['genus'].strip().lower()
        
    elif not is_empty_wi_item(taxon['family']):
        
        assert not is_empty_wi_item(taxon['class']) and \
            not is_empty_wi_item(taxon['order'])
        taxon_name = taxon['family'].strip().lower()
                    
    elif not is_empty_wi_item(taxon['order']):
        
        assert not is_empty_wi_item(taxon['class'])
        taxon_name = taxon['order'].strip().lower()
        
    elif not is_empty_wi_item(taxon['class']):
        
        taxon_name = taxon['class'].strip().lower()
    
    if taxon_name is not None:
        assert taxon['taxonomyType'] == 'biological'
    else:
        assert taxon['taxonomyType'] == 'object'
        taxon_name = taxon['commonNameEnglish'].strip().lower()        
    
    if taxon_name in wi_taxon_name_to_taxon:
        previous_taxon = wi_taxon_name_to_taxon[taxon_name]
        for level in ['class','order','family','genus','species']:
            assert taxonomy_items_equal(previous_taxon[level],taxon[level])
        
    taxon['taxon_name'] = taxon_name    
    wi_taxon_name_to_taxon[taxon_name] = taxon
    
# ...for each taxon

assert unknown_taxon is not None    
assert animal_taxon is not None
assert blank_taxon is not None


# object_name_to_taxon.keys()
#
# dict_keys(['dirt bike', 'motorcycle', 'truck', 'atv', 'vehicle', 'official vehicle',
#            'setup_pickup', 'measurement scale', 'timelapse', 'snowmobile', 'misfire',
#            'trash', 'snow', 'fire', 'water craft'])


#%% Read supplementary mappings

with open(lila_to_wi_supplementary_mapping_file,'r') as f:
    lines = f.readlines()

supplementary_lila_query_to_wi_query = {}

for line in lines:
    # Each line is [lila query],[WI taxon name],[notes]
    tokens = line.strip().split(',')
    assert len(tokens) == 3
    lila_query = tokens[0].strip().lower()
    wi_taxon_name = tokens[1].strip().lower()
    assert wi_taxon_name in wi_taxon_name_to_taxon
    supplementary_lila_query_to_wi_query[lila_query] = wi_taxon_name
    
    
#%% Map LILA categories to WI categories

mismatches = set()
mismatches_with_common_mappings = set()
supplementary_mappings = set()

all_searches = set()

# Must be ordered from kingdom --> species
lila_taxonomy_levels = ['kingdom','phylum','subphylum','superclass','class','subclass',
                        'infraclass','superorder','order','suborder','infraorder',
                        'superfamily','family','subfamily','tribe','genus','species']

unknown_queries = set(['unidentifiable','other','unidentified','unknown','unclassifiable'])
blank_queries = set(['empty'])
animal_queries = set(['animalia'])

# TODO:
# ['subspecies','variety']

query_to_wi_taxon = {}

# i_taxon = 0; taxon = lila_taxonomy[i_taxon]; print(taxon)
for i_taxon,lila_taxon in enumerate(lila_taxonomy):
    
    query = None

    # Go from kingdom --> species, choosing the lowest-level description as the query    
    for level in lila_taxonomy_levels:
        if isinstance(lila_taxon[level],str):
            query = lila_taxon[level]        
            all_searches.add(query)
        
    if query is None:
        # E.g., 'car'
        query = lila_taxon['query']

    wi_taxon = None
    
    if query in unknown_queries:    
        
        wi_taxon = unknown_taxon
    
    elif query in blank_queries:
        
        wi_taxon = blank_taxon
        
    elif query in animal_queries:
        
        wi_taxon = animal_taxon
    
    elif query in wi_taxon_name_to_taxon:
        
        wi_taxon = wi_taxon_name_to_taxon[query]
        
    elif query in supplementary_lila_query_to_wi_query:
        
        wi_taxon = wi_taxon_name_to_taxon[supplementary_lila_query_to_wi_query[query]]
        supplementary_mappings.add(query)
        # print('Made a supplementary mapping from {} to {}'.format(query,wi_taxon['taxon_name']))
        
    else:
        
        # print('No match for {}'.format(query))
        lila_common_name = lila_taxon['common_name']
        
        if lila_common_name in wi_common_name_to_taxon:
            wi_taxon = wi_common_name_to_taxon[lila_common_name]            
            wi_common_name = wi_taxon['commonNameEnglish']
            wi_taxon_name = wi_taxon['taxon_name']
            if False:
                print('LILA common name {} maps to WI taxon {} ({})'.format(lila_common_name,
                                                                            wi_taxon_name,
                                                                            wi_common_name))
            mismatches_with_common_mappings.add(query)
        
        else:
            
            mismatches.add(query)
            
    query_to_wi_taxon[query] = wi_taxon
    
# ...for each LILA taxon

print('Of {} entities, there are {} mismatches ({} mapped by common name) ({} mapped by supplementary mapping file)'.format(
    len(all_searches),len(mismatches),len(mismatches_with_common_mappings),len(supplementary_mappings)))

assert len(mismatches) == 0


#%% Manual mapping

if not os.path.isfile(lila_to_wi_supplementary_mapping_file):
    print('Creating mapping file {}'.format(lila_to_wi_supplementary_mapping_file))
    with open(lila_to_wi_supplementary_mapping_file,'w') as f:
        for query in mismatches:
            f.write(query + ',' + '\n')
else:
    print('{} exists, not re-writing'.format(lila_to_wi_supplementary_mapping_file)) 


#%% Pr
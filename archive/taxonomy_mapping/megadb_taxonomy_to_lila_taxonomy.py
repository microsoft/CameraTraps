# 
# Takes the megadb taxonomy mapping, extracts the rows that are relevant to
# LILA, and does some cleanup.
#

#%% Constants and imports

import os
import pandas as pd
import json

# This is a partially-completed taxonomy file that was created from a different set of
# scripts, but covers *most* of LILA as of June 2022
input_taxonomy_file = r"G:\git\agentmorrisprivate\lila-taxonomy\lila-taxonomy-mapping-input.csv"

# Created by get_lila_category_list.py
input_lila_category_list_file = r"G:\temp\lila\lila_categories_list\lila_dataset_to_categories.json"

output_taxonomy_file = r"G:\git\agentmorrisprivate\lila-taxonomy\lila-taxonomy-mapping.csv"


#%% Read the input files

input_taxonomy_df = pd.read_csv(input_taxonomy_file)

# Get everything out of pandas
input_taxonomy_rows = input_taxonomy_df.to_dict('records')

with open(input_lila_category_list_file,'r') as f:
    input_lila_categories = json.load(f)


#%% Find all unique dataset names in the input list, compare them with data names from LILA

input_taxonomy_datasets = set()

# d = input_taxonomy_rows[0]
for d in input_taxonomy_rows:
    input_taxonomy_datasets.add(d['dataset_name'])

lila_datasets = set()

for dataset_name in input_lila_categories.keys():
    # The script that generates this dictionary creates a separate entry for bounding box
    # metadata files, but those don't represent new dataset names
    lila_datasets.add(dataset_name.replace('_bbox',''))
    

#%% Map input columns to output datasets

input_taxonomy_to_lila_dataset_mapping = \
    {'caltech':'Caltech Camera Traps',
     'ena24':'ENA24',
     'idfg':'Idaho Camera Traps',
     'idfg_swwlf_2019':'Idaho Camera Traps',
     'islandconservation_190705':'Island Conservation Camera Traps',
     'islandconservation_200529':'Island Conservation Camera Traps',
     'islandconservation_200529_private':'Island Conservation Camera Traps',
     'nacti':'NACTI',
     'nacti_private':'NACTI',
     'snapshot_camdeboo':'Snapshot Camdeboo',
     'snapshot_enonkishu':'Snapshot Enonkishu',
     'snapshot_karoo':'Snapshot Karoo',
     'snapshot_kgalagadi':'Snapshot Kgalagadi',
     'snapshot_kruger':'Snapshot Kruger',
     'snapshot_mountain_zebra':'Snapshot Mountain Zebra',
     'snapshot_safari_private':None,
     'snapshotserengeti':'Snapshot Serengeti',
     'snapshotserengeti_private':'Snapshot Serengeti',
     'wcs':'WCS Camera Traps',
     'wcs_private':'WCS Camera Traps'}

# Make sure all of those datasets actually correspond to datasets on LILA    
mapped_lila_datasets = set()
unmapped_lila_datasets = set()

for c in input_taxonomy_to_lila_dataset_mapping.keys():
    ds = input_taxonomy_to_lila_dataset_mapping[c]
    if ds is not None:
        assert ds in lila_datasets
        mapped_lila_datasets.add(ds)
        
for s in lila_datasets:
    if s not in mapped_lila_datasets:
        print('Warning: no mappings for dataset {}'.format(s))
        unmapped_lila_datasets.add(s)
    

#%% Re-write the input taxonomy file to refer to LILA datasets

# Map the string datasetname:token to a taxonomic tree json
taxonomy_mappings = {}

n_replacements = 0

# mapping = input_taxonomy_rows[0]
for mapping in input_taxonomy_rows:
    
    input_ds_name = mapping['dataset_name']
    
    if input_taxonomy_to_lila_dataset_mapping[input_ds_name] is None:
        assert 'private' in input_ds_name
        continue
    
    if input_ds_name not in input_taxonomy_to_lila_dataset_mapping:        
        assert input_ds_name in unmapped_lila_datasets
        continue
    
    output_ds_name = input_taxonomy_to_lila_dataset_mapping[input_ds_name]
    
    query = mapping['query']
    assert ':' not in query
    
    mapping_string = output_ds_name + ':' + query
    taxonomy_string = mapping['taxonomy_string']
    source = mapping['source']    
    
    source_priorities = {'manual':0,'inat':1,'gbif':2}
    
    # Make sure that all occurrences of this mapping_string give us the same output
    if mapping_string in taxonomy_mappings:
        # assert taxonomy_string == taxonomy_mappings[mapping_string]
        previous_taxonomy_string = taxonomy_mappings[mapping_string]['taxonomy_string']
        if taxonomy_string != previous_taxonomy_string:
            previous_source = taxonomy_mappings[mapping_string]['source']
            if source == previous_source:
                if len(taxonomy_string) > len(previous_taxonomy_string):
                    print('For mapping {}, replacing {}\n{}\n\nwith\n\n{}\n{}\n'.format(
                        mapping_string,previous_source,previous_taxonomy_string,
                        source,taxonomy_string))
                    taxonomy_mappings[mapping_string] = mapping
                    n_replacements += 1
            elif source_priorities[source] < source_priorities[previous_source]:
                print('For mapping {}, replacing {}\n{}\n\nwith\n\n{}\n{}\n'.format(
                    mapping_string,previous_source,previous_taxonomy_string,
                    source,taxonomy_string))
                taxonomy_mappings[mapping_string] = mapping
                n_replacements += 1
    else:
        taxonomy_mappings[mapping_string] = mapping
    
print('Made {} replacements'.format(n_replacements))


#%% Re-write the input file in the target format

assert not os.path.isfile(output_taxonomy_file), 'You don\'t really want to overwrite the output file'
output_entries = []

# mapping_string = list(taxonomy_mappings.keys())[0]
for mapping_string in taxonomy_mappings.keys():
    
    tokens = mapping_string.split(':')
    assert len(tokens) == 2
    dataset_name = tokens[0]
    query = tokens[1]
    output_mapping = {}
    output_mapping['dataset_name'] = dataset_name
    output_mapping['query'] = query
    
    mapping = taxonomy_mappings[mapping_string]
    
    assert query == mapping['query']
    for fn in ['taxonomy_level','scientific_name','common_name','source','taxonomy_string']:
        output_mapping[fn] = mapping[fn]
        
    output_entries.append(output_mapping)
    
df = pd.DataFrame(output_entries)
cols = df.columns.tolist()
cols = ['dataset_name','query','source','taxonomy_level','scientific_name','common_name','taxonomy_string']
df = df[cols]
df.to_csv(output_taxonomy_file)

print('Wrote updated table to {}'.format(output_taxonomy_file))


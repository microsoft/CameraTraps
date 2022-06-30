#
# Does some consistency-checking on the LILA taxonomy file, and generates
# an HTML preview page that we can use to determine whether the mappings
# make sense.
#

#%% Imports and constants

import os
import pandas as pd

lila_taxonomy_file = r"G:\git\agentmorrisprivate\lila-taxonomy\lila-taxonomy-mapping.csv"
# lila_taxonomy_file = r"G:\temp\lila\lila_additions_2022.06.29.csv"

preview_base = r'g:\temp\lila_taxonomy_preview'
os.makedirs(preview_base,exist_ok=True)
html_output_file = os.path.join(preview_base,'index.html')


#%% Read the output file back

df = pd.read_csv(lila_taxonomy_file)


#%% List null mappings

#
# These should all be things like "unidentified" and "fire"
#

# i_row = 0; row = df.iloc[i_row]
for i_row,row in df.iterrows():
    if (not isinstance(row['taxonomy_string'],str)) or (len(row['taxonomy_string']) == 0):
        print('No mapping for {}:{}'.format(row['dataset_name'],row['query']))


#%% List mappings without common names

for i_row,row in df.iterrows():
    cn = row['common_name']
    sn = row['scientific_name']
    ts = row['taxonomy_string']
    if (isinstance(ts,str)) and (len(ts) >= 0):
        if (not isinstance(cn,str)) or (len(cn) == 0):
            print('No mapping for {}:{}:{}'.format(row['dataset_name'],row['query'],row['scientific_name']))


#%% List mappings that map to different things in different data sets

import numpy as np
def isnan(x):
    if not isinstance(x,float):
        return False
    return np.isnan(x)

from collections import defaultdict
query_to_rows = defaultdict(list)

queries_with_multiple_mappings = set()

n_suppressed = 0

suppress_multiple_matches = [
    ['porcupine','Snapshot Camdeboo','Idaho Camera Traps'],
    ['porcupine','Snapshot Enonkishu','Idaho Camera Traps'],
    ['porcupine','Snapshot Karoo','Idaho Camera Traps'],
    ['porcupine','Snapshot Kgalagadi','Idaho Camera Traps'],
    ['porcupine','Snapshot Kruger','Idaho Camera Traps'],
    ['porcupine','Snapshot Mountain Zebra','Idaho Camera Traps'],
    ['porcupine','Snapshot Serengeti','Idaho Camera Traps'],
    ['fox','Caltech Camera Traps','Channel Islands Camera Traps'],
    ['fox','Idaho Camera Traps','Channel Islands Camera Traps'],
    ['pangolin','Snapshot Serengeti','SWG Camera Traps']
]

for i_row,row in df.iterrows():
    
    query = row['query']
    taxonomy_string = row['taxonomy_string']
    
    for previous_i_row in query_to_rows[query]:
        
        previous_row = df.iloc[previous_i_row]
        assert previous_row['query'] == query
        query_match = False
        if isnan(row['taxonomy_string']):
            query_match = isnan(previous_row['taxonomy_string'])
        elif isnan(previous_row['taxonomy_string']):
            query_match = isnan(row['taxonomy_string'])
        else:
            query_match = previous_row['taxonomy_string'][0:10] == taxonomy_string[0:10]
        
        if not query_match:
            
            suppress = False
            
            # x = suppress_multiple_matches[-1]
            for x in suppress_multiple_matches:
                if x[0] == query and \
                    ( \
                    (x[1] == row['dataset_name'] and x[2] == previous_row['dataset_name']) \
                    or \
                    (x[2] == row['dataset_name'] and x[1] == previous_row['dataset_name']) \
                    ):
                    suppress = True
                    n_suppressed += 1
                    break
                
            if not suppress:
                print('Query {} in {} and {}:\n\n{}\n\n{}\n'.format(
                    query, row['dataset_name'], previous_row['dataset_name'],
                    taxonomy_string, previous_row['taxonomy_string']))
                
            queries_with_multiple_mappings.add(query)
                
    # ...for each row where we saw this query
    
    query_to_rows[query].append(i_row)
    
# ...for each row

print('Found {} queries with multiple mappings ({} occurrences suppressed)'.format(
    len(queries_with_multiple_mappings),n_suppressed))


#%% Verify that nothing "unidentified" maps to a species or subspecies

allowable_unknown_species = [
    'unknown_tayra' # AFAIK this is a unique species, I'm not sure what's implied here
]

unk_queries = ['skunk']
for i_row,row in df.iterrows():

    query = row['query']
    level = row['taxonomy_level']

    if not isinstance(level,str):
        assert not isinstance(row['taxonomy_string'],str)
        continue

    if ( \
        'unidentified' in query or \
        ('unk' in query and ('skunk' not in query and 'chipmunk' not in query))\
        ) \
        and \
        ('species' in level):
        
        if query not in allowable_unknown_species:
            
            print('Warning: query {}:{} maps to {} {}'.format(
                row['dataset_name'],
                row['query'],
                row['taxonomy_level'],
                row['scientific_name']
                ))


#%% Download sample images for all scientific names

import os
from taxonomy_mapping import retrieve_sample_image
scientific_name_to_paths = {}
image_base = os.path.join(preview_base,'images')
images_per_query = 15
min_valid_images_per_query = 3
min_valid_image_size = 3000

# i_row = 0; row = df.iloc[i_row]
for i_row,row in df.iterrows():
    
    s = row['scientific_name']
    
    if not isinstance(s,str):
        continue
    
    query = s.replace(' ','+')
    query_folder = os.path.join(image_base,query)
    if os.path.isdir(query_folder):
        image_files = os.listdir(query_folder)
        image_fullpaths = [os.path.join(query_folder,fn) for fn in image_files]
        sizes = [os.path.getsize(p) for p in image_fullpaths]
        sizes_above_threshold = [x for x in sizes if x > min_valid_image_size]
        if len(sizes_above_threshold) > min_valid_images_per_query:
            print('Skipping query {}, already have {} images'.format(s,len(sizes_above_threshold)))
            continue
    
    if (not isinstance(s,str)) or (len(s)==0):
        continue
    if s in scientific_name_to_paths:
        continue
    print('Processing query {} of {} ({})'.format(i_row,len(df),s))
    paths = retrieve_sample_image.download_images(query=s,
                                             output_directory=image_base,
                                             limit=images_per_query,
                                             verbose=True)
    print('Downloaded {} images for {}'.format(len(paths),s))
    scientific_name_to_paths[s] = paths

# ...for each row in the mapping table


#%% Choose representative images for each scientific name

max_images_per_query = 4
scientific_name_to_preferred_images = {}

# s = list(scientific_name_to_paths.keys())[0]
for s in list(df.scientific_name):
    
    if not isinstance(s,str):
        continue
    
    query = s.replace(' ','+')
    query_folder = os.path.join(image_base,query)
    assert os.path.isdir(query_folder)
    image_files = os.listdir(query_folder)
    image_fullpaths = [os.path.join(query_folder,fn) for fn in image_files]    
    sizes = [os.path.getsize(p) for p in image_fullpaths]
    paths_by_size = [x for _, x in sorted(zip(sizes, image_fullpaths),reverse=True)]
    preferred_paths = paths_by_size[:max_images_per_query]
    scientific_name_to_preferred_images[s] = preferred_paths

# ...for each scientific name    


#%% Produce HTML preview

from tqdm import tqdm

with open(html_output_file, 'w') as f:
    
    f.write('<html><head></head><body>\n')

    names = scientific_name_to_preferred_images.keys()
    names = sorted(names)
    
    f.write('<p class="speciesinfo_p" style="font-weight:bold;font-size:130%">')
    f.write('datset_name: <b><u>category</u></b> mapped to taxonomy_level scientific_name (taxonomic_common_name) (manual_common_name)</p>\n'.format())
    f.write('</p>')
    # i_row = 2; row = df.iloc[i_row]
    for i_row, row in tqdm(df.iterrows(), total=len(df)):
        
        s = row['scientific_name']
        
        taxonomy_string = row['taxonomy_string']
        if isinstance(taxonomy_string,str):
            taxonomic_match = eval(taxonomy_string)        
            matched_entity = taxonomic_match[0]
            assert len(matched_entity) == 4
            common_names = matched_entity[3]
            if len(common_names) == 1:
                common_name_string = common_names[0]
            else:
                common_name_string = str(common_names)
        else:
            common_name_string = ''
        
        f.write('<p class="speciesinfo_p" style="font-weight:bold;font-size:130%">')
        
        if isinstance(row.scientific_name,str):
            f.write('{}: <b><u>{}</u></b> mapped to {} {} ({}) ({})</p>\n'.format(
                row.dataset_name, row.query, 
                row.taxonomy_level, row.scientific_name, common_name_string,
                row.common_name))
        else:
            f.write('{}: <b><u>{}</u></b> unmapped'.format(row.dataset_name,row.query))
        
        if s is None or s not in names:
            f.write('<p class="content_p">no images available</p>')
        else:
            image_paths = scientific_name_to_preferred_images[s]
            basedir = os.path.dirname(html_output_file)
            relative_paths = [os.path.relpath(p,basedir) for p in image_paths]
            image_paths = [s.replace('\\','/') for s in relative_paths]
            n_images = len(image_paths)
            # image_paths = [os.path.relpath(p, output_base) for p in image_paths]
            image_width_percent = round(100 / n_images)
            f.write('<table class="image_table"><tr>\n')
            for image_path in image_paths:
                f.write('<td style="vertical-align:top;" width="{}%">'
                        '<img src="{}" style="display:block; width:100%; vertical-align:top; height:auto;">'
                        '</td>\n'.format(image_width_percent, image_path))
            f.write('</tr></table>\n')

    # ...for each row

    f.write('</body></html>\n')


#%% Open HTML preview

from path_utils import open_file # from ai4eutils
open_file(html_output_file)

#
# snapshot_serengeti_lila.py
#
# Create zipfiles of Snapshot Serengeti S1-S11.
#
# Create a metadata file for S1-S10, plus separate metadata files
# for S1-S11.  At the time this code was written, S11 was under embargo.
#
# Create zip archives of each season without humans.
#
# Create a human zip archive.
#

#%% Constants and imports

import pandas as pd
import json
import os
import uuid
import pickle
import humanfriendly
import time
import pprint
import numpy as np
import re
import glob

from PIL import Image
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from zipfile import ZipFile
import zipfile

# import sys; sys.path.append(r'c:\git\ai4eutils')
import path_utils

# import sys; sys.path.append(r'c:\git\cameratraps')
   
metadata_base = r'e:\snapshot-serengeti\MetaData\SER'
image_base = r'e:\snapshot-serengeti\SER'

bbox_file = r'e:\snapshot_serengeti_bboxes_20190409.json'
old_json_file = r'e:\SnapshotSerengeti.json'

temp_base = r'e:\snapshot_temp'
output_base = r'e:\snapshot_out'
output_zip_base = r'e:\snapshot_out'

os.makedirs(temp_base,exist_ok=True)
os.makedirs(output_base,exist_ok=True)

# assert(os.path.isdir(metadata_base))
assert(os.path.isdir(image_base))
assert(os.path.isfile(bbox_file))
assert(os.path.isfile(old_json_file))

nSeasons = 11
non_public_seasons = set()

initial_dictionary_cache_file = os.path.join(temp_base,'initial_dictionaries.p')
resized_images_dictionary_cache_file = os.path.join(temp_base,'resized_image_dictionaries.p')
revised_annotations_dictionary_cache_file = os.path.join(temp_base,'revised_annotations.p')
final_dictionary_cache_file = os.path.join(temp_base,'final_annotations.p')

# There are two redundant categories, and we re-map "blank" to "empty" as per CCT convention
category_mappings = {'blank':'empty','birdother':'otherbird','vervetmonkey':'monkeyvervet'}

process_images_n_threads = 20


#%% Load metadata files, concatenate into a single table

per_season_image_tables = []
per_season_annotation_tables = []

print('Reading tables...')

# iSeason = 1
for iSeason in tqdm(range(1,nSeasons+1)):
    image_table_fn = os.path.join(metadata_base,'SER_S' + str(iSeason) + '_report_lila_image_inventory.csv')
    annotation_table_fn = os.path.join(metadata_base,'SER_S' + str(iSeason) + '_report_lila.csv')
    assert os.path.isfile(image_table_fn) and os.path.isfile(annotation_table_fn)
    image_table = pd.read_csv(image_table_fn)
    annotation_table = pd.read_csv(annotation_table_fn)
    per_season_image_tables.append(image_table)
    per_season_annotation_tables.append(annotation_table)
    
print('Finished reading tables, concatenating...')    

image_table = pd.concat(per_season_image_tables)
annotation_table = pd.concat(per_season_annotation_tables)

print('Finished concatenating {} images and {} annotations'.format(len(image_table),len(annotation_table)))


#%% Convert to dictionaries (prep)

im_id_to_image = {}
images = []
seq_id_to_images = {}    
seq_id_to_annotations = {}

annotations = []
categories = []

species_to_category = {}

empty_category_id = 0
empty_category_name = 'empty'

empty_cat = {}
empty_cat['id'] = empty_category_id
empty_cat['name'] = empty_category_name
empty_cat['count'] = 0
species_to_category['empty'] = empty_cat
categories.append(empty_cat)

next_category_id = empty_category_id + 1


#%% Convert to dictionaries (loops)
    
# TODO: iterrows() is a terrible way to do this, but this is one of those days
# where I want to get this done, not get better at Python.

print('Processing image table')

start_time = time.time()

# irow = 0; row = image_table.iloc[0]
for iRow,row in tqdm(image_table.iterrows()):
    
    # Loaded as an int64, converting to int here
    frame_num = int(row['image_rank_in_capture'])
    assert frame_num > 0    
    sequence_id = row['capture_id']
    frame_num = int(frame_num)
    filename = row['image_path_rel']
    tokens = filename.split('.')
    assert(len(tokens)==2)
    assert(tokens[1] == 'JPG')
    id = tokens[0]
    
    im = {}
    im['id'] = id
    im['file_name'] = filename
    im['frame_num'] = frame_num
    im['seq_id'] = sequence_id
            
    assert id not in im_id_to_image
    im_id_to_image[id] = im
    seq_id_to_images.setdefault(sequence_id,[]).append(im)

    images.append(im)
        
# ...for each row in the image table

# Make sure image IDs are what we think they are
for im in tqdm(images):
    assert im['id'] == im['file_name'].replace('.JPG','')

print('Processing annotation table')

for iRow,row in tqdm(annotation_table.iterrows()):

    sequence_id = row['capture_id']
    
    species = row['question__species'].lower()
    if species in category_mappings:
        species = category_mappings[species]
        
    category = None
    
    if species not in species_to_category:
        category = {}
        category['id'] = next_category_id
        next_category_id = next_category_id + 1
        category['name'] = species
        category['count'] = 1
        categories.append(category)
        species_to_category[species] = category
    else:
        category = species_to_category[species]
        category['count'] += 1
        
    ann = {}
    ann['sequence_level_annotation'] = True
    ann['id'] = str(uuid.uuid1())    
    ann['category_id'] = category['id']
    ann['seq_id'] = sequence_id
    
    ann['season'] = row['season']
    ann['site'] = row['site']
    ann['datetime'] = row['capture_date_local'] + ' ' + row['capture_time_local']
    ann['subject_id'] = row['subject_id']
    ann['count'] = row['question__count_median']
    ann['standing'] = row['question__standing']    
    ann['resting'] = row['question__resting']
    ann['moving'] = row['question__moving']
    ann['interacting'] = row['question__interacting']
    ann['young_present'] = row['question__young_present']    
    
    seq_id_to_annotations.setdefault(sequence_id,[]).append(ann)
    
    annotations.append(ann)

# ...for each row in the annotation table

elapsed = time.time() - start_time
print('Done converting tables to dictionaries in {}'.format(humanfriendly.format_timespan(elapsed)))

print('Saving dictionaries to {}'.format(initial_dictionary_cache_file))
cct_cache = [im_id_to_image, images, seq_id_to_images, 
             seq_id_to_annotations, annotations, categories, species_to_category]        
with open(initial_dictionary_cache_file, 'wb') as f:
    pickle.dump(cct_cache, f, protocol=pickle.HIGHEST_PROTOCOL)


#%% Load previously-saved dictionaries when re-starting mid-script
    
if False:
    
    #%%
    
    print('Loading dictionaries from {}'.format(initial_dictionary_cache_file))
    with open(initial_dictionary_cache_file, 'rb') as f:
        cct_cache = pickle.load(f)    
    im_id_to_image,images,seq_id_to_images, \
             seq_id_to_annotations,annotations,categories,species_to_category = cct_cache
         

#%% Take a look at categories (just sanity-checking)

if False:
    
    #%%
    
    assert(len(im_id_to_image)==len(images)) 
    print('Loaded metadata about {} images and {} sequences'.format(len(images),len(seq_id_to_annotations)))

    categories_by_species = sorted(categories, key = lambda i: i['name'])
    categories_by_count = sorted(categories, key = lambda i: i['count'])
    
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(categories_by_species)
    pp.pprint(categories_by_count)


#%% Fill in some image fields we didn't have when we created the image table

# width, height, corrupt, seq_num_frames, location, datetime

def process_image(im):
    
    im['width'] = -1
    im['height'] = -1    
    im['corrupt'] = False
    im['location'] = 'unknown'
    im['seq_num_frames'] = -1
    im['datetime'] = 'unknown'
    im['status'] = ''
    
    if im['seq_id'] not in seq_id_to_annotations:
        im['status'] = 'no_annotation'
        return im
    
    seq_annotations = seq_id_to_annotations[im['seq_id']]
    
    # Every annotation in this list should have the same sequence ID
    assert all(ann['seq_id'] == im['seq_id'] for ann in seq_annotations) , 'Error on image {}'.format(im['id'])

    # Figure out "seq_num_frames", which really should be done in a separate lopp;
    # there's no reason to do this redundantly for every image
    images_in_sequence = seq_id_to_images[im['seq_id']]
    
    # Every image in this sequence should point back to the same equence
    assert all(seqim['seq_id'] == im['seq_id'] for seqim in images_in_sequence), 'Error on image {}'.format(im['id'])
    
    frame_nums = [seqim['frame_num'] for seqim in images_in_sequence]
    seq_num_frames = max(frame_nums)
    im['seq_num_frames'] = seq_num_frames
    
    im['location'] = seq_annotations[0]['site']
    
    # Every annotation in this list should have the same location
    assert all(ann['site'] == im['location'] for ann in seq_annotations), 'Error on image {}'.format(im['id'])
    
    im['datetime'] = seq_annotations[0]['datetime']
    
    # Every annotation in this list should have the same datetime
    assert all(ann['datetime'] == im['datetime'] for ann in seq_annotations), 'Error on image {}'.format(im['id'])
    
    # Is this image on disk?
    fullpath = os.path.join(image_base,im['file_name'])
    if not os.path.isfile(fullpath):
        im['status'] = 'not_on_disk'
        return im

    try:
        
        pil_im = Image.open(fullpath)
        im['height'] = pil_im.height
        im['width'] = pil_im.width
            
    except:
            
        im['corrupt'] = True        

    return im

    
if process_images_n_threads <= 1:
    
    # iImage = 0; im = images[0]    
    for iImage,im in tqdm(enumerate(images),total=len(images)):
        process_image(im)
    # ...for each image    

else:
    
    pool = ThreadPool(process_images_n_threads)
    
    # images_processed = pool.map(process_image, images)
    # images_processed = list(tqdm(pool.imap_unordered(process_image, images), total=len(images)))
    images_processed = list(tqdm(pool.imap(process_image, images), total=len(images)))

print('Saving size-checked dictionaries to {}'.format(resized_images_dictionary_cache_file))
cct_cache = [im_id_to_image, images, seq_id_to_images, 
             seq_id_to_annotations, annotations, categories, species_to_category]        
with open(resized_images_dictionary_cache_file, 'wb') as f:
    pickle.dump(cct_cache, f, protocol=pickle.HIGHEST_PROTOCOL)


if False:
    
    #%%
    
    print('Loading dictionaries with size information from {}'.format(resized_images_dictionary_cache_file))
    with open(resized_images_dictionary_cache_file, 'rb') as f:
        cct_cache = pickle.load(f)    
    im_id_to_image,images,seq_id_to_images, \
             seq_id_to_annotations,annotations,categories,species_to_category = cct_cache
    

#%% Count missing/corrupted images    

n_missing = 0
n_corrupt = 0
n_no_annotation = 0

for im in tqdm(images):
    
    if im['corrupt']:
        n_corrupt += 1
        
    if im['status'] == '':
        continue
    elif im['status'] == 'not_on_disk':
        n_missing += 1
    elif im['status'] == 'no_annotation':
        n_no_annotation += 1
    else:
        raise ValueError('Unrecognized status {}'.format(im['status']))

print('\nOf {} images: {} missing, {} corrupt, {} no annotation'.format(len(images),
      n_missing, n_corrupt, n_no_annotation))
    

#%% Print distribution of sequence lengths (sanity-check)

seq_id_to_sequence_length = {}

for im in tqdm(images):
    
    seq_id = im['seq_id']
    seq_num_frames = im['seq_num_frames']
    if seq_id not in seq_id_to_sequence_length:
        seq_id_to_sequence_length[seq_id] = seq_num_frames

sequence_lengths = list(seq_id_to_sequence_length.values())        
    
print('\nMean/min/max sequence length is {}/{}/{}'.format(np.mean(sequence_lengths),min(sequence_lengths),max(sequence_lengths)))


#%% Replicate annotations across images

annotations_replicated = []

# iAnn = 0; ann = annotations[iAnn]
for iAnn,ann in tqdm(enumerate(annotations), total=len(annotations)):
    
    associated_images = seq_id_to_images[ann['seq_id']]
    assert len(associated_images) > 0
    for associated_image in associated_images:
        new_ann = ann.copy()
        new_ann['image_id'] = associated_image['id']
        new_ann['id'] = str(uuid.uuid1())
        annotations_replicated.append(new_ann)
        
print('\nCreated {} replicated annotations from {} original annotations'.format(len(annotations_replicated),
      len(annotations)))

annotations = annotations_replicated


#%% See what files are on disk but not annotated (~15 mins)
    
print('Listing images from disk...')
start_time = time.time()
all_files = path_utils.recursive_file_list(image_base)
elapsed = time.time() - start_time
print('Finished listing {} files in {}'.format(len(all_files),humanfriendly.format_timespan(elapsed)))

files_not_in_db = []

for fn in tqdm(all_files):
    id = os.path.relpath(fn,image_base).replace('\\','/').replace('.JPG','')
    if id not in im_id_to_image:
        files_not_in_db.append(fn)

print('{} files not in the database (of {})'.format(len(files_not_in_db),len(all_files)))

# 247370 files not in the database (of 7425810)


#%% Load old image database

print('Loading old .json file...',end='')
with open(old_json_file,'r') as f:
    cct_old = json.load(f)
print('done')


#%% Look for old images not in the new DB and vice-versa

# At the time this was written, "old" was S1-S6

# old_im = cct_old['images'][0]
old_im_id_to_im = {}
old_images_not_in_new_db = []
new_images_not_in_old_db = []
size_mismatches = []
for old_im in tqdm(cct_old['images']):
    old_im_id_to_im[old_im['id']] = old_im
    if old_im['id'] not in im_id_to_image:
        old_images_not_in_new_db.append(old_im)
    else:
        new_im = im_id_to_image[old_im['id']]
        if (old_im['width'] != new_im['width']) or (old_im['height'] != new_im['height']):
            size_mismatches.append(old_im)
        
# new_im = images[0]
for new_im in tqdm(images):
    new_id = new_im['id']
    if new_id.startswith('SER_S11'):
        continue
    m = re.search('^S(\d+)/',new_id)
    if m is None:
        print('Failed to match season number in {}'.format(id))
        continue
    season = int(m.group(1))
    if season > 6:
        continue
    if new_id not in old_im_id_to_im:
        new_images_not_in_old_db.append(new_im)
        
print('{} old images not in new db'.format(len(old_images_not_in_new_db)))        
print('{} new images not in old db'.format(len(new_images_not_in_old_db)))
print('{} size mismatches'.format(len(size_mismatches)))

# 4 old images not in new db
# 12 new images not in old db


#%% Save our work

print('Saving revised-annotation dictionaries to {}'.format(revised_annotations_dictionary_cache_file))
cct_cache = [im_id_to_image, images, seq_id_to_images, 
             seq_id_to_annotations, annotations, categories, species_to_category, all_files]        
with open(revised_annotations_dictionary_cache_file, 'wb') as f:
    pickle.dump(cct_cache, f, protocol=pickle.HIGHEST_PROTOCOL)


#%% Load our work
    
if False:
    
    #%%
    print('Loading dictionaries from {}'.format(revised_annotations_dictionary_cache_file))
    with open(revised_annotations_dictionary_cache_file, 'rb') as f:
        cct_cache = pickle.load(f)    
    im_id_to_image, images, seq_id_to_images, \
        seq_id_to_annotations, annotations, categories, species_to_category, all_files = cct_cache


#%% Examine size mismatches

# i_mismatch = -1; old_im = size_mismatches[i_mismatch]
for i_mismatch,old_im in enumerate(size_mismatches):
    new_im = im_id_to_image[old_im['id']]    

seasons = list(range(1,7))
mismatches_by_season = []
for season in seasons:
    season_mismatches = [x for x in size_mismatches if x['id'].startswith('S' + str(season))]
    mismatches_by_season.append(season_mismatches)

for iSeason,season_mismatches in enumerate(mismatches_by_season):
    print('Size mismatches in season {}: {}'.format(iSeason+1,len(mismatches_by_season[iSeason])))


#%% Sanity-check image and annotation uniqueness
             
tmp_img_ids = set()
tmp_ann_ids = set()

for im in tqdm(images):
    assert im['id'] not in tmp_img_ids
    tmp_img_ids.add(im['id'])

for ann in tqdm(annotations):
    assert ann['id'] not in tmp_ann_ids
    tmp_ann_ids.add(ann['id'])
    

#%% Split data by seasons, create master list for public seasons
             
annotations_by_season = [[] for i in range(nSeasons)]
annotations_public = []

image_ids_by_season = [set() for i in range(nSeasons)]
image_ids_public = set()

# ann = annotations[0]
for ann in tqdm(annotations):
    season_id = ann['image_id'].split('/')[0]
    assert season_id is not None and season_id.startswith('S')
    season = int(season_id.replace('SER_','S').replace('S',''))
    assert season >=1 and season <= nSeasons
    i_season = season - 1
    annotations_by_season[i_season].append(ann)
    im = im_id_to_image[ann['image_id']]
    image_ids_by_season[i_season].add(im['id'])

    if season not in non_public_seasons:
        annotations_public.append(ann)
        image_ids_public.add(im['id'])

images_by_season = []
for id_list in image_ids_by_season:
    season_images = []
    for id in id_list:
        season_images.append(im_id_to_image[id])
    images_by_season.append(season_images)
    
images_public = []
for id in image_ids_public:
    images_public.append(im_id_to_image[id])

for season in range(1,nSeasons+1):
    i_season = season - 1
    print('Season {}: {} images, {} annotations'.format(season,len(images_by_season[i_season]),
          len(annotations_by_season[i_season])))

print('Public total: {} images, {} annotations'.format(len(images_public),
      len(annotations_public)))


#%% Minor updates to fields
    
for ann in tqdm(annotations):
    ann['location'] = ann['site']
    del ann['site']
    try:
        icount = ann['count']
    except:
        icount = -1
    ann['count'] = icount
    
for im in tqdm(images):
    del im['status']
    
for c in categories:
    del c['count']


#%% Write master .json out for S1-10, write individual season .jsons (including S11)

info = {}
info['version'] = '2.0'
info['description'] = 'Camera trap data from the Snapshot Serengeti program'
info['date_created'] = '2019'
info['contributor'] = 'University of Minnesota Lion Center'

# Loop over all seasons, plus one iteration for the "all public data" iteration, and
# one for the "all data" iteration
for season in range(1,nSeasons+3):
    i_season = season - 1
    data = {}
    data['info'] = info.copy()
    data['categories'] = categories
    if i_season == nSeasons + 1:
        data['info']['version'] = '2.1'
        data['info']['description'] = data['info']['description'] + ', seasons 1-11'
        data['images'] = images
        data['annotations'] = annotations
        fn = os.path.join(output_base,'SnapshotSeregeti_v2.1.json'.format(season))
    elif i_season == nSeasons:
        data['info']['description'] = data['info']['description'] + ', seasons 1-10'
        data['images'] = images_public
        data['annotations'] = annotations_public        
        fn = os.path.join(output_base,'SnapshotSeregeti_v2.0.json'.format(season))
    else:
        data['info']['description'] = data['info']['description'] + ', season {}'.format(season)    
        data['images'] = images_by_season[i_season]
        data['annotations'] = annotations_by_season[i_season]
        fn = os.path.join(output_base,'SnapshotSerengetiS{:0>2d}.json'.format(season))
    
    print('Writing data for season {} to {}'.format(season,fn))
    
    s = json.dumps(data,indent=1)
    with open(fn, "w+") as f:
        f.write(s)

    
#%% Find categories that only exist in S11

# List of categories in each season
categories_by_season = [set() for i in range(nSeasons)]

for ann in tqdm(annotations):
    season_id = ann['image_id'].split('/')[0]
    assert season_id is not None and season_id.startswith('S')
    season = int(season_id.replace('SER_','S').replace('S',''))
    assert season >=1 and season <= nSeasons
    i_season = season - 1
    categories_by_season[i_season].add(ann['category_id'])

cat_id_to_cat = {} 
for c in categories:
    cat_id_to_cat[c['id']] = c
    
category_counts_by_season = [len(c) for c in categories_by_season]
target_season_idx = 10

for id in categories_by_season[target_season_idx]:
    b_found = False
    for i_season,season_categories in enumerate(categories_by_season):
        if i_season == target_season_idx:
            continue
        if id in season_categories:
            b_found = True
            break
    if not b_found:
        print('Category {} ({}) only in S{}'.format(id,cat_id_to_cat[id]['name'],target_season_idx+1))

for cat in categories:
    if cat['id'] not in categories_by_season[target_season_idx]:
        print('Category {} ({}) not in S{}'.format(cat['id'],cat['name'],target_season_idx+1))

# Category 55 (fire) only in S11
# Category 56 (hyenabrown) only in S11
# Category 57 (wilddog) only in S11
# Category 58 (kudu) only in S11
# Category 59 (pangolin) only in S11
# Category 60 (lioncub) only in S11


#%% Prepare season-specific .csv files

per_season_image_tables = []
per_season_annotation_tables = []

print('Reading tables...')

# iSeason = 1
for iSeason in tqdm(range(1,nSeasons+1)):
    image_table_fn = os.path.join(metadata_base,'SER_S' + str(iSeason) + '_report_lila_image_inventory.csv')
    annotation_table_fn = os.path.join(metadata_base,'SER_S' + str(iSeason) + '_report_lila.csv')
    assert os.path.isfile(image_table_fn) and os.path.isfile(annotation_table_fn)
    image_table = pd.read_csv(image_table_fn)
    annotation_table = pd.read_csv(annotation_table_fn)
    per_season_image_tables.append(image_table)
    per_season_annotation_tables.append(annotation_table)
    
print('Finished reading tables, concatenating...')    

image_table_public = pd.concat(per_season_image_tables[0:-1],sort=False)
annotation_table_public = pd.concat(per_season_annotation_tables[0:-1],sort=False)

image_table_all = pd.concat(per_season_image_tables,sort=False)
annotation_table_all = pd.concat(per_season_annotation_tables,sort=False)

print('Finished concatenating {} images and {} annotations'.format(len(image_table),len(annotation_table)))

fn_image_csv_public = os.path.join(output_base,'SnapshotSerengeti_v2_0_images.csv')
fn_annotation_csv_public = os.path.join(output_base,'SnapshotSerengeti_v2_0_annotations.csv')
fn_image_csv_all = os.path.join(output_base,'SnapshotSerengeti_v2_1_images.csv')
fn_annotation_csv_all = os.path.join(output_base,'SnapshotSerengeti_v2_1_annotations.csv')

image_table_public.to_csv(fn_image_csv_public)
annotation_table_public.to_csv(fn_annotation_csv_public)

image_table_all.to_csv(fn_image_csv_all)
annotation_table_all.to_csv(fn_annotation_csv_all)


#%% Create a list of human files

human_image_ids = set()
human_id = species_to_category['human']['id']

# ann = annotations[0]
for ann in tqdm(annotations):
    if ann['category_id'] == human_id:
        human_image_ids.add(ann['image_id'])

print('Found {} images with humans'.format(len(human_image_ids)))


#%% Save our work

print('Saving final dictionaries to {}'.format(final_dictionary_cache_file))
cct_cache = [im_id_to_image, images, seq_id_to_images, 
             seq_id_to_annotations, annotations, categories, species_to_category, all_files, 
             human_id, human_image_ids]        
with open(final_dictionary_cache_file, 'wb') as f:
    pickle.dump(cct_cache, f, protocol=pickle.HIGHEST_PROTOCOL)


#%% Load our work
    
if False:
    
    #%%
    print('Loading dictionaries from {}'.format(final_dictionary_cache_file))
    with open(final_dictionary_cache_file, 'rb') as f:
        cct_cache = pickle.load(f)    
    im_id_to_image, images, seq_id_to_images, \
        seq_id_to_annotations, annotations, categories, species_to_category, all_files, \
        human_id, human_image_ids = cct_cache


#%% Create archives (human, per-season) (prep)

human_zipfile = os.path.join(output_zip_base,'SnapshotSerengeti_humans_v2.0.zip')
os.makedirs(output_zip_base,exist_ok=True)

debug_max_files = -1
n_dot = 1000
n_print = 10000
max_files_per_archive = 500000

def create_human_archive():
    
    n_images_added = 0
    with ZipFile(human_zipfile,'w') as zip:
        
        print('Creating archive {}'.format(human_zipfile))
    
        # im = images[0]
        for iImage,im in enumerate(images):
            if im['id'] in human_image_ids:
                n_images_added += 1
                if debug_max_files > 0 and n_images_added > debug_max_files:
                    break
                if (n_images_added % n_dot)==0:
                    print('.',end='')
                if (n_images_added % n_print)==0:
                    print('{} images added to {}'.format(n_images_added,human_zipfile))
                source_file = os.path.join(image_base,im['file_name'])
                dest_file = im['file_name']
                zip.write(source_file,dest_file,zipfile.ZIP_STORED)
    
    print('\nFinished writing {}, added {} files'.format(human_zipfile,n_images_added))
    
    return n_images_added

def create_season_archive(i_season):
    
    season = i_season + 1
    zipfilename = os.path.join(output_zip_base,'SnapshotSerengeti_S{:>02d}_v2_0.zip'.format(season))

    n_images_added = 0
    zip = ZipFile(zipfilename,'w')
        
    print('Creating archive {}'.format(zipfilename))
    
    # im = images[0]
    for iImage,im in enumerate(images):

        # Don't include humans
        if im['id'] in human_image_ids:
            continue     
        
        # Only include files from this season
        season_id = im['id'].split('/')[0]
        assert season_id is not None and season_id.startswith('S')
        season = int(season_id.replace('SER_','S').replace('S',''))
        assert season >=1 and season <= nSeasons
        
        if (season != i_season + 1):
            continue
        
        n_images_added += 1
        
        # Possibly start a new archive
        if n_images_added >= max_files_per_archive:
            zip.close()
            zipfilename = zipfilename.replace('.zip','.{}.zip'.format(n_images_added))
            print('Starting new archive for season {}: {}'.format(i_season+1,zipfilename))
            zip = ZipFile(zipfilename,'w')
            n_images_added = 0
            
        if (n_images_added % n_dot)==0:
            print('.',end='')
        if (n_images_added % n_print)==0:
            print('{} images added to {}'.format(n_images_added,zipfilename))            
        if debug_max_files > 0 and n_images_added > debug_max_files:
            break
        
        source_file = os.path.join(image_base,im['file_name'])
        dest_file = im['file_name']
        zip.write(source_file,dest_file,zipfile.ZIP_STORED)

    # ...for each image
    
    zip.close()
    print('\nFinished writing {}, added {} files'.format(zipfilename,n_images_added))

    return n_images_added

# i_season = 0
# for i_season in range(0,nSeasons):
#   create_season_archive(i_season)

def create_archive(i_season):
    if i_season == -1:
        return create_human_archive()
    else:
        return create_season_archive(i_season)
        
    
#%% Create archives (loop)    
        
# pool = ThreadPool(nSeasons+1)
# n_images = pool.map(create_archive, range(-1,nSeasons))
# seasons_to_zip = range(-1,nSeasons)
seasons_to_zip = [ 4,6 ]
for i_season in seasons_to_zip:
    create_archive(i_season)
    
# ...for each season

 
#%% Sanity-check .json files

# %logstart -o r'E:\snapshot_temp\python.txt'
    
from data_management.databases import sanity_check_json_db

files_to_check = glob.glob(os.path.join(output_base,'*.json'))

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = image_base
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = False
    
for fn in files_to_check:    
    sortedCategories, data = sanity_check_json_db.sanity_check_json_db(fn,options)


#%% Zip up .json and .csv files

def zip_single_file(fn):
    
    zipfilename = fn + '.zip'
    print('Zipping {} to {}'.format(fn,zipfilename))
    with ZipFile(zipfilename,'w') as zip:
        source_file = fn
        dest_file = os.path.basename(fn)
        zip.write(source_file,dest_file,zipfile.ZIP_DEFLATED)

files_to_zip = []
files_to_zip.extend(glob.glob(os.path.join(output_base,'*.csv')))
files_to_zip.extend(glob.glob(os.path.join(output_base,'*.json')))

# pool = ThreadPool(len(files_to_zip))
# pool.map(zip_single_file, files_to_zip)
for fn in tqdm(files_to_zip):
    if os.path.isfile(fn + '.zip'):
        print('Skipping {}'.format(fn))
        continue
    zip_single_file(fn)
  
 
#%% Super-sanity-check that S11 info isn't leaking

files_to_check = glob.glob(os.path.join(output_base,'*.json'))

for jsonFn in files_to_check:

    if '11' in jsonFn or '2_1' in jsonFn:
        print('Skipping file {}'.format(jsonFn))
        continue
    
    print('Processing file {}'.format(jsonFn))

    with open(jsonFn,'r') as f:
        data_public = json.load(f)

    # im = data_public['images'][0]
    for im in tqdm(data_public['images']):
        assert (not im['id'].startswith('S11')) and (not im['id'].startswith('SER11'))
        assert (not im['file_name'].startswith('S11')) and (not im['file_name'].startswith('SER11'))
        sequence_tokens = im['seq_id'].split('#')
        assert '11' not in sequence_tokens[0]        

    # ann = data_public['annotations'][0]
    for ann in tqdm(data_public['annotations']):
        assert (not ann['image_id'].startswith('S11')) and (not ann['image_id'].startswith('SER11'))
        sequence_tokens = ann['seq_id'].split('#')
        assert '11' not in sequence_tokens[0]        
    
print('Done checking .json files')    

annotation_csv = "E:\snapshot_out\SnapshotSerengeti_v2_0_annotations.csv"
image_csv = "E:\snapshot_out\SnapshotSerengeti_v2_0_images.csv"

annotation_df = pd.read_csv(annotation_csv)
image_df = pd.read_csv(image_csv)

# iRow = 0; row = annotation_df.iloc[iRow]
for iRow,row in tqdm(annotation_df.iterrows(),total=len(annotation_df)):
    sequence_tokens = row['capture_id'].split('#')
    assert '11' not in sequence_tokens[0]
    assert '11' not in row['season']

# iRow = 0; row = image_df.iloc[iRow]
for iRow,row in tqdm(image_df.iterrows(),total=len(image_df)):
    sequence_tokens = row['capture_id'].split('#')
    assert '11' not in sequence_tokens[0]
    fn = row['image_path_rel']
    assert (not fn.startswith('S11')) and (not fn.startswith('SER11'))
    
print('Done checking .csv files')


#%% Create bounding box archive

bbox_json_fn = r"E:\snapshot_serengeti_bboxes_20190409.json"

with open(bbox_json_fn,'r') as f:
        bbox_data = json.load(f)

json_fn = r"E:\snapshot_out\SnapshotSeregeti_v2.0.json"

with open(json_fn,'r') as f:
        data = json.load(f)

print('Finished reading annotations and bounding boxes')

available_images = set()

# i_image = 0; im = data['images'][0]
for i_image,im in enumerate(data['images']):
    available_images.add(im['id'])

print('{} images available'.format(len(available_images)))    

missing_images = []
found_images = []
# i_box = 0; boxann = bbox_data['annotations'][0]
for i_ann,ann in enumerate(bbox_data['annotations']):
    id = ann['image_id']
    if id not in available_images:
        missing_images.append(id)
    else:
        found_images.append(id)
        
print('{} missing images in {} bounding boxes ({} found)'.format(len(missing_images), len(bbox_data['annotations']), len(found_images)))


#%% Sanity-check a few files to make sure bounding boxes are still sensible

# import sys; sys.path.append(r'C:\git\CameraTraps')
from visualization import visualize_db
output_base = r'E:\snapshot_temp'

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 500
viz_options.trim_to_images_with_bboxes = True
viz_options.add_search_links = True
viz_options.sort_by_filename = False
html_output_file,bbox_db = visualize_db.process_images(bbox_json_fn,os.path.join(output_base,'preview2'),image_base,viz_options)
os.startfile(html_output_file)


#%% Check categories

json_fn_all = r"E:\snapshot_out\SnapshotSeregeti_v2.0.json"

with open(json_fn_all,'r') as f:
    data_all = json.load(f)

data_by_season = []
data_10 = None
i_season = 9
fn = r'e:\snapshot_out\SnapshotSerengetiS{:0>2d}.json'.format(i_season+1)
with open(fn,'r') as f:
    data_10 = json.load(f)
        
n_categories_all = len(data_all['categories'])
n_categories_s10 = len(data_10['categories'])


#%% Summary prep for LILA

import json
from tqdm import tqdm
import os

json_fn = r"D:\temp\SnapshotSeregeti_v2.0.json"
with open(json_fn,'r') as f:
    data = json.load(f)

categories = data['categories']
annotations = data['annotations']
images = data['images']
output_base = r'd:\temp'

n_empty = 0
n_species = len(categories)
n_images = len(images)

sequences = set()
for im in tqdm(images):
    sequences.add(im['seq_id'])

category_id_to_count = {}
for ann in tqdm(annotations):    
    if ann['category_id'] == 0:
        n_empty += 1
    if ann['category_id'] in category_id_to_count:
        category_id_to_count[ann['category_id']] += 1
    else:
        category_id_to_count[ann['category_id']] = 1

empty_categories = []    
for c in categories:
    if c['id'] in category_id_to_count:
        c['count'] = category_id_to_count[c['id']]
    else:
        empty_categories.append(c)
        c['count'] = 0

categories = [c for c in categories if c['count'] > 0]
sorted_categories = sorted(categories, key=lambda k: k['count'], reverse=True) 

fn = os.path.join(output_base,'ss_specieslist.csv')
with open(fn,'w') as f:
    for c in sorted_categories:
        f.write(c['name'] + ',' + str(c['count']) + '\n')

print('Found {} images ({} empty, {}%) in {} sequences, in {} categories'.format(
        n_images,n_empty,100*n_empty/n_images,len(sequences),len(categories)))


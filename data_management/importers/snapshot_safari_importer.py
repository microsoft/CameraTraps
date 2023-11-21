#
# Import a Snapshot Safari project (one project, one season)
#
# Before running this script:
#
# * Mount the blob container where the images live, or copy the 
#   images to local storage
#
# What this script does:
#
# * Creates a .json file
# * Creates zip archives of the season without humans.
# * Copies animals and humans to separate folders 
#
# After running this script:
#
# * Create or update LILA page
# * Push zipfile and unzipped images to LILA
# * Push unzipped humans to wildlifeblobssc
# * Delete images from UMN uplaod storage
#
# Snapshot Serengeti is handled specially, because we're dealing with bounding
# boxes too.  See snapshot_serengeti_lila.py.
#

#%% Imports

import pandas as pd
import json
import os
import uuid
import humanfriendly
import time
import pprint
import numpy as np
import shutil

from PIL import Image
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from zipfile import ZipFile
import zipfile

# From ai4eutils
import path_utils

# From CameraTraps
from visualization import visualize_db


#%% Constants

# project_name = 'XXX'; season_name = 'S1'; project_friendly_name = 'Snapshot Unknown'
# project_name = 'SER'; season_name = 'S1-11'; project_friendly_name = 'Snapshot Serengeti'
# project_name = 'KRU'; season_name = 'S1'; project_friendly_name = 'Snapshot Kruger'
# project_name = 'CDB'; season_name = 'S1'; project_friendly_name = 'Snapshot Camdeboo'
# project_name = 'MTZ'; season_name = 'S1'; project_friendly_name = 'Snapshot Mountain Zebra'
# project_name = 'ENO'; season_name = 'S1'; project_friendly_name = 'Snapshot Enonkishu'
# project_name = 'KAR'; season_name = 'S1'; project_friendly_name = 'Snapshot Karoo'
# project_name = 'KGA'; season_name = 'S1'; project_friendly_name = 'Snapshot Kgalagadi'
project_name = 'SER'; season_name = 'S1'; project_friendly_name = 'APN'

json_version = '2.1'

snapshot_safari_input_base = 'f:\\'
snapshot_safari_output_base = r'g:\temp\snapshot-safari-out'

category_mappings = {'blank':'empty'}

process_images_n_threads = 20


#%% Folder/file creation

# E.g. KRU_S1
project_season_name = project_name + '_' + season_name

# E.g. Z:\KRU
project_base = os.path.join(snapshot_safari_input_base,project_name)
assert(os.path.isdir(project_base))

# E.g. Z:\KRU\KRU_S1
season_base = os.path.join(project_base,project_season_name)
assert(os.path.isdir(season_base))

# Contains annotations for each capture event (sequence)
annotation_file = os.path.join(project_base,project_season_name + '_report_lila.csv')

# Maps image IDs to filenames; each line looks like:
#
# KRU_S1#1#1#2,3,KRU_S1/1/1_R1/KRU_S1_1_R1_IMAG0004.JPG
image_inventory_file = os.path.join(project_base,project_season_name + '_report_lila_image_inventory.csv')

# Total number of each answer to each question, e.g. total number of times each species was identified
#
# Not used here
response_overview_file = os.path.join(project_base,project_season_name + '_report_lila_overview.csv')

assert(os.path.isfile(annotation_file))
assert(os.path.isfile(image_inventory_file))
assert(os.path.isfile(response_overview_file))

# Create output folders
assert(os.path.isdir(snapshot_safari_output_base))

output_base = os.path.join(snapshot_safari_output_base,project_name)

json_filename = os.path.join(output_base,project_friendly_name.replace(' ','') + '_' + season_name \
                             + '_v' + json_version + '.json')
species_list_filename = os.path.join(output_base,project_friendly_name.replace(' ','') + '_' + season_name \
                             + '_v' + json_version + '.species_list.csv')
summary_info_filename = os.path.join(output_base,project_friendly_name.replace(' ','') + '_' + season_name \
                             + '_v' + json_version + '.summary_info.txt')

# Images will be placed in a season-specific folder inside this (the source data includes
# this in path names)
output_public_folder = os.path.join(output_base,project_name + '_public')

output_public_zipfile = os.path.join(output_base,project_season_name + '.lila.zip')
output_private_folder = os.path.join(output_base,project_season_name + '_private')
output_preview_folder = os.path.join(output_base,project_season_name + '_preview')

os.makedirs(output_base,exist_ok=True)
os.makedirs(output_public_folder,exist_ok=True)
os.makedirs(output_private_folder,exist_ok=True)
os.makedirs(output_preview_folder,exist_ok=True)


#%% Load metadata files

image_table = pd.read_csv(image_inventory_file)
annotation_table = pd.read_csv(annotation_file)

print('Finished loading {} image mappings and {} annotations'.format(len(image_table),len(annotation_table)))


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
for iRow,row in tqdm(image_table.iterrows(),total=len(image_table)):
    
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

def is_float_and_nan(x):
    return isinstance(x,float) and np.isnan(x)

n_invalid_dates = 0
    
for iRow,row in tqdm(annotation_table.iterrows(),total=len(annotation_table)):

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
    if is_float_and_nan(row['capture_date_local']) or is_float_and_nan(row['capture_time_local']):
        ann['datetime'] = ''
        n_invalid_dates += 1
    else:
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

print('Converted {} annotations, {} images, {} categories ({} invalid dates)'.format(
    len(annotations),len(images),len(categories),n_invalid_dates))

    
#%% Take a look at categories (just sanity-checking)

assert(len(im_id_to_image)==len(images)) 
print('Loaded metadata about {} images and {} sequences'.format(len(images),len(seq_id_to_annotations)))

categories_by_species = sorted(categories, key = lambda i: i['name'])
categories_by_count = sorted(categories, key = lambda i: i['count'])

pp = pprint.PrettyPrinter(depth=6)

# print('\nCategories by species:')
# pp.pprint(categories_by_species)
print('\nCategories by count:')
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
    
    im['location'] = str(seq_annotations[0]['site'])
    
    # Every annotation in this list should have the same location
    assert all(str(ann['site']) == im['location'] for ann in seq_annotations), 'Error on image {}'.format(im['id'])
    
    im['datetime'] = seq_annotations[0]['datetime']
    
    # Every annotation in this list should have the same datetime
    assert all(ann['datetime'] == im['datetime'] for ann in seq_annotations), 'Error on image {}'.format(im['id'])
    
    # Is this image on disk?
    fullpath = os.path.join(project_base,im['file_name'])
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
    
print('Finished adding missing fields to {} images'.format(len(images_processed)))


#%% Count missing/corrupted images    

n_missing = 0
n_corrupt = 0
n_no_annotation = 0

corrupted_images = []
missing_images = []
no_annotation_images = []

for im in tqdm(images):
    
    if im['corrupt']:
        n_corrupt += 1
        corrupted_images.append(im['file_name'])
    if im['status'] == '':
        continue
    elif im['status'] == 'not_on_disk':
        n_missing += 1
        missing_images.append(im['file_name'])
    elif im['status'] == 'no_annotation':
        n_no_annotation += 1
        no_annotation_images.append(im['file_name'])
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


#%% See what files are on disk but not annotated
    
print('Listing images from disk...')
start_time = time.time()
image_files = path_utils.find_images(project_base,bRecursive=True)
elapsed = time.time() - start_time
print('Finished listing {} files in {}'.format(len(image_files),humanfriendly.format_timespan(elapsed)))

files_not_in_db = []

for fn in tqdm(image_files):
    id = os.path.relpath(fn,project_base).replace('\\','/').replace('.JPG','')
    if id not in im_id_to_image:
        files_not_in_db.append(fn)

print('{} files not in the database (of {})'.format(len(files_not_in_db),len(image_files)))
del fn


#%% Sanity-check image and annotation uniqueness
             
tmp_img_ids = set()
tmp_ann_ids = set()

for im in tqdm(images):
    assert im['id'] not in tmp_img_ids
    tmp_img_ids.add(im['id'])

for ann in tqdm(annotations):
    assert ann['id'] not in tmp_ann_ids
    tmp_ann_ids.add(ann['id'])
    
print('Finished uniqueness sanity-check')


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


#%% Write .json file

info = {}
info['version'] = json_version
info['description'] = 'Camera trap data from the {} program'.format(project_friendly_name)
info['date_created'] = '2019'
info['contributor'] = 'Snapshot Safari'

data = {}
data['info'] = info
data['categories'] = categories
data['annotations'] = annotations
data['images'] = images

print('Writing data to {}'.format(json_filename))

s = json.dumps(data,indent=1)
with open(json_filename, "w+") as f:
    f.write(s)

    
#%% Create a list of human files

human_image_ids = set()
human_id = species_to_category['human']['id']

# ann = annotations[0]
for ann in tqdm(annotations):
    if ann['category_id'] == human_id:
        human_image_ids.add(ann['image_id'])

print('Found {} images with humans'.format(len(human_image_ids)))


#%% Create public archive and public/private folders

debug_max_files = -1
n_dot = 1000
n_print = 10000

n_images_added = 0
zipfilename = output_public_zipfile
zip = ZipFile(zipfilename,'w')
    
print('Creating archive {}'.format(output_public_zipfile))

# im = images[0]
for iImage,im in tqdm(enumerate(images),total=len(images)):

    # E.g. KRU_S1/1/1_R1/KRU_S1_1_R1_IMAG0001.JPG
    im_relative_path = im['file_name']
    im_absolute_path = os.path.join(project_base,im_relative_path)
    assert(os.path.isfile(im_absolute_path))
    
    image_is_private = (im['id'] in human_image_ids)

    if image_is_private:
        
        # Copy to private output folder            
        output_file = os.path.join(output_private_folder,im_relative_path)
        os.makedirs(os.path.dirname(output_file),exist_ok=True)
        shutil.copyfile(src=im_absolute_path,dst=output_file)
        continue     
    
    # Add to zipfile        
    n_images_added += 1
    
    # Possibly start a new archive
    if n_images_added >= max_files_per_archive:
        zip.close()
        zipfilename = zipfilename.replace('.zip','.{}.zip'.format(n_images_added))
        print('Starting new archive: {}'.format(zipfilename))
        zip = ZipFile(zipfilename,'w')
        n_images_added = 0
        
    if (n_images_added % n_dot)==0:
        print('.',end='')
    if (n_images_added % n_print)==0:
        print('{} images added to {}'.format(n_images_added,zipfilename))            
    if debug_max_files > 0 and n_images_added > debug_max_files:
        break
    
    source_file = os.path.join(project_base,im_relative_path)
    dest_file = im['file_name']
    zip.write(source_file,dest_file,zipfile.ZIP_STORED)

    # Copy to public output folder
    output_file = os.path.join(output_public_folder,im_relative_path)
    os.makedirs(os.path.dirname(output_file),exist_ok=True)        
    shutil.copyfile(src=im_absolute_path,dst=output_file)

# ...for each image

zip.close()

print('\nFinished writing {}, added {} files'.format(zipfilename,n_images_added))


#%% Sanity-check .json file

from data_management.databases import sanity_check_json_db

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = output_public_folder
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = False

sortedCategories, data, errorInfo = sanity_check_json_db.sanity_check_json_db(json_filename,options)

# This will produce some validation errors, because this zipfile doesn't include humans
assert(len(errorInfo['validationErrors']) == len(human_image_ids))


#%% Zip up .json and .csv files

def zip_single_file(fn,zipfilename=None):
    '''
    Zips a single file fn, by default to fn.zip
    
    Discards path information, only uses fn's base name.
    '''
    if zipfilename is None:
        zipfilename = fn + '.zip'
        
    print('Zipping {} to {}'.format(fn,zipfilename))
    with ZipFile(zipfilename,'w') as zip:
        source_file = fn
        dest_file = os.path.basename(fn)
        zip.write(source_file,dest_file,zipfile.ZIP_DEFLATED)
    return zipfilename

def zip_files_to_flat(filenames,zipfilename):
    '''
    Remove path information from everything in [filenames] and
    zip to a flat file.  Does not check uniqueness.
    '''
    with ZipFile(zipfilename,'w') as zip:
        for fn in filenames:
            source_file = fn
            dest_file = os.path.basename(fn)
            zip.write(source_file,dest_file,zipfile.ZIP_DEFLATED)

zip_single_file(json_filename)
zip_files_to_flat([annotation_file,image_inventory_file],os.path.join(output_base,project_season_name + '.csv.zip'))

print('Finished zipping .csv and .json files')


#%% When I skip to this part (using a pre-rendered .json file)

if False:

    #%%

    species_to_category = {}
    for cat in categories:
        species_to_category[cat['name']] = cat
        
    #%%
    
    human_image_ids = set()
    human_id = species_to_category['human']['id']
    
    # ann = annotations[0]
    for ann in tqdm(annotations):
        if ann['category_id'] == human_id:
            human_image_ids.add(ann['image_id'])
    
    print('Found {} images with humans'.format(len(human_image_ids)))

 
#%% Summary prep for LILA

with open(json_filename,'r') as f:
    data = json.load(f)

categories = data['categories']
annotations = data['annotations']
images = data['images']

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

with open(species_list_filename,'w') as f:
    for c in sorted_categories:
        f.write(c['name'] + ',' + str(c['count']) + '\n')

n_images = len(images) - len(human_image_ids)
n_sequences = len(sequences)
percent_empty = (100*n_empty)/len(images)
n_categories = len(categories)
top_categories = []

for i_category in range(0,len(sorted_categories)):
    c = sorted_categories[i_category]
    cat_name = c['name']
    if cat_name != 'human' and cat_name != 'empty':        
        top_categories.append(cat_name)
        if len(top_categories) == 3:
            break
    
s = 'This data set contains {} sequences of camera trap images, totaling {} images, from the {} project. Labels are provided for {} categories, primarily at the species level (for example, the most common labels are {}, {}, and {}). Approximately {:.2f}% of images are labeled as empty. A full list of species and associated image counts is available <a href="{}">here</a>.'.format(
    n_sequences,n_images,project_friendly_name,n_categories,
    top_categories[0],top_categories[1],top_categories[2],
    percent_empty,
    'https://lilablobssc.blob.core.windows.net/snapshot-safari/{}/{}_{}_v{}.species_list.csv'.format(
        project_name,project_friendly_name.replace(' ',''),season_name,json_version))
print(s)

with open(summary_info_filename,'w') as f:
    f.write(s)
    

#%% Generate preview, sanity-check labels
    
from visualization import visualize_db
viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 5000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.classes_to_exclude = ['test','empty']
# viz_options.classes_to_include = ['jackalblackbacked','bustardkori']
html_output_file, image_db = visualize_db.process_images(db_path=json_filename,
                                                         output_dir=output_preview_folder,
                                                         image_base_dir=output_public_folder,
                                                         options=viz_options)
os.startfile(html_output_file)


#%% Scrap

if False:
    
    pass

    #%% Find annotations for a particular image

    fn = missing_images[1000]
    id = fn.replace('.JPG','')
    im = im_id_to_image[id]
    seq_id = im['seq_id']
    matching_annotations = [ann for ann in annotations if ann['seq_id'] == seq_id]
    print(matching_annotations)
    
    #%% Write a list of missing images
    
    with open(os.path.join(output_base,project_name + '_' + season_name + '_missing_images.txt'), 'w') as f:
        for fn in missing_images:
            f.write('{}\n'.format(fn))
                    

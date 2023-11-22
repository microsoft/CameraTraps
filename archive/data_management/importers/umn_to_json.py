#
# umn_to_json.py
#
# Prepare images and metadata for the Orinoquía Camera Traps dataset.
#

#%% Imports and constants

import os
import json
import pandas as pd
import shutil
import uuid
import datetime
import dateutil.parser

from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from multiprocessing.pool import ThreadPool

input_base = "f:\\"
image_base = os.path.join(input_base,'2021.11.24-images\jan2020')
ground_truth_file = os.path.join(input_base,'images_hv_jan2020_reviewed_force_nonblank.csv')

# For two deployments, we're only processing imagse in the "detections" subfolder
detection_only_deployments = ['N23','N32']
deployments_to_ignore = ['N18','N28']

MISSING_COMMON_NAME_TOKEN = 'MISSING'
    
assert os.path.isfile(ground_truth_file)
assert os.path.isdir(image_base)


#%% Enumerate deployment folders

deployment_folders = os.listdir(image_base)
deployment_folders = [s for s in deployment_folders if os.path.isdir(os.path.join(image_base,s))]
deployment_folders = set(deployment_folders)
print('Listed {} deployment folders'.format(len(deployment_folders)))


#%% Load ground truth

ground_truth_df = pd.read_csv(ground_truth_file)

print('Loaded {} ground truth annotations'.format(
    len(ground_truth_df)))

# i_row = 0; row = ground_truth_df.iloc[i_row]
for i_row,row in tqdm(ground_truth_df.iterrows()):
    if not isinstance(row['common_name'],str):
        print('Warning: missing common name for {}'.format(row['filename']))
        row['common_name'] = MISSING_COMMON_NAME_TOKEN
    
    
#%% Create relative paths for ground truth data

# Some deployment folders have no subfolders, e.g. this is a valid file name:
# 
# M00/01010132.JPG
#
# But some deployment folders have subfolders, e.g. this is also a valid file name:
#
# N17/100EK113/07160020.JPG
#
# So we can't find files by just concatenating folder and file names, we have to enumerate and explicitly
# map what will appear in the ground truth as "folder/filename" to complete relative paths.

deployment_name_to_file_mappings = {}

n_filenames_ignored = 0
n_deployments_ignored = 0

# deployment_name = list(deployment_folders)[0]
for deployment_name in tqdm(deployment_folders):
    
    file_mappings = {}
    
    if deployment_name in deployments_to_ignore:
        print('Ignoring deployment {}'.format(deployment_name))
        n_deployments_ignored += 1
        continue
    
    # Enumerate all files in this folder
    absolute_deployment_folder = os.path.join(image_base,deployment_name)
    assert os.path.isdir(absolute_deployment_folder)
    
    files = list(Path(absolute_deployment_folder).rglob('*'))
    files = [p for p in files if not p.is_dir()]
    files = [str(s) for s in files]
    files = [s.replace('\\','/') for s in files]
    # print('Enumerated {} files for deployment {}'.format(len(files),deployment_name))
    
    # filename = files[100]
    for filename in files:
        
        if deployment_name in detection_only_deployments and 'detection' not in filename:
            n_filenames_ignored += 1
            continue
        
        if '.DS_Store' in filename:
            n_filenames_ignored += 1
            continue
        
        relative_path = os.path.relpath(filename,absolute_deployment_folder).replace('\\','/')
        image_name = relative_path.split('/')[-1]
        assert image_name not in file_mappings, 'Redundant image name {} in deployment {}'.format(
            image_name,deployment_name)
        assert '\\' not in relative_path
        file_mappings[image_name] = relative_path
    
    # ...for each file in this deployment
    
    deployment_name_to_file_mappings[deployment_name] = file_mappings

# ...for each deployment

print('Processed deployments, ignored {} deployments and {} files'.format(
    n_deployments_ignored,n_filenames_ignored))
    

#%% Add relative paths to our ground truth table

ground_truth_df['relative_path'] = None

# i_row = 0; row = ground_truth_df.iloc[i_row]
for i_row,row in tqdm(ground_truth_df.iterrows(),total=len(ground_truth_df)):
        
    # row['filename'] looks like, e.g. A01/01080001.JPG.  This is not actually a path, it's
    # just the deployment ID and the image name, separated by a slash.

    deployment_name = row['filename'].split('/')[0]
    
    assert deployment_name in deployment_folders, 'Could not find deployment folder {}'.format(deployment_name)
    assert deployment_name in deployment_name_to_file_mappings, 'Could not find deployment folder {}'.format(deployment_name)
    
    file_mappings = deployment_name_to_file_mappings[deployment_name]
            
    # Find the relative path for this image    
    image_name = row['filename'].split('/')[-1]
    assert image_name in file_mappings, 'No mappings for image {} in deployment {}'.format(
        image_name,deployment_name)    
    relative_path = os.path.join(deployment_name,file_mappings[image_name]).replace('\\','/')
    
    # Make sure this image file exists
    absolute_path = os.path.join(image_base,relative_path)
    assert os.path.isfile(absolute_path), 'Could not find file {}'.format(absolute_path)
    
    ground_truth_df.loc[i_row,'relative_path'] = relative_path

# ...for each row in the ground truth table


#%% Take everything out of Pandas

ground_truth_dicts = ground_truth_df.to_dict('records')


#%% Convert string timestamps to Python datetimes

all_locations = set()

# im = ground_truth_dicts[0]
for im in tqdm(ground_truth_dicts):
    dt = dateutil.parser.isoparse(im['timestamp'])
    assert dt.year == 2020
    im['datetime'] = dt
    
    # Filenames look like, e.g., N36/100EK113/06040726.JPG    
    im['location'] = im['relative_path'].split('/')[0]
    assert len(im['location']) == 3
    all_locations.add(im['location'])
    

#%% Synthesize sequence information

locations = all_locations
print('Found {} locations'.format(len(locations)))

locations = list(locations)

sequences = set()
sequence_to_images = defaultdict(list) 
images = ground_truth_dicts
max_seconds_within_sequence = 10

# Sort images by time within each location
# i_location=0; location = locations[i_location]
for i_location,location in tqdm(enumerate(locations)):
    
    images_this_location = [im for im in images if im['location'] == location]
    sorted_images_this_location = sorted(images_this_location, key = lambda im: im['datetime'])
    
    current_sequence_id = None
    next_frame_number = 0
    previous_datetime = None
        
    # previous_datetime = sorted_images_this_location[0]['datetime']
    # im = sorted_images_this_camera[1]
    for i_image,im in enumerate(sorted_images_this_location):

        # Timestamp for this image, may be None
        dt = im['datetime']
        
        # Start a new sequence if:
        #
        # * This image has no timestamp
        # * This iamge has a frame number of zero
        # * We have no previous image timestamp
        #
        if dt is None:
            delta = None
        elif previous_datetime is None:
            delta = None
        else:
            assert isinstance(dt,datetime.datetime)
            delta = (dt - previous_datetime).total_seconds()
        
        # Start a new sequence if necessary
        if delta is None or delta > max_seconds_within_sequence:
            next_frame_number = 0
            current_sequence_id = str(uuid.uuid1())
            sequences.add(current_sequence_id)
        assert current_sequence_id is not None
        
        im['seq_id'] = current_sequence_id
        im['synthetic_frame_number'] = next_frame_number
        next_frame_number = next_frame_number + 1
        previous_datetime = dt
        sequence_to_images[im['seq_id']].append(im)
    
    # ...for each image in this location

# ...for each location


#%% Create category dict and category IDs
    
categories_to_counts = defaultdict(int)
category_mappings = {'blank':'empty',
                     'mammal':'unknown_mammal',
                     'dasypus_species':'unknown_armadillo',
                     'bird':'unknown_bird',
                     'bos_species':'cattle',
                     'possum_family':'unknown_possum',
                     'cervidae_family':'unknown_cervid',
                     'unknown_species':'unknown',
                     'lizards_and_snakes':'unknown_reptile',
                     'caprimulgidae_family':'unknown_nightjar',
                     'turtle_order':'unknown_turtle',
                     'ornate_tití_monkey':'ornate_titi_monkey',
                     'saimiri_species':'unknown_squirrel_monkey',
                     'peccary_family':'unknown_peccary',
                     'pecari_species':'unknown_peccary',
                     'alouatta_species':'unknown_howler_monkey',
                     'human-camera_trapper':'human',
                     'weasel_family':'unknown_weasel',
                     'motorcycle':'human',
                     'eira_species':'unknown_tayra',
                     'sapajus_species':'unknown_capuchin_monkey',
                     'red_brocket':'red_brocket_deer'
                     }

for c in category_mappings.values():
    assert ' ' not in c
    
# im = images[0]
for im in tqdm(images):
    
    category_name = im['common_name'].lower().replace("'",'').replace(' ','_')
    if category_name in category_mappings:
        category_name = category_mappings[category_name]
    categories_to_counts[category_name] += 1
    im['category_name'] = category_name


categories_to_counts_sorted = {k: v for k, v in sorted(categories_to_counts.items(),
                                                       key=lambda item: item[1],reverse=True)}

for s in categories_to_counts_sorted.keys():
    print('{}: {}'.format(s,categories_to_counts_sorted[s]))


#%% Imports and constants (.json generation)
    
import os
import uuid
import datetime
from tqdm import tqdm

from data_management.databases import sanity_check_json_db

output_base = 'f:\orinoquia_camera_traps'
output_image_base = os.path.join(output_base,'images')
os.makedirs(output_image_base,exist_ok=True)

output_json_filename = os.path.join(output_base, 'orinoquia_camera_traps.json')
output_encoding = 'utf-8'
read_image_sizes = False

info = {}
info['year'] = 2020
info['version'] = '1.0'
info['description'] = 'Orinoquia Camera Traps'
info['contributor'] = 'University of Minnesota'
info['date_created'] = str(datetime.date.today())


#%% Count frames in each sequence

sequence_id_to_n_frames = defaultdict(int)

for im in tqdm(images):
    seq_id = im['seq_id']
    sequence_id_to_n_frames[seq_id] = sequence_id_to_n_frames[seq_id] + 1

for im in tqdm(images):
    seq_id = im['seq_id']
    im['seq_num_frames'] = sequence_id_to_n_frames[seq_id]


#%% Double check images with multiple annotations

filename_to_images = defaultdict(list)

# im = images[0]
for im in tqdm(images):
    fn = im['relative_path']
    filename_to_images[fn].append(im)
    
filenames_with_multiple_annotations = [fn for fn in filename_to_images.keys() if len(filename_to_images[fn]) > 1]

print('Found {} filenames with multiple annotations'.format(len(filenames_with_multiple_annotations)))

for fn in filenames_with_multiple_annotations:
    images_this_file = filename_to_images[fn]
    print(fn + ': ')
    for im in images_this_file:
        print(im['category_name'])
    print('')


#%% Assemble dictionaries

images_out = []
image_id_to_image = {}
annotations = []
categories = []

category_name_to_category = {}
category_id_to_category = {}

# Force the empty category to be ID 0
empty_category = {}
empty_category['name'] = 'empty'
empty_category['id'] = 0
empty_category['count'] = 0

category_id_to_category[0] = empty_category
category_name_to_category['empty'] = empty_category
categories.append(empty_category)
next_id = 1

# input_im = images[0]
for input_im in tqdm(images):

    category_name = input_im['category_name'].lower().strip()
    
    if category_name not in category_name_to_category:        

        category_id = next_id
        next_id += 1
        category = {}
        category['id'] = category_id
        category['name'] = category_name
        category['count'] = 0
        categories.append(category)
        category_name_to_category[category_name] = category
        category_id_to_category[category_id] = category

    else:
        
        category = category_name_to_category[category_name]
        
    category_id = category['id']    
    category['count'] += 1
    
    im = {}
    im['id'] = input_im['relative_path'].replace('/','_')
    im['datetime'] = str(input_im['datetime'])
    im['file_name'] = input_im['relative_path']
    im['seq_id'] = input_im['seq_id']
    im['frame_num'] = input_im['synthetic_frame_number']
    im['seq_num_frames'] = input_im['seq_num_frames']
    im['location'] = input_im['location']
    
    if im['id'] in image_id_to_image:
        print('Warning: image ID {} ({}) has multiple annotations'.format(im['id'],im['id'].replace('_','/')))
    else:
        image_id_to_image[im['id']] = im    
        images_out.append(im)
    
    ann = {}
    
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']
    ann['category_id'] = category_id
    ann['sequence_level_annotation'] = False
    annotations.append(ann)

# ...for each image


#%% Write output .json

data = {}
data['info'] = info
data['images'] = images_out
data['annotations'] = annotations
data['categories'] = categories

with open(output_json_filename, 'w') as f:
    json.dump(data, f, indent=1)
    
print('Finished writing json to {}'.format(output_json_filename))

    
#%% Validate .json file

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = output_base
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False

_, _, _ = sanity_check_json_db.sanity_check_json_db(output_json_filename, options)


#%% Map relative paths to annotation categories

category_id_to_category_names = {c['id']:c['name'] for c in data['categories']}
image_id_to_category_names = defaultdict(list)

# ann = data['annotations'][0]
for ann in data['annotations']:
    category_name = category_id_to_category_names[ann['category_id']]
    image_id_to_category_names[ann['image_id']].append(category_name)
    

#%% Copy images to output

# EXCLUDE HUMAN AND MISSING

# im = data['images'][0]
def copy_image(im):
    
    image_id = im['id']
    category_names_this_image = image_id_to_category_names[image_id]
    assert len(category_names_this_image) > 0
    if ('human' in category_names_this_image) or ('missing' in category_names_this_image):
        prefix = 'private'
    else:
        prefix = 'public'
    input_fn_absolute = os.path.join(image_base,im['file_name'])
    output_fn_absolute = os.path.join(output_image_base,prefix,im['file_name'])
    dirname = os.path.dirname(output_fn_absolute)
    os.makedirs(dirname,exist_ok=True)
    shutil.copy(input_fn_absolute,output_fn_absolute)

n_threads = 10

# im = images[0]
if n_threads == 1:    
    for im in tqdm(data['images']):
        copy_image(im)
else:
    pool = ThreadPool(n_threads)
    with tqdm(total=len(data['images'])) as pbar:
        for i,_ in enumerate(pool.imap_unordered(copy_image,data['images'])):
            pbar.update()
            
    
#%% Preview labels

from visualization import visualize_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 100
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.include_filename_links = True

# viz_options.classes_to_exclude = ['test']
html_output_file, _ = visualize_db.process_images(db_path=output_json_filename,
                                                         output_dir=os.path.join(
                                                         output_base,'preview'),
                                                         image_base_dir=os.path.join(output_image_base,'public'),
                                                         options=viz_options)
os.startfile(html_output_file)



#
# wi_to_json
#
# Prepares CCT-formatted metadata based on a Wildlife Insights data export.
# 
# Mostly assumes you have the images also, for validation/QA.
#

#%% Imports and constants

import os
import json
import pandas as pd
import shutil
import uuid
import datetime
import dateutil.parser
import sys
import subprocess
import copy

from collections import defaultdict
from tqdm import tqdm
from visualization import visualize_db
from data_management.databases import integrity_check_json_db

organization_name = 'organization'
input_base = os.path.expanduser('~/data/' + organization_name)
image_base = os.path.join(input_base,'deployment')
image_csv = os.path.join(input_base,'images.csv')
output_json_filename = os.path.join(input_base, organization_name + '_camera_traps.json')
preview_base = os.path.expanduser('~/data/' + organization_name + '/preview')

assert os.path.isfile(image_csv)
assert os.path.isdir(image_base)

MISSING_COMMON_NAME_TOKEN = 'MISSING'

output_encoding = 'utf-8'

# Because WI filenames are GUIDs, it's not practical to page through sequences in an
# image viewer.  So we're going to (optionally) create a copy of the data set where
# images are ordered.
create_ordered_dataset = False

ordered_image_base = os.path.join(input_base,'deployment-ordered')
ordered_json_filename = os.path.join(input_base, organization_name + '_camera_traps_ordered.json')
ordered_preview_base = os.path.expanduser('~/data/' + organization_name + '/preview-ordered')

info = {}
info['year'] = 2020
info['version'] = '1.0'
info['description'] = organization_name + ' camera traps)'
info['contributor'] = organization_name
info['date_created'] = str(datetime.date.today())

def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


#%% Load ground truth

images_df = pd.read_csv(image_csv)

print('Loaded {} ground truth annotations'.format(
    len(images_df)))


#%% Take everything out of Pandas

images = images_df.to_dict('records')


#%% Synthesize common names when they're not available

for im in images:
        
    if not isinstance(im['common_name'],str):
        
        # Blank rows should always have "Blank" as the common name
        assert im['is_blank'] == 0
        assert isinstance(im['genus'],str) and isinstance(im['species'],str)
        # print('Warning: missing common name for row {} ({})'.format(i_row,row['filename']))
        im['common_name'] = im['genus'].strip() + ' ' + im['species'].strip()
    
    
#%% Convert string timestamps to Python datetimes

all_locations = set()

# im = ground_truth_dicts[0]
for im in tqdm(images):
    dt = dateutil.parser.isoparse(im['timestamp'])
    assert dt.year >= 2019 and dt.year <= 2021
    im['datetime'] = dt
    
    # The field called "location" in the WI .csv file is a URL, we want to reclaim
    # the "location" keyword for CCT output
    im['url'] = im['location']
    
    # Filenames look like, e.g., N36/100EK113/06040726.JPG    
    im['location'] = im['deployment_id']
    all_locations.add(im['location'])
    

#%% Synthesize sequence information

locations = all_locations
print('Found {} locations'.format(len(locations)))

locations = list(locations)

sequences = set()
sequence_to_images = defaultdict(list) 
max_seconds_within_sequence = 10

# Sort images by time within each location
# i_location=0; location = locations[i_location]
for i_location,location in tqdm(enumerate(locations),total=len(locations)):
    
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
                     'bird':'unknown_bird',
                     'unknown_species':'unknown'
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

print('\n')
for s in categories_to_counts_sorted.keys():
    print('{}: {}'.format(s,categories_to_counts_sorted[s]))


#%% Count frames in each sequence

sequence_id_to_n_frames = defaultdict(int)

for im in tqdm(images):
    seq_id = im['seq_id']
    sequence_id_to_n_frames[seq_id] = sequence_id_to_n_frames[seq_id] + 1

for im in tqdm(images):
    seq_id = im['seq_id']
    im['seq_num_frames'] = sequence_id_to_n_frames[seq_id]


#%% Build relative paths

missing_images = []

# im = images[0]
for i_image,im in enumerate(tqdm(images)):
    # Sample URL:
    #
    # gs://project-asfasdfd/deployment/21444549/asdfasdfd-616a-4d10-a921-45ac456c568a.jpg'
    relative_path = im['url'].split('/deployment/')[1]
    assert relative_path is not None and len(relative_path) > 0
    im['relative_path'] = relative_path
    
    if not os.path.isfile(os.path.join(image_base,relative_path)):
        missing_images.append(im)
    
print('{} images are missing'.format(len(missing_images)))


#%% Double check images with multiple annotations

filename_to_images = defaultdict(list)

# im = images[0]
for im in tqdm(images):
    filename_to_images[im['relative_path']].append(im)
    
filenames_with_multiple_annotations = [fn for fn in filename_to_images.keys() if len(filename_to_images[fn]) > 1]

print('\nFound {} filenames with multiple annotations'.format(len(filenames_with_multiple_annotations)))


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
        # print('Warning: image ID {} ({}) has multiple annotations'.format(im['id'],im['id'].replace('_','/')))
        pass
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

options = integrity_check_json_db.IntegrityCheckOptions()
options.baseDir = image_base
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True

_, _, _ = integrity_check_json_db.integrity_check_json_db(output_json_filename, options)


#%% Preview labels

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 300
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.include_filename_links = True

html_output_file, _ = visualize_db.process_images(db_path=output_json_filename,
                                                         output_dir=preview_base,
                                                         image_base_dir=image_base,
                                                         options=viz_options)
open_file(html_output_file)
# open_file(os.path.join(image_base,'2100703/1141a545-88d2-498b-a684-7431f7aeb324.jpg'))


#%%

if create_ordered_dataset:
    
    pass

    #%% Create ordered dataset
    
    # Because WI filenames are GUIDs, it's not practical to page through sequences in an
    # image viewer.  So we're going to create a copy of the data set where images are 
    # ordered.
    
    os.makedirs(ordered_image_base,exist_ok=True)
    
    ordered_images = {}
    
    # im = images_out[0]; im
    for im in tqdm(images_out):
        im_out = copy.deepcopy(im)
        ordered_filename = im['location'] + '_' + im['seq_id'] + '_' +\
            str(im['frame_num']) + '_' + os.path.basename(im['file_name'])
        assert ordered_filename not in ordered_images
        im_out['original_file'] = im_out['file_name']
        im_out['file_name'] = ordered_filename
        ordered_images[ordered_filename] = im_out
        
    ordered_images = list(ordered_images.values())
    
    
    #%% Create ordered .json 
    
    data_ordered = copy.copy(data)
    data_ordered['images'] = ordered_images
    
    with open(ordered_json_filename, 'w') as f:
        json.dump(data_ordered, f, indent=1)
        
    print('Finished writing json to {}'.format(ordered_json_filename))
    
    
    #%% Copy files to their new locations
        
    # im = ordered_images[0]
    for im in tqdm(ordered_images):
        output_file = os.path.join(ordered_image_base,im['file_name'])
        input_file = os.path.join(image_base,im['original_file'])
        if not os.path.isfile(input_file):
            print('Warning: file {} is missing'.format(input_file))
            continue
        shutil.copyfile(input_file,output_file)
    
    original_fn_to_ordered_fn = {}
    # im = data_ordered['images'][0]
    for im in data_ordered['images']:
        original_fn_to_ordered_fn[im['original_file']] = im['file_name']    
    
    
    #%% Preview labels in the ordered dataset
    
    viz_options = visualize_db.DbVizOptions()
    viz_options.num_to_visualize = 300
    viz_options.trim_to_images_with_bboxes = False
    viz_options.add_search_links = True
    viz_options.sort_by_filename = False
    viz_options.parallelize_rendering = True
    viz_options.include_filename_links = True
    
    html_output_file, _ = visualize_db.process_images(db_path=ordered_json_filename,
                                                             output_dir=ordered_preview_base,
                                                             image_base_dir=ordered_image_base,
                                                             options=viz_options)
    open_file(html_output_file)
    # open_file(os.path.join(image_base,'2100703/1141a545-88d2-498b-a684-7431f7aeb324.jpg'))
        
    
    #%% Open an ordered filename from the unordered filename    
    
    unordered_filename = '2100557/54e5c751-28b4-42e3-b6d4-e8ee290228ae.jpg'
    fn = os.path.join(ordered_image_base,original_fn_to_ordered_fn[unordered_filename])
    open_file(fn)
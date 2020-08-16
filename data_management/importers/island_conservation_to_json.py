#
# island_conservation_to_json.py
#
# Convert the Island Conservation data set to a COCO-camera-traps .json file
#

#%% Constants and environment

import os
import csv
import json
import uuid
import datetime
import shutil
import zipfile
import pandas as pd

from PIL import Image
from tqdm import tqdm
from collections import defaultdict 

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
from path_utils import find_images

# Base directory for all input images and metadata
#
# Should contain folders "ic-images", "icr2-images", and "icr3-images"
input_dir_base = r'e:\island-conservation\20200529drop'

# Folder containing the input metadata files (unzipped from IC_AI4Earth_2019_timelapse_by_project.zip)
input_dir_json = os.path.join(
    input_dir_base, 'IC_AI4Earth_2019_timelapse_by_project')

# Dictionary mapping original island names to obfuscated names
island_name_mapping_file = os.path.join(input_dir_base,'island_name_mappings.csv')

# List of images that have not been reviewed by a human
images_not_reviewed_file = os.path.join(input_dir_base,'IC_AI4E_2019_images_not_reviewed.csv')

assert(os.path.isdir(input_dir_base))
assert(os.path.isdir(input_dir_json))
assert(os.path.isfile(island_name_mapping_file))

output_dir_base = r'f:\data_staging\island_conservation'
os.makedirs(output_dir_base, exist_ok=True)
output_dir_images = os.path.join(output_dir_base, 'images')
os.makedirs(output_dir_images, exist_ok=True)

output_json_file = os.path.join(output_dir_base, 'island_conservation.json')

human_dir = os.path.join(output_dir_base, 'human')
non_human_dir = os.path.join(output_dir_base, 'non-human')

human_zipfile = os.path.join(output_dir_base, 'island_conservation_camera_traps_humans.zip')
non_human_zipfile = os.path.join(output_dir_base, 'island_conservation_camera_traps.zip')

# Clean existing output folders/zipfiles
if os.path.isdir(human_dir):
    shutil.rmtree(human_dir)
if os.path.isdir(non_human_dir):
    shutil.rmtree(non_human_dir)

if os.path.isfile(human_zipfile):
    os.remove(human_zipfile)
if os.path.isfile(human_zipfile):
    os.remove(non_human_zipfile)

os.makedirs(human_dir, exist_ok=True)
os.makedirs(non_human_dir, exist_ok=True)

category_mapping = {"buow": "burrowing owl",
                    "baow": "barred owl",
                    "grhe": "green heron",
                    "amke": "american kestrel",
                    "gbhe": "great blue heron",
                    "brno": "brown noddy",
                    "ycnh": "yellow-crowned night heron",
                    "wwdo": "white-winged dove",
                    "seow": "short-eared owl",
                    "person": "human",
                    "null": "empty"}


#%% Load island name mappings from .csv

# We map island names to generic region/country names using an external .csv file

island_name_mappings = None

with open(island_name_mapping_file, mode='r') as f:
    reader = csv.reader(f)
    island_name_mappings = {rows[0]:rows[1] for rows in reader}


#%% Enumerate input images

image_full_paths = find_images(input_dir_base, recursive=True)
print('Enumerated {} images from {}'.format(len(image_full_paths),input_dir_base))


#%% Load list of non-reviewed images

df = pd.read_csv(images_not_reviewed_file)

non_reviewed_filenames = []
for i_row,row in df.iterrows():
    non_reviewed_filenames.append(os.path.join(row['folder'],row['filename']).lower().replace('\\','/'))

non_reviewed_filenames_set = set(non_reviewed_filenames)

print('Enumerated {} non-reviewed filenames ({} unique)'.\
      format(len(non_reviewed_filenames),len(non_reviewed_filenames_set)))


#%% Rename files for obfuscation
    
output_file_names = set()

# PERF: this should be parallelized, it's almost entirely I/O-bound

# image_full_path = image_full_paths[0]
for image_full_path in tqdm(image_full_paths):
    
    destination_relative_path = os.path.relpath(image_full_path,input_dir_base).replace('\\','/').lower()
    
    # Remove "ic*-images" from paths; these were redunant    
    destination_relative_path = destination_relative_path.replace('ic-images/', '').\
        replace('icr2-images/', '').\
        replace('icr3-images/', '')
        
    for k, v in island_name_mappings.items():
        destination_relative_path = destination_relative_path.replace(k, v)
    
    assert destination_relative_path not in output_file_names
    output_file_names.add(destination_relative_path)
    
    destination_full_path = os.path.join(output_dir_images, destination_relative_path)
    os.makedirs(os.path.dirname(destination_full_path),exist_ok=True)
    shutil.copy(image_full_path, destination_full_path)

# ...for each image

del output_file_names
print('\nFinished renaming IC images')


#%% Extract location and date/time information from filenames

sample_paths = [
    
    # Two folders deep
    r'ecuador1\cam1613\ecuador1_cam1613_20150101_055929_img_0043.jpg',
    r'palau\cam02a\cam02a12132018\palau_cam02a12132018_20180822_120000_rcnx0001.jpg',
    r'ecuador2\ic1603\ecuador2_ic1603_20150101_000058_img_0004.jpg',
    r'puertorico\2a\puertorico_2a_20141111_004833_img_0006.jpg',

    # Three folders deep    
    r'dominicanrepublic\camara01\cam0101noviembre2015\dominicanrepublic_cam0101noviembre2015_20151028_080024_sunp0001.jpg',
    r'micronesia\cam05\cam05april2019\micronesia_cam05april2019_20190411_190527_rcnx0001.jpg',
    
    # Three folders deep with a different timestamp format
    r'chile\filipiananbek\filipiananbek2013\chile_filipiananbek2013_0111201375113.jpg'
    ]

def parse_ic_relative_filename(relative_path):    
    """
    Takes a relative path in one of the three formats above and parses the location, date, and -
    if available - time.  Date and time are combined into a single string.
    
    return location,timestamp
    """
    
    relative_path = relative_path.replace('\\','/')
    
    for k in island_name_mappings.keys():
        assert k not in relative_path
        
    folders = relative_path.split('/')
    site = folders[1]
    if site.startswith('camara'):
        site = site.replace('camara','cam_')
    elif site.startswith('cam'):
        site = site.replace('cam','cam_')     
    else:
        site = 'cam_' + site
      
    filename = os.path.basename(relative_path)
    tokens = filename.split('_')
    
    # Country can be a country, or "ecuador1"
    #
    # Either way, the first folder and the first token in the basename should be the same
    country = folders[0]
    assert country == tokens[0]
    
    # Make sure we have a valid country designator
    assert country in island_name_mappings.values()

    location = country + '_' + site
    
    if country in ['ecuador1','palau','ecuador2','puertorico','dominicanrepublic','micronesia']:
        datestring = tokens[2]
        timestring = tokens[3]
        assert (len(datestring) == 8) and (len(timestring) == 6)
        timestamp = datetime.datetime.strptime(datestring + timestring,'%Y%m%d%H%M%S')
    elif country == 'chile':
        # I'm not sure how to parse time from this, so not trying...
        # 0111201375113
        datestring = tokens[2][0:8]
        timestamp = datetime.datetime.strptime(datestring,'%m%d%Y')
    else:
        raise ValueError('Unknown country {}'.format(country))
            
    return location,timestamp


#%% Test driver for location/date parsing
        
if False:
    
    #%%
    relative_path = sample_paths[0]
    for relative_path in sample_paths:
        location,timestamp = parse_ic_relative_filename(relative_path)
        print('Parsed {} to:\n{},{}\n'.format(relative_path,location,timestamp))
    
    
#%% Create CCT dictionaries

images = []
annotations = []

image_ids_to_annotations = defaultdict(list)

# image_ids_to_images = {}

category_name_to_category = {}

# Force the empty category to be ID 0
empty_category = {}
empty_category['name'] = 'empty'
empty_category['id'] = 0
category_name_to_category['empty'] = empty_category
next_category_id = 1

json_files = os.listdir(input_dir_json)

all_locations = set()
all_image_ids = set()

# json_file = json_files[0]
for json_file in json_files:

    print('Processing .json file {}'.format(json_file))
    
    # Example filename:
    #
    # IC_AI4Earth_2019_timelapse_Cabritos.json
    
    # Store the non-obfuscated folder name, we need this to check against the non-reviewed image list
    dataset_folder_original = json_file.split('.')[0].replace('IC_AI4Earth_2019_timelapse_', '').lower()
    
    # Otherwise we work on the obfuscated folder name
    dataset_folder = dataset_folder_original
    for k, v in island_name_mappings.items():
        dataset_folder = dataset_folder.replace(k, v)
    
    # Load .json annotations for this data set
    with open(os.path.join(input_dir_json, json_file), 'r') as f:
        data = f.read()        
    data = json.loads(data)
    
    categories_this_dataset = data['detection_categories']
    
    # i_entry = 0; entry = data['images'][i_entry]
    #
    # PERF: Not exactly trivially parallelizable, but about 100% of the 
    # time here is spent reading image sizes (which we need to do to get from 
    # absolute to relative coordinates), so worth parallelizing.
    for i_entry,entry in enumerate(tqdm(data['images'])):
        
        image_path_relative_to_dataset = entry['file']
        
        original_filename = os.path.join(dataset_folder_original,image_path_relative_to_dataset).lower().replace('\\','/')
        
        reviewed_image = (original_filename not in non_reviewed_filenames_set)
        
        image_relative_path = os.path.join(dataset_folder, image_path_relative_to_dataset).lower().replace('\\','/')
        
        for k, v in island_name_mappings.items():
            image_relative_path = image_relative_path.replace(k, v)
            
        assert image_relative_path.startswith(dataset_folder)
        
        # Generate a unique ID from the path
        image_id = image_relative_path.split('.')[0].replace(
            '\\', '/').replace('/', '_').replace(' ', '_')
        
        assert image_id not in all_image_ids
        all_image_ids.add(image_id)
        
        im = {}
        im['reviewed'] = reviewed_image
        im['id'] = image_id
        im['file_name'] = image_relative_path
        image_full_path = os.path.join(output_dir_images, image_relative_path)
        assert(os.path.isfile(image_full_path))
        
        pil_image = Image.open(image_full_path)        
        width, height = pil_image.size
        im['width'] = width
        im['height'] = height
    
        location,timestamp = parse_ic_relative_filename(relative_path)
        all_locations.add(location)
                
        images.append(im)
        
        detections = entry['detections']
        
        # detection = detections[0]
        for detection in detections:
            
            category_name = categories_this_dataset[detection['category']]
            category_name = category_name.strip().lower()            
            category_name = category_mapping.get(
                category_name, category_name)
            category_name = category_name.replace(' ','_')        
            
            # Have we seen this category before?
            if category_name in category_name_to_category:
                category_id = category_name_to_category[category_name]['id']
            else:
                category_id = next_category_id
                category = {}
                category['id'] = category_id
                print('Adding category {}'.format(category_name))
                category['name'] = category_name
                category_name_to_category[category_name] = category
                next_category_id += 1
            
            # Create an annotation
            ann = {}        
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = im['id']    
            ann['category_id'] = category_id
            
            if category_id != 0:
                ann['bbox'] = detection['bbox']
                # MegaDetector: [x,y,width,eight] (normalized, origin upper-left)
                # CCT: [x,y,width,height] (absolute, origin upper-left)
                # os.startfile(image_full_path)
                ann['bbox'][0] = ann['bbox'][0] * im['width']
                ann['bbox'][1] = ann['bbox'][1] * im['height']
                ann['bbox'][2] = ann['bbox'][2] * im['width']
                ann['bbox'][3] = ann['bbox'][3] * im['height']
            else:
                assert(detection['bbox'] == [0,0,0,0])
            annotations.append(ann)
            image_ids_to_annotations[im['id']].append(ann)
            
        # ...for each detection
    
    # ...for each image
            
# ...for each database
            
print('Finished creating CCT dictionaries')

# Remove non-reviewed images and associated annotations

#%%

reviewed_images = [im for im in images if im['reviewed']]

reviewed_image_ids = set([im['id'] for im in reviewed_images])

reviewed_annotations = [ann for ann in annotations if ann['image_id'] in reviewed_image_ids]

print('Found {} reviewed images (of {} total in .json files, {} total on disk)'.format(
    len(reviewed_images),len(images),len(image_full_paths)))
print('{} of {} annotations refer to reviewed images'.format(len(reviewed_annotations),
                                                             len(annotations)))

#%% Create info struct

info = dict()
info['year'] = 2020
info['version'] = 1.0
info['description'] = 'Island Conservation Camera Traps'
info['contributor'] = 'Conservation Metrics and Island Conservation'


#%% Write .json output

for im in images:
    del im['reviewed']
    
categories = list(category_name_to_category.values())

json_data = {}

json_data['images'] = reviewed_images
json_data['annotations'] = reviewed_annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_json_file, 'w'), indent=2)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
    len(images), len(annotations), len(categories)))


#%% Clean start

### Everything after this should work from a clean start ###


#%% Validate output

fn = output_json_file
options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = output_dir_images
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False
sorted_categories, data, errors = sanity_check_json_db.sanity_check_json_db(fn, options)


#%% Preview animal labels

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 3000
viz_options.trim_to_images_with_bboxes = False
viz_options.classes_to_exclude = ['empty','human']
# viz_options.classes_to_include = ['human']
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file, image_db = visualize_db.process_images(db_path=output_json_file,
                                                         output_dir=os.path.join(
                                                             output_dir_base, 'preview_animals'),
                                                         image_base_dir=output_dir_images,
                                                         options=viz_options)
os.startfile(html_output_file)


#%% Preview empty labels

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 3000
viz_options.trim_to_images_with_bboxes = False
# viz_options.classes_to_exclude = ['empty','human']
viz_options.classes_to_include = ['empty']
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file, image_db = visualize_db.process_images(db_path=output_json_file,
                                                         output_dir=os.path.join(
                                                             output_dir_base, 'preview_empty'),
                                                         image_base_dir=output_dir_images,
                                                         options=viz_options)
os.startfile(html_output_file)


#%% Load data back from .json (basically to facilitate re-starting from here)

with open(output_json_file,'r') as f:
    data = json.load(f)

images = data['images']
categories = data['categories']
annotations = data['annotations']

category_id_to_name = {cat['id']:cat['name']  for cat in categories}
image_ids_to_annotations = defaultdict(list)
for ann in annotations:
    image_ids_to_annotations[ann['image_id']].append(ann)
assert len(images) == len(image_ids_to_annotations)    


#%% Copy images out to human/non-human folders

category_id_to_name = {cat['id']:cat['name'] for cat in categories}

def copy_image_to_output_dir(im):
    
    image_relative_path = im['file_name']
    
    # Find all category names associated with this image
    assert im['id'] in image_ids_to_annotations
    image_cat_ids = [ann['category_id'] for ann in image_ids_to_annotations[im['id']]]
    image_cat_names = [category_id_to_name[cat_id] for cat_id in image_cat_ids]
    
    # Copy this image to the appropriate output folder (human or non-human)            
    if 'human' in image_cat_names:
        target_file = os.path.join(human_dir, image_relative_path)
    else:
        target_file = os.path.join(non_human_dir, image_relative_path)
    
    target_dir = os.path.dirname(target_file)
    os.makedirs(target_dir,exist_ok=True)
    
    source_file = os.path.join(output_dir_images, image_relative_path)
    assert os.path.isfile(source_file)
    
    # print('Copying {} to {}'.format(source_file,target_file))
    shutil.copy(source_file,target_file)
    
# im = images[0]
    
# Serial version    
if True:    
    for im in tqdm(images):
        copy_image_to_output_dir(im)

# Joblib version (slower than multiprocessing for me, whether I used processes or threads)
if False:
    from joblib import Parallel,delayed
    Parallel(n_jobs=50, prefer='processes')(
        delayed(copy_image_to_output_dir)(im) for im in images)

# Multiprocessing version
if False:
    from multiprocessing.pool import ThreadPool    
    pool = ThreadPool(100)
    results = list(tqdm(pool.imap(copy_image_to_output_dir, images)))
    print('Finished copying images')


#%% Create zipfiles for human/non-human folders

def zipdir(path, zipfilename, basepath=None):
    """
    Zip everything in [path] into [zipfilename], with paths in the zipfile relative to [basepath]
    """
    ziph = zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_STORED)

    try:
        for root, dirs, files in os.walk(path):            
            for file in files:
                src = os.path.join(root, file)
                if basepath is None:
                    dst = file
                else:
                    dst = os.path.relpath(src, basepath)
                ziph.write(src, dst, zipfile.ZIP_STORED)
    finally:
        ziph.close()

zipdir(human_dir,human_zipfile,human_dir)
zipdir(non_human_dir,non_human_zipfile,non_human_dir)

print('Done creating zipfiles')


#%% Scrap

if False:
    
    pass

    #%% Generate human-only .json file

    with open(output_json_file,'r') as f:
        data = json.load(f)
    
    images = data['images']
    categories = data['categories']
    annotations = data['annotations']
    
    human_cat_id = None
    for cat in categories:
        if cat['name'] == 'human':
            human_cat_id = cat['id']
    assert human_cat_id
    
    human_image_ids = set()
    human_annotations = []
    
    for ann in annotations:
        if ann['category_id'] == human_cat_id:
            human_image_ids.add(ann['image_id'])
            human_annotations.append(ann)
                
    human_images = [im for im in images if im['id'] in human_image_ids]
    
    json_data = {}
    json_data['images'] = human_images
    json_data['annotations'] = human_annotations
    json_data['categories'] = categories
    json_data['info'] = info
    output_json_file_human = output_json_file.replace('.json','.human.json')
    json.dump(json_data, open(output_json_file_human, 'w'), indent=2)
    
    print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images), len(annotations), len(categories)))
    
    
    ##%% Validate output
    
    fn = output_json_file_human
    options = sanity_check_json_db.SanityCheckOptions()
    options.baseDir = output_dir_images
    options.bCheckImageSizes = False
    options.bCheckImageExistence = False
    options.bFindUnusedImages = False
    sorted_categories, data, errors = sanity_check_json_db.sanity_check_json_db(fn, options)
    
    
    ##%% Preview labels
    
    viz_options = visualize_db.DbVizOptions()
    viz_options.num_to_visualize = 3000
    viz_options.trim_to_images_with_bboxes = False
    # viz_options.classes_to_exclude = ['empty']
    viz_options.classes_to_include = ['human']
    viz_options.add_search_links = False
    viz_options.sort_by_filename = False
    viz_options.parallelize_rendering = True
    html_output_file, image_db = visualize_db.process_images(db_path=output_json_file_human,
                                                             output_dir=os.path.join(
                                                                 output_dir_base, 'preview_human'),
                                                             image_base_dir=output_dir_images,
                                                             options=viz_options)
    os.startfile(html_output_file)
    

#
# island_conservation_to_json.py
#
# Convert the Island Conservation data set to a COCO-camera-traps .json file
#

#%% Constants and environment

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
import os
import csv
import json
import uuid
from PIL import Image
from tqdm import tqdm
import shutil
import zipfile
from path_utils import find_images

# Base directory for all input images and metadata
#
# Should contain folders "ic-images", "icr2-images", and "icr3-images"
input_dir_base = r'e:\island-conservation\20200529drop'

# Folder containing the input metadata files (unzipped from IC_AI4Earth_2019_timelapse_by_project.zip)
input_dir_json = os.path.join(
    input_dir_base, 'IC_AI4Earth_2019_timelapse_by_project')

island_name_mapping_file = os.path.join(input_dir_base,'island_name_mappings.csv')

assert(os.path.isdir(input_dir_base))
assert(os.path.isdir(input_dir_json))
assert(os.path.isfile(island_name_mapping_file))

output_dir_base = r'f:\data_staging\island-conservation'
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

category_mapping = {"BUOW": "burrowing owl",
                    "BAOW": "barred owl",
                    "GRHE": "green heron",
                    "AMKE": "american kestrel",
                    "GBHE": "great blue heron",
                    "BRNO": "brown noddy",
                    "YCNH": "yellow-crowned night heron",
                    "WWDO": "white-winged dove",
                    "SEOW": "short-eared owl"}

read_image_sizes = False


#%% Load island name mappings from .csv

island_name_mappings = None

with open(island_name_mapping_file, mode='r') as f:
    reader = csv.reader(f)
    island_name_mappings = {rows[0]:rows[1] for rows in reader}


#%% Support functions


def zipdir(path, zipfilename, basepath=None):
    """
    Zip everything in [path] into [zipfilename], with paths in the zipfile relative to [basepath]
    """
    ziph = zipfile.ZipFile(zipfilename, 'w', zipfile.ZIP_STORED)

    for root, dirs, files in os.walk(path):
        for file in files:
            src = os.path.join(root, file)
            if basepath is None:
                dst = file
            else:
                dst = os.path.relpath(src, basepath)
            ziph.write(src, dst, zipfile.ZIP_STORED)

    ziph.close()


#%% Enumerate input images
    
image_full_paths = find_images(input_dir_base, recursive=True)
print('Enumerated {} images from {}'.format(len(image_full_paths),input_dir_base))


#%% Rename files for obfuscation
    
output_file_names = set()

# image_full_path = image_full_paths[0]
for image_full_path in tqdm(image_full_paths):
    
    destination_relative_path = os.path.relpath(image_full_path,input_dir_base).replace('\\','/').lower()
    
    # Remove "ic*-images" from paths; these were redunant    
    destination_relative_path = destination_relative_path.replace('ic-images/', '').replace('icr2-images/', '').\
        replace('icr3-images/', '')
    for k, v in island_name_mappings.items():
        destination_relative_path = destination_relative_path.replace(k, v)
    
    assert destination_relative_path not in output_file_names
    output_file_names.add(destination_relative_path)
    
    destination_full_path = os.path.join(output_dir_images, destination_relative_path)
    os.makedirs(os.path.dirname(destination_full_path),exist_ok=True)
    shutil.copy(image_full_path, destination_full_path)

# ...for each image

print('Finished renaming IC images')


#%% Create CCT dictionaries

images = []
annotations = []
categories = []

# image_ids_to_images = {}

category_name_to_category = {}

# Force the empty category to be ID 0
empty_category = {}
empty_category['name'] = 'empty'
empty_category['id'] = 0
category_name_to_category['empty'] = empty_category
categories.append(empty_category)
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
    
    dataset_folder = json_file.split('.')[0].replace('IC_AI4Earth_2019_timelapse_', '').lower()
    for k, v in island_name_mappings.items():
        dataset_folder = dataset_folder.replace(k, v)
    
    # Load .json annotations for this data set
    with open(os.path.join(input_dir_json, json_file), 'r') as f:
        data = f.read()        
    data = json.loads(data)
    
    categories_this_dataset = data['detection_categories']
    
    # entry = data['images'][0]
    for entry in tqdm(data['images']):
        
        image_path_relative_to_dataset = entry['file']
        
        # E.g.:
        #
        # dominican_republic/camara02/cam0226junio2015/cabritos_cam0226junio2015_20131026_063520_sunp0022.jpg
        # dominican_republic/camara101/cam10118mayo2017/dominican_republic_cam10118mayo2017_20170425_175810_img_0019.jpg
        #
        image_relative_path = os.path.join(dataset_folder, image_path_relative_to_dataset).lower().replace('\\','/')
        
        for k, v in island_name_mappings.items():
            image_relative_path = image_relative_path.replace(k, v)
            
        assert image_relative_path.startswith(dataset_folder)
        
        image_id = image_relative_path.split('.')[0].replace(
            '\\', '/').replace('/', '_').replace(' ', '_')
        
        assert image_id not in all_image_ids
        all_image_ids.add(image_id)
        
        # E.g. cam0226junio2015_20131026_063520_sunp0022.jpg
        fn_without_location = os.path.basename(image_relative_path).replace(dataset_folder + '_','')
        
        # E.g. ['cam0226junio2015', '20131026', '063520', 'sunp0022']
        tokens = fn_without_location.split('_')
        
        assert(len(tokens) >= 4)
        
        im = {}
        im['id'] = image_id
        im['file_name'] = image_relative_path
        image_full_path = os.path.join(output_dir_images, image_relative_path)
        assert(os.path.isfile(image_full_path))
        
        if read_image_sizes:
            pil_image = Image.open(image_full_path)        
            width, height = pil_image.size
            im['width'] = width
            im['height'] = height
        
        location = dataset_folder + '_' + tokens[0]
        all_locations.add(location)
                
        assert(tokens[1].isdecimal())
        assert(tokens[2].isdecimal())
        timestamp = tokens[1] + '_' + tokens[2]
        
        images.append(im)
        
        detections = entry['detections']
        image_cats = []
        for detection in detections:
            category_name = categories_this_dataset[detection['category']]
            if category_name == 'NULL':
                category_name = 'empty'
            else:
                category_name = category_mapping.get(
                    category_name, category_name)
            category_name = category_name.strip().lower()
            image_cats.append(category_name)

            # Have we seen this category before?
            if category_name in category_name_to_category:
                category_id = category_name_to_category[category_name]['id']
            else:
                category_id = next_category_id
                category = {}
                category['id'] = category_id
                category['name'] = category_name
                category_name_to_category[category_name] = category
                categories.append(category)
                next_category_id += 1
            
            # Create an annotation
            ann = {}        
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = im['id']    
            ann['category_id'] = category_id
            ann['conf'] = detection['conf']
            if category_id != 0:
                ann['bbox'] = detection['bbox']
            else:
                assert(detection['bbox'] == [0,0,0,0])
            annotations.append(ann)

        # Copy this image to the appropriate output folder (human or non-human)            
        if 'person' in image_cats or 'human' in image_cats:
            shutil.copy(os.path.join(output_dir_images, image_relative_path), os.path.join(output_dir_base, human_dir))
        else:
            shutil.copy(os.path.join(output_dir_images, image_relative_path), os.path.join(output_dir_base, non_human_dir))

print('Finished creating CCT dictionaries')


#%% Create info struct

info = dict()
info['year'] = 2018
info['version'] = 1.0
info['description'] = 'Island Conservation Camera Traps'
info['contributor'] = 'Conservation Metrics and Island Conservation'


#%% Write .json output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_json_file, 'w'), indent=2)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
    len(images), len(annotations), len(categories)))


#%% Create ZIP files for human and non human

zipdir(human_dir,human_zipfile)
zipdir(non_human_dir,non_human_zipfile)


#%% Validate output

fn = output_json_file
options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = output_dir_images
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False
sorted_categories, data, errors = sanity_check_json_db.sanity_check_json_db(fn, options)


#%% Preview labels

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 1000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file, image_db = visualize_db.process_images(db_path=output_json_file,
                                                         output_dir=os.path.join(
                                                             output_dir_base, 'preview'),
                                                         image_output_dir_base=output_dir_images,
                                                         options=viz_options)
os.startfile(html_output_file)

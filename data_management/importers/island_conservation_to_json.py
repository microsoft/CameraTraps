#
# island_conservation_to_json_2017.py
#
# Convert the Island Conservation data set to a COCO-camera-traps .json file
#

#%% Constants and environment

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
import os
import json
import uuid
import time
import humanfriendly
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
import zipfile
from path_utils import find_images


src_directory = r'/mnt/blobfuse/wildlifeblobssc/20200529drop/'
base_dir = r'/home/gramener/island_conservation'
dest_directory = os.path.join(base_dir, 'images')
os.makedirs(dest_directory, exist_ok=True)
json_directory = os.path.join(
    base_dir, 'IC_AI4Earth_2019_timelapse_by_project')
output_json_file = os.path.join(base_dir, 'island_conservation.json')
assert(os.path.isdir(src_directory))
assert(os.path.isdir(dest_directory))

human_dir = os.path.join(base_dir, 'human')
non_human_dir = os.path.join(base_dir, 'non-human')

human_zipfile = os.path.join(base_dir, 'island_humans.zip')
non_human_zipfile = os.path.join(base_dir, 'island_nonhumans.zip')

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

island_mapping = {"Cabritos": "dominican_republic",
                  "Floreana": "ecuador",
                  "JFI": "chile",
                  "Mona": "puerto_rico",
                  "Ngeruktabel": "palau",
                  "SantaCruz": "ecuador",
                  "Ulithi": "micronesia",
                  "ULITHI": "micronesia"}

category_mapping = {"BUOW": "burrowing owl",
                    "BAOW": "barred owl",
                    "GRHE": "green heron",
                    "AMKE": "american kestrel",
                    "GBHE": "great blue heron",
                    "BRNO": "brown noddy",
                    "YCNH": "yellow-crowned night heron",
                    "WWDO": "white-winged dove",
                    "SEOW": "short-eared owl"}

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

def update_file_names():
    """ Function to update the filenames and copy to dest folder."""
    image_full_paths = find_images(src_directory, bRecursive=True)
    for img in image_full_paths:
        fullpath = img
        dest_name = os.path.basename(fullpath)
        img = img.replace(src_directory, '').replace('ic-images/', '').replace('icr2-images/', '').replace('icr3-images/', '')
        for k, v in island_mapping.items():
            img = img.replace(k, v)
            dest_name = dest_name.replace(k, v)
        assert not os.path.isabs(img)
        dstdir = os.path.join(dest_directory, os.path.dirname(img))
        try:
            os.makedirs(dstdir)
        except Exception as e:
            pass
        shutil.copy(fullpath, os.path.join(dstdir, dest_name))

print("START!! copying files to destination folder")
update_file_names()
print("DONE!! copying files to destination folder")

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


json_files = os.listdir(json_directory)
for json_file in json_files:
    folder = json_file.split(".")[0].replace("IC_AI4Earth_2019_timelapse_", "")
    for k, v in island_mapping.items():
        folder = folder.replace(k, v)
    with open(os.path.join(json_directory, json_file), 'r') as inp:
        data = inp.read()
    data = json.loads(data)
    tmp_cat = data['detection_categories']
    for entry in tqdm(data['images']):
        image_name = entry['file']
        image_name = os.path.join(folder, image_name)
        tmp = os.path.basename(os.path.join(
            dest_directory, image_name)).split(".")[0].split("_")
        try:
            assert(len(tmp) >= 2 and len(tmp) <= 6)
        except Exception:
            import pdb;pdb.set_trace()
        for k, v in island_mapping.items():
            image_name = image_name.replace(k, v)
        img_id = image_name.split(".")[0].replace(
            '\\', '/').replace('/', '_').replace(' ', '_')
        location = None
        timestamp = None
        im = {}
        im['id'] = img_id
        im['file_name'] = image_name
        tmp[0] = island_mapping[tmp[0]]
        if len(tmp) <= 3:
            location = "_".join(tmp[:2])
        elif len(tmp) > 3:
            location = "_".join(tmp[:2])
            timestamp = tmp[2]
        if timestamp:
            im['datetime'] = timestamp
        if location:
            im['location'] = location.lower()
        
        # image_ids_to_images[img_id] = im
        images.append(im)
        detections = entry['detections']
        image_cats = []
        for detection in detections:
            category_name = tmp_cat[detection['category']]
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
            annotations.append(ann)
        if "person" in image_cats or "human" in image_cats:
            shutil.copy(os.path.join(dest_directory, image_name), os.path.join(base_dir, human_dir))
        else:
            shutil.copy(os.path.join(dest_directory, image_name), os.path.join(base_dir, non_human_dir))

print('Finished creating CCT dictionaries')
# import pdb;pdb.set_trace()

#%% Create info struct

info = dict()
info['year'] = 2018
info['version'] = 1
info['description'] = 'Island Conservation'
info['contributor'] = 'CMI_manual'

#%% Write output

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
options.baseDir = dest_directory
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False
sortedCategories, data, errors = sanity_check_json_db.sanity_check_json_db(fn, options)


#%% Preview labels


viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 1000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file, image_db = visualize_db.process_images(db_path=output_json_file,
                                                         output_dir=os.path.join(
                                                             base_dir, 'preview'),
                                                         image_base_dir=dest_directory,
                                                         options=viz_options)
os.startfile(html_output_file)

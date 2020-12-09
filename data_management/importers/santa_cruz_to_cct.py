#
# santa_cruz_to_cct.py
#
# Convert the Santa Cruz data set to a COCO-camera-traps .json file
#

#%% Constants and environment

import os
import json
import uuid
import time
import humanfriendly
import numpy as np
from PIL import Image
from tqdm import tqdm
import urllib
import urllib.request
import glob
import ntpath
import PIL.ExifTags
import subprocess

# base_directory = r'D:\Projects\Microsoft\Santa_Cruz\SCI Cameratrap Samasource Labels'
base_directory = r'/home/Vardhan/SCI Cameratrap Samasource Labels'
output_file = os.path.join(base_directory,'santa_cruz.json')
image_directory = os.path.join(base_directory,'images')
os.makedirs(image_directory,exist_ok=True)


#%% Support functions

def download_image(url):
    """
    Download the image from the URL.
    """
    print(url)
    path = urllib.parse.urlparse(url).path
    sub_dir, filename = os.path.split(path)
    sub_directory = os.path.join(image_directory, sub_dir[1:])
    if not os.path.isdir(sub_directory):
        os.makedirs(sub_directory)
    fullfilename = os.path.join(sub_directory, filename)
    print(fullfilename)
    if not os.path.exists(fullfilename):
        urllib.request.urlretrieve(url, fullfilename)
    return fullfilename, sub_dir[1:]


def get_bbox(coords):
    """
    Derive the bounding boxes from the provided coordinates
    """
    x = coords[0][0]
    y = coords[0][1]
    h = coords[2][1] - coords[0][1]
    w = coords[1][0] - coords[0][0]
    return [x, y, w, h]


def proces_makernotes(file_path):
    """
    Get MakerNotes EXIF data for an image
    """
    proc = subprocess.Popen(['exiftool', '-G', file_path],stdout=subprocess.PIPE, encoding='utf8')
    maker_notes = {}
    date_present = False
    while True:
        line = proc.stdout.readline()
        line = line.strip()
        if not line:
            break
        if line.startswith('[MakerNotes]'):
            if "Sequence" in line:
                seq = line.split(": ")
                seq_id, seq_num_frames = seq[1].split(" of ")
                maker_notes['seq_id'] = seq_id
                maker_notes['seq_num_frames'] = seq_num_frames
            if "Serial Number" in line:
                location = line.split(": ")[1]
                maker_notes['location'] = location
            if "Time" in line:
                datetime = line.split(": ")[1]
                maker_notes['datetime'] = datetime
                date_present = True
        if not date_present:
            if ("DateTime Original" in line or "Date/Time Created" in line or "Date/Time Original" in line):
                datetime = line.split(": ")[1]
                maker_notes['datetime'] = datetime
                date_present = True
    for each in ['seq_id', 'seq_num_frames', 'location', 'datetime']:
        if not each in list(maker_notes.keys()):
            maker_notes[each] = None
    return maker_notes


#%% Create CCT dictionaries

images = []
annotations = []

# Map categories to integer IDs (that's what COCO likes)
nextCategoryID = 1
categoriesToCategoryId = {}
categoriesToCategoryId['empty'] = 0
categoriesToCounts = {}
categoriesToCounts['empty'] = 0

# For each image
#
# Because in practice images are 1:1 with annotations in this data set,
# this is also a loop over annotations.

startTime = time.time()
json_files = []
for folder in os.listdir(base_directory):
    for file in glob.glob(base_directory+"/"+folder+'**/*.json'):
        json_files.append(file)
    for file in glob.glob(base_directory+"/"+folder+'**/**/*.json'):
        json_files.append(file)

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        for each in data:
            if each['url'].endswith("DS_Store"):
                continue
            file_path, sub_directory = download_image(each['url'])
            contains_human = False
            im = {}
            im['id'] = str(uuid.uuid1()) #filename.split('.')[0]
            filename = ntpath.basename(file_path)
            im['file_name'] = os.path.join(sub_directory, filename)
            # Check image height and width
            pilImage = Image.open(file_path)
            width, height = pilImage.size
            im['width'] = width
            im['height'] = height
            maker_notes = proces_makernotes(file_path)
            im['seq_id'] = maker_notes['seq_id']
            im['seq_num_frames'] = maker_notes['seq_num_frames']
            im['datetime'] = maker_notes['datetime']
            im['location'] = maker_notes['location']
            print(im)
            images.append(im)
            categories_this_image = set()
            if each['Output']:
                for category in each['Output']:
                    if 'Object' in list(category['tags']):
                        category_name = category['tags']['Object']
                    else:
                        category_name = 'empty'
                    category_name = category_name.strip().lower()
                    categories_this_image.add(category_name)
            
                    # Have we seen this category before?
                    if category_name in categoriesToCategoryId:
                        categoryID = categoriesToCategoryId[category_name]
                        categoriesToCounts[category_name] += 1
                    else:
                        categoryID = nextCategoryID
                        categoriesToCategoryId[category_name] = categoryID
                        categoriesToCounts[category_name] = 0
                        nextCategoryID += 1
                    # Create an annotation
                    ann = {}
                    
                    ann['id'] = str(uuid.uuid1())
                    ann['image_id'] = im['id']    
                    ann['category_id'] = categoryID
                    ann["sequence_level_annotation"] = False
                    ann['bbox'] = get_bbox(category['points'])
                    annotations.append(ann)

# # Convert categories to a CCT-style dictionary

categories = []

for category in categoriesToCounts:
    print('Category {}, count {}'.format(category, categoriesToCounts[category]))
    categoryID = categoriesToCategoryId[category]
    cat = {}
    cat['name'] = category
    cat['id'] = categoryID
    categories.append(cat)    
    
elapsed = time.time() - startTime
print('Finished creating CCT dictionaries in {}'.format(
      humanfriendly.format_timespan(elapsed)))

#%% Create info struct

info = {}
info['year'] = 2020
info['version'] = 1.0
info['description'] = 'Cameratrap data collected from the Channel Islands, California'
info['secondary_contributor'] = 'Converted to COCO .json by Vardhan Duvvuri'
info['contributor'] = 'The Nature Conservancy of California'


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_file, 'w'), indent=2)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))




#%% Validate output

from data_management.databases import sanity_check_json_db

fn = output_file
options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = image_directory
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True
    
sortedCategories, data, error = sanity_check_json_db.sanity_check_json_db(fn,options)


#%% Preview labels

from visualization import visualize_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = None
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file,image_db = visualize_db.process_images(db_path=output_file,
                                                        output_dir=os.path.join(base_directory,'preview'),
                                                        image_base_dir=image_directory,
                                                        options=viz_options)
# os.startfile(html_output_file)
import sys, subprocess
opener = "open" if sys.platform == "darwin" else "xdg-open"
subprocess.call([opener, html_output_file])

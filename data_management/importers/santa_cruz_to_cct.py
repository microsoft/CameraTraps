#
# santa_cruz_to_cct.py
#
# Convert the Santa Cruz data set to a COCO-camera-traps .json file
#
# Uses the command-line tool ExifTool (exiftool.org/) to pull EXIF tags from images,
# because every Python package we tried failed to pull the "Maker Notes" field properly.
# Consequently, we assume that exiftool (or exiftool.exe) is on the system path.
#

#%% Imports

import os
import json
import uuid
import time
import humanfriendly
import urllib
import urllib.request
import glob
import ntpath
import subprocess

from shutil import which
from PIL import Image
from tqdm import tqdm


#%% Constants

required_input_annotation_fields = set([
    'task_id','batch_id','name','url','Object','Output','teams','task_url','Step A Agent'
    ])


#%% Path setup

input_base = r'e:\santa-cruz-in'
input_annotation_folder = os.path.join(input_base,'SCI Cameratrap Samasource Labels')
input_image_folder = os.path.join(input_base,'images')

assert os.path.isdir(input_base)
assert os.path.isdir(input_annotation_folder)
assert os.path.isdir(input_image_folder)
assert not input_annotation_folder.endswith('/')

output_base = r'e:\santa-cruz-out'
output_file = os.path.join(output_base,'santa_cruz_camera_traps.json')
output_image_folder = os.path.join(output_base,'images')

os.makedirs(output_base,exist_ok=True)
os.makedirs(output_image_folder,exist_ok=True)


# Confirm that exiftool is available
assert which('exiftool') is not None, 'Could not locate the ExifTool executable'


#%% Load information from every .json file

json_files = glob.glob(input_annotation_folder+'/**/*.json', recursive=True)
print('Found {} .json files'.format(len(json_files)))

# Ignore the sample file
sample_files = [fn for fn in json_files if 'sample' in fn]
assert len(sample_files) == 1

json_files = [fn for fn in json_files if 'sample' not in fn]
input_annotations = []

json_basenames = set()

# json_file = json_files[0]
for json_file in tqdm(json_files):
    
    json_filename = os.path.basename(json_file)
    assert json_filename not in json_basenames
    json_basenames.add(json_filename)
    
    with open(json_file,'r') as f:        
        annotations = json.load(f)
                
    # ann = annotations[0]
    for ann in annotations:
        
        assert isinstance(ann,dict)
        ann_keys = set(ann.keys())
        assert required_input_annotation_fields == ann_keys
        ann['json_filename'] = json_filename
        input_annotations.append(ann)
        
    # ...for each annotation in this file

# ...for each .json file
        
print('Loaded {} annotations from {} .json files'.format(len(input_annotations),len(json_files)))

image_urls = [ann['url'] for ann in input_annotations]


#%% Download files (functions)

# https://www.quickprogrammingtips.com/python/how-to-download-multiple-files-concurrently-in-python.html
import requests
from multiprocessing.pool import ThreadPool
from urllib.parse import urlparse

def download_relative_url(url,overwrite=False):
    """
    Download:
        
    https://somestuff.com/my/relative/path/image.jpg
    
    ...to:
        
    [input_image_folder]/my/relative/path/image.jpg
    """
    
    parsed_url = urlparse(url)
    relative_path = parsed_url.path
    
    # This is returned with a leading slash, remove it
    relative_path = relative_path[1:]
    
    target_file = os.path.join(input_image_folder,relative_path).replace('\\','/')
    
    if os.path.isfile(target_file and not overwrite):
        print('{} exists, skipping'.format(target_file))
        return url
    
    os.makedirs(os.path.dirname(target_file),exist_ok=True)  
    
    print('Downloading {} to {}'.format(url, target_file))
    
    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(target_file, 'wb') as f:
            for data in r:
                f.write(data)
    return url
 
def download_relative_urls(urls,n_threads = 10) :    
    
    if n_threads == 1:    
        results = []
        for url in urls:
            results.append(download_relative_url(url))
    else:
        results = ThreadPool(n_threads).imap(download_relative_url, urls)        
    return results
    

#%% Download files (execution)
    
download_relative_urls(image_urls)


#%% Read EXIF data (functions)

def process_makernotes(file_path):
    """
    Get MakerNotes EXIF data for an image
    """
    
    proc = subprocess.Popen(['exiftool', '-G', file_path],stdout=subprocess.PIPE, encoding='utf8')
    exif_lines = proc.stdout.readlines()
    exif_lines = [s.strip() for s in exif_lines]
    
    maker_notes = {}
    date_present = False
    
    for line in exif_lines:
        if not line:
            break
        if line.startswith('[MakerNotes]'):
            if 'Sequence' in line:
                seq = line.split(": ")
                seq_id, seq_num_frames = seq[1].split(" of ")
                maker_notes['seq_id'] = seq_id
                maker_notes['seq_num_frames'] = seq_num_frames
            if 'Serial Number' in line:
                location = line.split(': ')[1]
                maker_notes['location'] = location
            if 'Time' in line:
                datetime = line.split(': ')[1]
                maker_notes['datetime'] = datetime
                date_present = True
        if not date_present:
            if ('DateTime Original' in line or 'Date/Time Created' in line or 'Date/Time Original' in line):
                datetime = line.split(": ")[1]
                maker_notes['datetime'] = datetime
                date_present = True
                
    for required_field in ['seq_id', 'seq_num_frames', 'location', 'datetime']:
        
        if not required_field in list(maker_notes.keys()):
            if required_field == 'location':
                maker_notes[required_field] = ''
            else:
                maker_notes[required_field] = None
                
    return maker_notes


#%% Read EXIF data (execution)


#%% Create output filenames for each image
    

#%% Support functions

def download_image(url):
    """
    Download the image from the URL
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

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        for each in data:
            if each['url'].endswith("DS_Store") or each['url'].endswith("dropbox.device"):
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
                    if category_name == 'none':
                        category_name = 'empty'

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
info['description'] = 'Camera trap data collected from the Channel Islands, California'
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

#
# channel_islands_to_cct.py
#
# Convert the Channel Islands data set to a COCO-camera-traps .json file
#
# Uses the command-line tool ExifTool (exiftool.org) to pull EXIF tags from images,
# because every Python package we tried failed to pull the "Maker Notes" field properly.
#

#%% Imports, constants, paths

# Imports

import os
import json
import uuid
import time
import datetime
import humanfriendly
import glob
import ntpath
import subprocess
import requests

from multiprocessing.pool import ThreadPool
from urllib.parse import urlparse
from PIL import Image
from tqdm import tqdm


# Constants

required_input_annotation_fields = set([
    'task_id','batch_id','name','url','Object','Output','teams','task_url','Step A Agent'
    ])

n_download_threads = 10
n_exif_threads = 20


# Paths

input_base = r'e:\santa-cruz-in'
output_base = r'g:\santa-cruz-out'
exiftool_command_name = r'c:\exiftool-12.13\exiftool(-k).exe'

input_annotation_folder = os.path.join(input_base,'SCI Cameratrap Samasource Labels')
input_image_folder = os.path.join(input_base,'images')

assert os.path.isdir(input_base)
assert os.path.isdir(input_annotation_folder)
assert os.path.isdir(input_image_folder)
assert not input_annotation_folder.endswith('/')

output_file = os.path.join(output_base,'santa_cruz_camera_traps.json')
output_image_folder = os.path.join(output_base,'images')

os.makedirs(output_base,exist_ok=True)
os.makedirs(output_image_folder,exist_ok=True)


# Confirm that exiftool is available
# assert which(exiftool_command_name) is not None, 'Could not locate the ExifTool executable'
assert os.path.isfile(exiftool_command_name), 'Could not locate the ExifTool executable'

parsed_input_file = os.path.join(output_base,'parsed_input.json')


#%% Load information from every .json file

json_files = glob.glob(input_annotation_folder+'/**/*.json', recursive=True)
print('Found {} .json files'.format(len(json_files)))

# Ignore the sample file... actually, first make sure there is a sample file
sample_files = [fn for fn in json_files if 'sample' in fn]
assert len(sample_files) == 1

# ...and now ignore that sample file.
json_files = [fn for fn in json_files if 'sample' not in fn]
input_images = []

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
        input_images.append(ann)
        
    # ...for each annotation in this file

# ...for each .json file
        
print('\nLoaded {} image records from {} .json files'.format(len(input_images),len(json_files)))

image_urls = [ann['url'] for ann in input_images]


#%% Confirm URL uniqueness, handle redundant tags

output_images = []

urls = set()
for im in tqdm(input_images):
    url = im['url']
    if url in urls:        
        for existing_im in input_images:
            # Have we already added this image?
            if url == existing_im['url']:
                
                # One .json file was basically duplicated, but as:
                #
                # Ellie_2016-2017 SC12.json
                # Ellie_2016-2017-SC12.json                
                assert im['json_filename'].replace('-',' ') == existing_im['json_filename'].replace('-',' ')
                
                # If the new image has no output, just leave the old one there
                if im['Output'] is None:
                    print('Warning: duplicate URL {}, keeping existing output'.format(url))                    
                    break
                
                # If the old image has no output, and the new one has output, default to the one with output
                if (existing_im['Output'] is None) and (im['Output'] is not None): 
                    print('Warning: duplicate URL {}, adding new output'.format(url))
                    existing_im['Output'] = im['Output']
                    break
                
                else:
                    # Don't worry about the cases where someone tagged 'fox' and someone tagged 'fox_partial'
                    obj1 = im['Output'][0]['tags']['Object'].replace('_partial','')
                    obj2 = existing_im['Output'][0]['tags']['Object'].replace('_partial','')
                    if obj1 != obj2:
                        print('Warning: image {} tagged with {} and {}'.format(url,obj1,obj2))
    else:
        urls.add(url)
        output_images.append(im)

print('Kept {} of {} annotation records'.format(len(output_images),len(input_images)))

images = output_images


#%% Save progress

with open(parsed_input_file,'w') as f:
    json.dump(images,f,indent=1)
    
if False:
    #%%
    with open(parsed_input_file,'r') as f:
        images = json.load(f)
        

#%% Download files (functions)

# https://www.quickprogrammingtips.com/python/how-to-download-multiple-files-concurrently-in-python.html

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
    else:
        print('Warning: failed to download {}'.format(url))
        
    return url
 
def download_relative_urls(urls,n_threads = n_download_threads):    
    
    if n_threads == 1:    
        results = []
        for url in urls:
            results.append(download_relative_url(url))
    else:
        results = ThreadPool(n_threads).map(download_relative_url, urls)        
    return results
    

#%% Download files (execution)
    
download_relative_urls(image_urls)


#%% Read required fields from EXIF data (function)

def process_exif(file_path):
    """
    Get relevant fields from EXIF data for an image
    """
    
    # -G means "Print group name for each tag", e.g. print:
    #
    # [File]          Bits Per Sample                 : 8
    #
    # ...instead of:
    #
    # Bits Per Sample                 : 8
    
    proc = subprocess.Popen([exiftool_command_name, '-G', file_path], stdout=subprocess.PIPE, encoding='utf8')
    exif_lines = proc.stdout.readlines()    
    exif_lines = [s.strip() for s in exif_lines]
    assert exif_lines is not None and len(exif_lines) > 0, 'Failed to read EXIF data from {}'.format(file_path)
    
    exif_tags = {}
    
    # line_raw = exif_lines[0]
    for line_raw in exif_lines:
        
        line = line_raw.lower()  
        
        # Split on the first occurrence of ":"
        tokens = line.split(':',1)
        
        assert(len(tokens) == 2)
        field_name = tokens[0].strip()
        field_value = tokens[1].strip()
        
        if field_name.startswith('[makernotes]'):
            
            if 'sequence' in field_name:
                # Typically:
                #
                # '[MakerNotes]    Sequence                        ', '1 of 3']
                frame_num, seq_num_frames = field_value.split('of')
                exif_tags['frame_num'] = int(frame_num.strip())
                exif_tags['seq_num_frames'] = int(seq_num_frames.strip())
            
            # Not a typo; we are using serial number as a location
            elif 'serial number' in line:
                exif_tags['location'] = field_value            

            elif ('date/time original' in line):
                
                previous_dt = None
                
                if 'datetime' in exif_tags:
                    previous_dt = exif_tags['datetime']
                dt = datetime.datetime.strptime(field_value,'%Y:%m:%d %H:%M:%S')
                
                # If there are multiple timestamps, make sure they're *almost* the same
                if previous_dt is not None:
                    delta = abs((dt-previous_dt).total_seconds())
                    assert delta < 1.0
                    
                exif_tags['datetime'] = dt
                
            if False:            
                if 'time' in line:
                    assert 'datetime' not in exif_tags
                    exif_tags['datetime'] = field_value
                    
        if ('datetime original' in line) or ('create date' in line) or ('date/time created' in line) or ('date/time original' in line):
            
            previous_dt = None
            
            if 'datetime' in exif_tags:
                previous_dt = exif_tags['datetime']
            dt = datetime.datetime.strptime(field_value,'%Y:%m:%d %H:%M:%S')
            
            # If there are multiple timestamps, make sure they're *almost* the same
            if previous_dt is not None:
                delta = abs((dt-previous_dt).total_seconds())
                assert delta < 1.0
                
            exif_tags['datetime'] = dt
                
        if 'image width' in line:
            exif_tags['width'] = int(field_value)
        
        if 'image height' in line:
            exif_tags['height'] = int(field_value)
        
        if 'temperature' in line and not 'fahrenheit' in line:
            exif_tags['temperature'] = field_value
                            
    for required_field in ['frame_num', 'seq_num_frames', 'location', 'datetime', 'temperature']:
        assert required_field in exif_tags, 'File {} missing field {}'.format(file_path,required_field)
    
    return exif_tags


def add_exif_data(im):
    
    url = im['url']
    try:
        parsed_url = urlparse(url)
        relative_path = parsed_url.path
        
        # This is returned with a leading slash, remove it
        relative_path = relative_path[1:]    
        
        input_image_path = os.path.join(input_image_folder,relative_path).replace('\\','/')
        assert os.path.isfile(input_image_path)
        exif_tags = process_exif(input_image_path)
        im['exif_tags'] = exif_tags
    except Exception as e:
        s = 'Error on {}: {}'.format(url,str(e))
        print(s)
        return s
    return None



#%% Read EXIF data (execution)

if n_exif_threads == 1:        
    # ann = images[0]
    for im in tqdm(images):
        add_exif_data(im)
else:
    pool = ThreadPool(n_exif_threads)
    r = list(tqdm(pool.imap(add_exif_data, images), total=len(images)))


#%% Check for EXIF read errors


#%% Find unique locations
    
locations = set()

for ann in tqdm(images):

    assert 'exif_tags' in ann
    location = ann['exif_tags']['location']
    assert location is not None and len(location) > 0
    locations.add(location)
        
    
#%% Synthesize sequence information        

print('Found {} locations'.format(len(locations)))

locations = list(locations)

sequences = set()
images = images
max_seconds_within_sequence = 10

# Sort images by time within each folder
# i_location=0; location = locations[i_location]
for i_location,location in enumerate(locations):
    
    images_this_location = [im for im in images if im['exif_tags']['location'] == location]
    sorted_images_this_location = sorted(images_this_location, key = lambda im: im['exif_tags']['datetime'])
    
    current_sequence_id = None
    next_sequence_index = 0
    previous_datetime = None
        
    # previous_datetime = sorted_images_this_location[0]['datetime']
    # im = sorted_images_this_camera[1]
    for im in sorted_images_this_location:
        
        if previous_datetime is None:
            delta = None
        else:
            delta = (im['datetime'] - previous_datetime).total_seconds()
        
        # Start a new sequence if necessary
        if delta is None or delta > max_seconds_within_sequence:
            next_sequence_index = 0
            current_sequence_id = str(uuid.uuid4())
            sequences.add(current_sequence_id)
            
        im['seq_id'] = current_sequence_id
        assert im['exif_tags']['frame_num'] == next_sequence_index + 1
        next_sequence_index = next_sequence_index + 1
        previous_datetime = im['datetime']
    
    # ...for each image in this location

# ...for each location

print('Created {} sequences from {} images'.format(len(sequences),len(images)))

# Double-check seq_num_frames
num_frames_per_sequence = {}
for seq_id in sequences:
    images_this_sequence = [im for im in images if im['seq_id'] == seq_id]
    num_frames_per_sequence[seq_id] = len(images_this_sequence)
    for im in images_this_sequence:
        assert im['exif_tags']['seq_num_frames'] == len(images_this_sequence)

    
#%% Create output filenames for each image
    
# Handle partials
        
#%% Support functions

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

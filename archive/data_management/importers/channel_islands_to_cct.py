#
# channel_islands_to_cct.py
#
# Convert the Channel Islands data set to a COCO-camera-traps .json file
#
# Uses the command-line tool ExifTool (exiftool.org) to pull EXIF tags from images,
# because every Python package we tried failed to pull the "Maker Notes" field properly.
#

#%% Imports, constants, paths

## Imports ##

import os
import json
import uuid
import datetime
import glob
import subprocess
import requests
import shutil

from multiprocessing.pool import ThreadPool
from collections import defaultdict 
from urllib.parse import urlparse
from tqdm import tqdm
from PIL import Image


## Constants ##

required_input_annotation_fields = set([
    'task_id','batch_id','name','url','Object','Output','teams','task_url','Step A Agent'
    ])

n_download_threads = 10
n_exif_threads = 20
n_copy_threads = n_exif_threads


## Paths ##

input_base = r'e:\channel-islands-in'
output_base = r'g:\channel-islands-out'
exiftool_command_name = r'c:\exiftool-12.13\exiftool(-k).exe'

input_annotation_folder = os.path.join(input_base,'SCI Cameratrap Samasource Labels')
input_image_folder = os.path.join(input_base,'images')

assert os.path.isdir(input_base)
assert os.path.isdir(input_annotation_folder)
assert os.path.isdir(input_image_folder)
assert not input_annotation_folder.endswith('/')

output_file = os.path.join(output_base,'channel_islands_camera_traps.json')
output_image_folder = os.path.join(output_base,'images')
output_image_folder_humans = os.path.join(output_base,'human_images')

os.makedirs(output_base,exist_ok=True)
os.makedirs(output_image_folder,exist_ok=True)
os.makedirs(output_image_folder_humans,exist_ok=True)

# Confirm that exiftool is available
# assert which(exiftool_command_name) is not None, 'Could not locate the ExifTool executable'
assert os.path.isfile(exiftool_command_name), 'Could not locate the ExifTool executable'

parsed_input_file = os.path.join(output_base,'parsed_input.json')
exif_load_results_file = os.path.join(output_base,'exif_load_results.json')
sequence_info_results_file = os.path.join(output_base,'sequence_info_results.json')


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
                        
        # ...for each image we've already added
                        
    else:
        
        urls.add(url)
        output_images.append(im)
        
    # ...if this URL is/isn't in the list of URLs we've already processed

# ...for each image
        
print('Kept {} of {} annotation records'.format(len(output_images),len(input_images)))

images = output_images


#%% Save progress

with open(parsed_input_file,'w') as f:
    json.dump(images,f,indent=1)

#%%
    
if False:
    
    #%%
    
    with open(parsed_input_file,'r') as f:
        images = json.load(f)
        assert not any(['exif_tags' in im for im in images])
        

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
    """
    Download all URLs in [urls]
    """
    if n_threads == 1:    
        results = []
        for url in urls:
            results.append(download_relative_url(url))
    else:
        results = ThreadPool(n_threads).map(download_relative_url, urls)        
    return results
    

#%% Download files (execution)
    
download_relative_urls(image_urls)


#%% Read required fields from EXIF data (functions)

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
    
    # If we don't get any EXIF information, this probably isn't an image
    assert any([s.lower().startswith('[exif]') for s in exif_lines])
    
    exif_tags = {}
    
    found_makernotes = False
    
    # line_raw = exif_lines[0]
    for line_raw in exif_lines:
        
        line = line_raw.lower()  
        
        # Split on the first occurrence of ":"
        tokens = line.split(':',1)
        
        assert(len(tokens) == 2)
        field_name = tokens[0].strip()
        field_value = tokens[1].strip()
        
        if field_name.startswith('[makernotes]'):
            
            found_makernotes = True
            
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

            elif ('date/time original' in line and '[file]' not in line and '[composite]' not in line):
                
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
                    
        if (('datetime original' in line) or ('create date' in line) or ('date/time created' in line) or ('date/time original' in line)) \
            and ('[file]' not in line) and ('[composite]' not in line):
            
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
    
    # ...for each line in the exiftool output
            
    makernotes_fields = ['frame_num', 'seq_num_frames', 'location', 'temperature']
    
    if not found_makernotes:        
        
        print('Warning: could not find maker notes in {}'.format(file_path))
        
        # This isn't directly related to the lack of maker notes, but it happens that files that are missing
        # maker notes also happen to be missing EXIF date information
        if not 'datetime' in exif_tags:
            print('Warning: could not find datetime information in {}'.format(file_path))
        
        for field_name in makernotes_fields:
            assert field_name not in exif_tags
            exif_tags[field_name] = 'unknown'
            
    else:
        
        assert 'datetime' in exif_tags, 'Could not find datetime information in {}'.format(file_path)    
        for field_name in makernotes_fields:
            assert field_name in exif_tags, 'Could not find {} in {}'.format(field_name,file_path)
    
    return exif_tags

# ...process_exif()
    

def get_image_local_path(im):
    
    url = im['url']
    parsed_url = urlparse(url)
    relative_path = parsed_url.path
    
    # This is returned with a leading slash, remove it
    relative_path = relative_path[1:]    
    
    absolute_path = os.path.join(input_image_folder,relative_path).replace('\\','/')
    return absolute_path

    
def add_exif_data(im, overwrite=False):
    
    if ('exif_tags' in im) and (overwrite==False):
        return None
    
    url = im['url']
    
    # Ignore non-image files
    if url.lower().endswith('ds_store') or ('dropbox.device' in url.lower()):
        im['exif_tags'] = None
        return
    
    try:
        input_image_path = get_image_local_path(im)
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
    exif_read_results = list(tqdm(pool.imap(add_exif_data, images), total=len(images)))


#%% Save progress

with open(exif_load_results_file,'w') as f:
    
    # Use default=str to handle datetime objects
    json.dump(images, f, indent=1, default=str)

#%% 
    
if False:
    
    #%%
    
    with open(exif_load_results_file,'r') as f:
        # Not deserializing datetimes yet, will do this if I actually need to run this
        images = json.load(f)
        
    
#%% Check for EXIF read errors

for i_result,result in enumerate(exif_read_results):
    
    if result is not None:
                
        print('\nError found on image {}: {}'.format(i_result,result))
        im = images[i_result]
        file_path = get_image_local_path(im)
        assert images[i_result] == im
        result = add_exif_data(im)
        assert result is None
        print('\nFixed!\n')
        exif_read_results[i_result] = result
        
        
#%% Remove junk
        
images_out = []
for im in images:
    
    url = im['url']
    
    # Ignore non-image files
    if ('ds_store' in url.lower()) or ('dropbox.device' in url.lower()):
        continue
    images_out.append(im)
    
images = images_out
   

#%% Fill in some None values 

# ...so we can sort by datetime later, and let None's be sorted arbitrarily

for im in images:
    if 'exif_tags' not in im:
        im['exif_tags'] = None
    if 'datetime' not in im['exif_tags']:
        im['exif_tags']['datetime'] = None
        
images = sorted(images, key = lambda im: im['url'])
                

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
sequence_to_images = defaultdict(list) 
images = images
max_seconds_within_sequence = 10

# Sort images by time within each location
# i_location=0; location = locations[i_location]
for i_location,location in enumerate(locations):
    
    images_this_location = [im for im in images if im['exif_tags']['location'] == location]
    sorted_images_this_location = sorted(images_this_location, key = lambda im: im['exif_tags']['datetime'])
    
    current_sequence_id = None
    next_frame_number = 0
    previous_datetime = None
        
    # previous_datetime = sorted_images_this_location[0]['datetime']
    # im = sorted_images_this_camera[1]
    for i_image,im in enumerate(sorted_images_this_location):

        # Timestamp for this image, may be None
        dt = im['exif_tags']['datetime']
        
        # Start a new sequence if:
        #
        # * This image has no timestamp
        # * This image has a frame number of zero
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


#%% Count frames in each sequence
        
print('Created {} sequences from {} images'.format(len(sequences),len(images)))

num_frames_per_sequence = {}
for seq_id in sequences:
    # images_this_sequence = [im for im in images if im['seq_id'] == seq_id]
    images_this_sequence = sequence_to_images[seq_id]
    num_frames_per_sequence[seq_id] = len(images_this_sequence)
    for im in images_this_sequence:
        im['synthetic_seq_num_frames'] = len(images_this_sequence)

    
#%% Create output filenames for each image, store original filenames
        
images_per_folder = 1000
output_paths = set()

# i_location = 0; location = locations[i_location]
for i_location,location in enumerate(locations):
    
    images_this_location = [im for im in images if im['exif_tags']['location'] == location]
    sorted_images_this_location = sorted(images_this_location, key = lambda im: im['exif_tags']['datetime'])

    # i_image = 0; im = sorted_images_this_location[i_image]
    for i_image,im in enumerate(sorted_images_this_location):
    
        url = im['url']
        parsed_url = urlparse(url)
        relative_path = parsed_url.path
        relative_path = relative_path[1:]
        im['original_relative_path'] = relative_path
        image_id = uuid.uuid1()
        im['id'] = image_id
        folder_number = i_image // images_per_folder
        image_number = i_image % images_per_folder
        output_relative_path = 'loc-' + location + '/' + '{0:03d}'.format(folder_number) + '/' + '{0:03d}'.format(image_number) + '.jpg'
        im['output_relative_path'] = output_relative_path
        assert output_relative_path not in output_paths
        output_paths.add(output_relative_path)
        
assert len(output_paths) == len(images)


#%% Save progress

with open(sequence_info_results_file,'w') as f:
    
    # Use default=str to handle datetime objects
    json.dump(images, f, indent=1, default=str)

#%% 
    
if False:
    
    #%%
    
    with open(sequence_info_results_file,'r') as f:
        images = json.load(f)
        
        
#%% Copy images to their output files (functions)

def copy_image_to_output(im):
    
    source_path = os.path.join(input_image_folder,im['original_relative_path'])
    assert(os.path.isfile(source_path))
    dest_path = os.path.join(output_image_folder,im['output_relative_path'])
    os.makedirs(os.path.dirname(dest_path),exist_ok=True)
    shutil.copyfile(source_path,dest_path)
    print('Copying {} to {}'.format(source_path,dest_path))
    return None


#%% Copy images to output files (execution)

if n_copy_threads == 1:        
    for im in tqdm(images):
        copy_image_to_output(im)
else:
    pool = ThreadPool(n_copy_threads)
    copy_image_results = list(tqdm(pool.imap(copy_image_to_output, images), total=len(images)))


#%% Rename the main image list for consistency with other scripts
    
all_image_info = images


#%% Create CCT dictionaries

def transform_bbox(coords):
    """
    Derive CCT-formatted bounding boxes from the SamaSource coordinate system.
    
    SamaSource provides a list of four points (x,y) that should make a box.
    
    CCT coordinates are absolute, with the origin at the upper-left, as x,y,w,h.
    """
    
    # Make sure this is really a box
    assert len(coords) == 4
    assert all(len(coord) == 2 for coord in coords)
    assert coords[0][1] == coords[1][1]
    assert coords[2][1] == coords[3][1]
    assert coords[0][0] == coords[2][0]
    assert coords[1][0] == coords[3][0]
    
    # Transform to CCT format
    x = coords[0][0]
    y = coords[0][1]
    h = coords[2][1] - coords[0][1]
    w = coords[1][0] - coords[0][0]
    return [x, y, w, h]

annotations = []
image_ids_to_images = {}
category_name_to_category = {}

# Force the empty category to be ID 0
empty_category = {}
empty_category['id'] = 0
empty_category['name'] = 'empty'
category_name_to_category['empty'] = empty_category
next_category_id = 1

default_annotation = {}
default_annotation['tags'] = {}
default_annotation['tags']['Object'] = None

# i_image = 0; input_im = all_image_info[0]
for i_image,input_im in tqdm(enumerate(all_image_info),total=len(all_image_info)):

    output_im = {}
    output_im['id'] = input_im['id']
    output_im['file_name'] = input_im['output_relative_path']
    output_im['seq_id'] = input_im['seq_id']
    output_im['seq_num_frames'] = input_im['synthetic_seq_num_frames']
    output_im['frame_num'] = input_im['synthetic_frame_number']
    output_im['original_relative_path'] = input_im['original_relative_path']
    
    # This issue only impacted one image that wasn't a real image, it was just a screenshot 
    # showing "no images available for this camera"
    if 'location' not in input_im['exif_tags'] or input_im['exif_tags']['location'] == 'unknown':        
        print('Warning: no location for image {}, skipping'.format(
            input_im['url']))
        continue
    output_im['location'] = input_im['exif_tags']['location']
    
    assert output_im['id'] not in image_ids_to_images
    image_ids_to_images[output_im['id']] = output_im
    
    exif_tags = input_im['exif_tags']
    
    # Convert datetime if necessary
    dt = exif_tags['datetime']
    if dt is not None and isinstance(dt,str):
        dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    
    # Process temperature if available
    output_im['temperature'] = exif_tags['temperature'] if 'temperature' in exif_tags else None
    
    # Read width and height if necessary
    w = None
    h = None
    
    if 'width' in exif_tags:    
        w = exif_tags['width']
    if 'height' in exif_tags:    
        h = exif_tags['height']
        
    output_image_full_path = os.path.join(output_image_folder,input_im['output_relative_path'])
        
    if w is None or h is None:
        pil_image = Image.open(output_image_full_path)        
        w, h = pil_image.size
        
    output_im['width'] = w
    output_im['height'] = h

    # I don't know what this field is; confirming that it's always None
    assert input_im['Object'] is None
    
    # Process object and bbox
    input_annotations = input_im['Output']
    
    if input_annotations is None:
        input_annotations = [default_annotation]
        
    # os.startfile(output_image_full_path)
    
    for i_ann,input_annotation in enumerate(input_annotations):
        
        bbox = None
        
        assert isinstance(input_annotation,dict)
        
        if input_annotation['tags']['Object'] is None:
        
            # Zero is hard-coded as the empty category, but check to be safe
            category_id = 0
            assert category_name_to_category['empty']['id'] == category_id
        
        else:
            
            # I can't figure out the 'index' field, but I'm not losing sleep about it
            # assert input_annotation['index'] == 1+i_ann
            
            points = input_annotation['points']
            assert points is not None and len(points) == 4
            bbox = transform_bbox(points)
            assert len(input_annotation['tags']) == 1 and 'Object' in input_annotation['tags']
            
            # Some annotators (but not all) included "_partial" when animals were partially obscured
            category_name = input_annotation['tags']['Object'].replace('_partial','').lower().strip()
            
            # Annotators *mostly* used 'none', but sometimes 'empty'.  'empty' is CCT-correct.
            if category_name == 'none':
                category_name = 'empty'
                
            category_id = None
            
            # If we've seen this category before...
            if category_name in category_name_to_category:
                    
                category = category_name_to_category[category_name]
                category_id = category['id'] 
              
            # If this is a new category...
            else:
                
                category_id = next_category_id
                category = {}
                category['id'] = category_id
                category['name'] = category_name
                category_name_to_category[category_name] = category
                next_category_id += 1
        
        # ...if this is an empty/non-empty annotation
                
        # Create an annotation
        annotation = {}        
        annotation['id'] = str(uuid.uuid1())
        annotation['image_id'] = output_im['id']    
        annotation['category_id'] = category_id
        annotation['sequence_level_annotation'] = False
        if bbox is not None:
            annotation['bbox'] = bbox
            
        annotations.append(annotation)
        
    # ...for each annotation on this image
        
# ...for each image

images = list(image_ids_to_images.values())
categories = list(category_name_to_category.values())
print('Loaded {} annotations in {} categories for {} images'.format(
    len(annotations),len(categories),len(images)))


#%% Change *two* annotations on images that I discovered contains a human after running MDv4

manual_human_ids = ['a07fc88a-6dd8-4d66-b552-d21d50fa39d0','285363f9-d76d-4727-b530-a6bd401bb4c7']
human_id = [cat['id'] for cat in categories if cat['name'] == 'human'][0]
for ann in tqdm(annotations):
    if ann['image_id'] in manual_human_ids:
        old_cat_id = ann['category_id']
        print('Changing annotation for image {} from {} to {}'.format(
            ann['image_id'],old_cat_id,human_id))
        ann['category_id'] = human_id
    
    
#%% Move human images

human_image_ids = set()
human_id = [cat['id'] for cat in categories if cat['name'] == 'human'][0]

# ann = annotations[0]
for ann in tqdm(annotations):
    if ann['category_id'] == human_id:
        human_image_ids.add(ann['image_id'])

print('\nFound {} human images'.format(len(human_image_ids)))

for im in tqdm(images):
    if im['id'] not in human_image_ids:
        continue
    source_path = os.path.join(output_image_folder,im['file_name'])
    if not os.path.isfile(source_path):
        continue
    target_path = os.path.join(output_image_folder_humans,im['file_name'])
    print('Moving {} to {}'.format(source_path,target_path))    
    os.makedirs(os.path.dirname(target_path),exist_ok=True)
    shutil.move(source_path,target_path)
    
    
#%% Count images by location

locations_to_images = defaultdict(list)
for im in tqdm(images):
    locations_to_images[im['location']].append(im)

    
#%% Write output

info = {}
info['year'] = 2020
info['version'] = 1.0
info['description'] = 'Camera trap data collected from the Channel Islands, California'
info['contributor'] = 'The Nature Conservancy of California'

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_file, 'w'), indent=1)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))


#%% Validate output

from data_management.databases import sanity_check_json_db

fn = output_file
options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = output_image_folder
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False
    
sortedCategories, data, error = sanity_check_json_db.sanity_check_json_db(fn,options)


#%% Preview labels

from visualization import visualize_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 159
# viz_options.classes_to_exclude = [0]
viz_options.classes_to_include = ['other']
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file,image_db = visualize_db.process_images(db_path=output_file,
                                                        output_dir=os.path.join(output_base,'preview'),
                                                        image_base_dir=output_image_folder,
                                                        options=viz_options)
os.startfile(html_output_file)

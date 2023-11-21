#
# bellevue_to_json.py
#
# "Bellevue Camera Traps" is the rather unremarkable camera trap data set
# used by one of the repo's maintainers for testing.  It's organized as:
#
# approximate_date/[loose_camera_specifier/]/species    
#
# E.g.:
#    
# "2018.03.30\coyote\DSCF0091.JPG"
# "2018.07.18\oldcam\empty\DSCF0001.JPG"
#


#%% Constants and imports

import json
import os
import uuid
import datetime

from PIL import Image
from PIL.ExifTags import TAGS
from tqdm import tqdm

# from the ai4eutils repo
from path_utils import find_images

# Filenames will be stored in the output .json relative to this base dir
base_dir = r'C:\temp\camera_trap_images_no_people'
output_base = r'c:\temp\previews'
output_filename = os.path.join(base_dir,'bellevue_camera_traps.{}.json'.format(str(datetime.date.today())))

class_mappings = {'transitional':'unlabeled','moving':'unlabeled','setup':'unlabeled','blurry':'unlabeled','transitional':'unlabeled','junk':'unlabeled','unknown':'unlabeled','blurry':'unlabeled'}
class_mappings['dan'] = 'human'
class_mappings['dan_and_dog'] = 'human,dog'
class_mappings['dan and dog'] = 'human,dog'
class_mappings['unknown'] = 'unknown animal'
class_mappings['racoon'] = 'raccoon'


info = {}
info['year'] = 2020
info['version'] = '2.0'
info['description'] = 'Bellevue Camera Traps'
info['contributor'] = 'Dan Morris'
info['date_created'] = str(datetime.date.today())

max_files = -1

max_seconds_within_sequence = 10.0

assert os.path.isdir(base_dir)

#%% Exif functions

def get_exif_tags(fn=None,im=None):
    
    assert (fn is not None) or (im is not None)
    ret = {}
    if im is None:
        im = Image.open(fn)
    info = im._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value
        
    return ret


#%% Enumerate files, create image/annotation/category info

annotations = []

category_name_to_category = {}

# Force the empty category to be ID 0
empty_category = {}
empty_category['id'] = 0
empty_category['name'] = 'empty'
category_name_to_category['empty'] = empty_category
next_category_id = 1

# Keep track of unique camera folders
camera_folders = set()

# Each element will be a dictionary with fields:
#
# relative_path, width, height, datetime    
images = []
non_image_files = []

print('Enumerating files from {}'.format(base_dir))

image_files = find_images(base_dir,recursive=True)
print('Enumerated {} images'.format(len(image_files)))

# fname = image_files[0]
for fname in tqdm(image_files):
  
    if max_files >= 0 and len(images) > max_files:            
        print('Warning: early break at {} files'.format(max_files))
        break
    
    full_path = fname
    relative_path = os.path.relpath(full_path,base_dir)
     
    try:
        im = Image.open(full_path)
        h = im.height
        w = im.width
        tags = get_exif_tags(None,im)
        s = tags['DateTimeOriginal']
        dt = datetime.datetime.strptime(s,'%Y:%m:%d %H:%M:%S')
    except:
        # Corrupt or not an image
        print('Warning: could not read {}'.format(fname))
        non_image_files.append(full_path)
        continue
    
    # Store file info
    image_info = {}
    image_info['file_name'] = relative_path
    image_info['width'] = w
    image_info['height'] = h
    image_info['datetime'] = dt
    image_info['location'] = 'unknown'
    image_info['id'] = str(uuid.uuid4())

    images.append(image_info)
    
    # E.g. 2018.03.30/coyote/DSCF0091.JPG
    relative_path = image_info['file_name'].replace('\\','/')
    tokens = relative_path.split('/')
    camera_path_tokens = tokens[0:-2]
    camera_path = '/'.join(camera_path_tokens)
    camera_folders.add(camera_path)
    image_info['camera_path'] = camera_path
    
    category_name = tokens[-2].lower()
    if category_name in class_mappings:
        category_name = class_mappings[category_name]
        
    if category_name not in category_name_to_category:
        category = {}
        category['id'] = next_category_id
        category['name'] = category_name
        next_category_id = next_category_id + 1
        category_name_to_category[category_name] = category
    else:
        category = category_name_to_category[category_name]
    
    annotation = {}
    annotation['sequence_level_annotation'] = False
    annotation['id'] = str(uuid.uuid4())
    annotation['category_id'] = category['id']
    annotation['image_id'] = image_info['id']
    annotations.append(annotation)
        
# ...for each image file

assert len(annotations) == len(images)

categories = list(category_name_to_category.values())


#%% Synthesize sequence information        

print('Found {} camera folders'.format(len(camera_folders)))

camera_folders = list(camera_folders)

all_sequences = set()

# Sort images by time within each folder
# camera_path = camera_folders[0]
for i_camera,camera_path in enumerate(camera_folders):
    
    images_this_camera = [im for im in images if im['camera_path'] == camera_path]
    sorted_images_this_camera = sorted(images_this_camera, key = lambda im: im['datetime'])
    
    current_sequence_id = None
    next_sequence_index = 0
    previous_datetime = None
        
    # previous_datetime = sorted_images_this_camera[0]['datetime']
    # im = sorted_images_this_camera[1]
    for im in sorted_images_this_camera:
        
        if previous_datetime is None:
            delta = None
        else:
            delta = (im['datetime'] - previous_datetime).total_seconds()
        
        # Start a new sequence if necessary
        if delta is None or delta > max_seconds_within_sequence:
            next_sequence_index = 0
            current_sequence_id = str(uuid.uuid4())
            all_sequences.add(current_sequence_id)
            
        im['seq_id'] = current_sequence_id
        im['seq_num_frames'] = None
        im['frame_num'] = next_sequence_index
        next_sequence_index = next_sequence_index + 1
        previous_datetime = im['datetime']
    
    # ...for each image in this camera

# ...for each camera

print('Created {} sequences from {} images'.format(len(all_sequences),len(images)))

# Fill in seq_num_frames
num_frames_per_sequence = {}
for seq_id in all_sequences:
    images_this_sequence = [im for im in images if im['seq_id'] == seq_id]
    num_frames_per_sequence[seq_id] = len(images_this_sequence)
    for im in images_this_sequence:
        im['seq_num_frames'] = len(images_this_sequence)


#%% A little cleanup 

for im in tqdm(images):
    if 'camera_path' in im:
        del im['camera_path']
    if not isinstance(im['datetime'],str):
        im['datetime'] = str(im['datetime'])
    
    
#%% Write output .json

data = {}
data['info'] = info
data['images'] = images
data['annotations'] = annotations
data['categories'] = categories

json.dump(data, open(output_filename,'w'), indent=1)

print('Finished writing json to {}'.format(output_filename))


#%% Sanity-check data

from data_management.databases import sanity_check_json_db

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = base_dir
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = False
    
sorted_categories  = sanity_check_json_db.sanity_check_json_db(output_filename,options)


#%% Label previews

from visualization import visualize_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = None
viz_options.parallelize_rendering_n_cores = 8
viz_options.parallelize_rendering = True
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
html_output_file,image_db = visualize_db.process_images(output_filename,
                                                        os.path.join(output_base,'preview'),
                                                        base_dir,viz_options)
os.startfile(html_output_file)


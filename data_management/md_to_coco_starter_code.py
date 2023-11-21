#
# Partially-formed stub to get from MegaDetector output files to COCO Camera Traps data.
#
# Was actually written to convert *many* MD .json files to a single CCT file, hence 
# the loop over .json files.
#
# THIS CODE HAS NEVER BEEN RUN, it was added as a demonstration of how to do this.  YMMV.
#
# You may find a more polished, command-line-ready version of this code at:
#
# https://github.com/StewartWILDlab/mdtools
#

#%% Constants and imports

import os
import json
import uuid

from tqdm import tqdm
from PIL import Image
from collections import defaultdict

from visualization import visualize_db
from data_management.databases import sanity_check_json_db

json_file = 'some_file'
output_json_file = 'some_other_file'

# Images sizes are required to convert between absolute and relative coordinates,
# so we need to read the images.
image_base_folder = None

# Only required if you want to write a database preview
output_dir_base = '/some/folder'


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

print('Processing .json file {}'.format(json_file))

# Load .json annotations for this data set
with open(json_file, 'r') as f:
    data = f.read()        
data = json.loads(data)

categories_this_dataset = data['detection_categories']

# i_entry = 0; entry = data['images'][i_entry]
#
# PERF: Not exactly trivially parallelizable, but about 100% of the 
# time here is spent reading image sizes (which we need to do to get from 
# absolute to relative coordinates), so worth parallelizing.
for i_entry,entry in enumerate(tqdm(data['images'])):
    
    image_relative_path = entry['file']
    
    # Generate a unique ID from the path
    image_id = image_relative_path.split('.')[0].replace(
        '\\', '/').replace('/', '_').replace(' ', '_')
    
    im = {}
    im['id'] = image_id
    im['file_name'] = image_relative_path
    
    pil_image = Image.open(os.path.join(image_base_folder,image_relative_path))
    width, height = pil_image.size
    im['width'] = width
    im['height'] = height

    images.append(im)
    
    detections = entry['detections']
    
    # detection = detections[0]
    for detection in detections:
        
        category_name = categories_this_dataset[detection['category']]
        category_name = category_name.strip().lower()            
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
            # MegaDetector: [x,y,width,height] (normalized, origin upper-left)
            # CCT: [x,y,width,height] (absolute, origin upper-left)
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
            
print('Finished creating CCT dictionaries')

# Remove non-reviewed images and associated annotations


#%% Create info struct

info = dict()
info['year'] = 2020
info['version'] = 1.0
info['description'] = 'Fun With Camera Traps'
info['contributor'] = 'Somebody'


#%% Write .json output

categories = list(category_name_to_category.values())

json_data = {}

json_data['images'] = images
json_data['annotations'] = annotations
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
options.baseDir = image_base_folder
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False
sorted_categories, data, errors = sanity_check_json_db.sanity_check_json_db(fn, options)


#%% Preview animal labels

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 3000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file, image_db = visualize_db.process_images(db_path=output_json_file,
                                                         output_dir=os.path.join(
                                                         output_dir_base, 'preview_animals'),
                                                         image_base_dir=image_base_folder,
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
                                                         image_base_dir=image_base_folder,
                                                         options=viz_options)
os.startfile(html_output_file)

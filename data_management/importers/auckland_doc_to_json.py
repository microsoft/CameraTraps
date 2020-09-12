#
# auckland_doc_to_json.py
#
# Convert Auckland DOC data set to COCO camera traps format.  This was
# for a training data set where class names were encoded in path names.
#

#%% Constants and imports

import json
import os
import uuid
import datetime
from tqdm import tqdm

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
from path_utils import find_images, split_path, insert_before_extension

# Filenames will be stored in the output .json relative to this base dir
input_base_dir = 'y:\\'
output_base_dir = r'f:\auckland-doc'
output_json_filename = os.path.join(output_base_dir, 'auckland-doc-Maukahuka_Auckland_Island.json')

assert os.path.isdir(input_base_dir)
os.makedirs(output_base_dir,exist_ok=True)

output_encoding = 'utf-8'
read_image_sizes = True

info = {}
info['year'] = 2019
info['version'] = '1.0'
info['description'] = 'Auckaland DOC Camera Traps'
info['contributor'] = 'Auckland DOC'
info['date_created'] = str(datetime.date.today())


#%% Enumerate files

print('Enumerating files from {}'.format(input_base_dir))
image_files = find_images(input_base_dir, bRecursive=True)
print('Enumerated {} images'.format(len(image_files)))


#%% Assemble dictionaries

images = []
image_id_to_image = {}
annotations = []
categories = []

category_name_to_category = {}
category_id_to_category = {}

# Force the empty category to be ID 0
empty_category = {}
empty_category['name'] = 'empty'
empty_category['id'] = 0
category_id_to_category[0] = empty_category
categories.append(empty_category)
next_id = 1

behaviors = set()

# fn = image_files[0]; print(fn)
for fn in tqdm(image_files):

    # Typically y:\Maukahuka_Auckland_Island\1_Training\Winter_Trial_2019\cat\cat\eat\20190903_IDdY_34_E3_tmp_201908240051.JPG
    relative_path = os.path.relpath(fn,input_base_dir)
    tokens = split_path(fn)
    assert tokens[1] == 'Maukahuka_Auckland_Island'
    
    trainval_split = tokens[2]
    assert trainval_split in ['1_Training','2_Testing']
    
    # This data set has two top-level folders, "1_Training" (which has class names encoded
    # in paths) and "2_Testing" (which has no class information).
    if trainval_split == '2_Testing':
        category_name = 'test'
    else:
        category_name = tokens[-3]
        if category_name.startswith('2_'):
            category_name = category_name.replace('2_', '')
        category_name = category_name.lower().strip()

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
    behavior = None
    if (category_name) != 'test':
        behavior = fn.split('\\')[-2]
        behaviors.add(behavior)            

    im = {}
    im['id'] = str(uuid.uuid1())
    im['file_name'] = relative_path
    image_id_to_image[im['id']] = im
    
    images.append(im)
    
    ann = {}
    
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']
    ann['category_id'] = category_id
    if behavior is not None:
        ann['behavior'] = behavior
    annotations.append(ann)

# ...for each image
    

#%% Write output .json

data = {}
data['info'] = info
data['images'] = images
data['annotations'] = annotations
data['categories'] = categories

json.dump(data, open(output_json_filename, 'w'), indent=2)
print('Finished writing json to {}'.format(output_json_filename))


#%% Write train/test .jsons

train_images = []; test_images = []
train_annotations = []; test_annotations = []

for ann in tqdm(annotations):
    category_id = ann['category_id']
    image_id = ann['image_id']
    category_name = category_id_to_category[category_id]['name']
    im = image_id_to_image[image_id]
    if category_name == 'test':
        test_images.append(im)
        test_annotations.append(ann)
    else:
        train_images.append(im)
        train_annotations.append(ann)

train_fn = insert_before_extension(output_json_filename,'train')
test_fn = insert_before_extension(output_json_filename,'test')

data['images'] = train_images
data['annotations'] = train_annotations
json.dump(data, open(train_fn, 'w'), indent=2)

data['images'] = test_images
data['annotations'] = test_annotations
json.dump(data, open(test_fn, 'w'), indent=2)

    
#%% Validate .json files

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = input_base_dir
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True

sortedCategories, data = sanity_check_json_db.sanity_check_json_db(output_json_filename, options)
sortedCategories, data = sanity_check_json_db.sanity_check_json_db(train_fn, options)
sortedCategories, data = sanity_check_json_db.sanity_check_json_db(test_fn, options)


#%% Preview labels

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 2000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.classes_to_exclude = ['test']
html_output_file, image_db = visualize_db.process_images(db_path=output_json_filename,
                                                         output_dir=os.path.join(
                                                         output_base_dir, 'preview'),
                                                         image_base_dir=input_base_dir,
                                                         options=viz_options)
os.startfile(html_output_file)

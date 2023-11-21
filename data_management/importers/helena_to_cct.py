#
# helena_to_cct.py
#
# Convert the Helena Detections data set to a COCO-camera-traps .json file
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
import pandas as pd
from datetime import datetime

from path_utils import find_images

base_directory = r'/mnt/blobfuse/wildlifeblobssc/'
output_directory = r'/home/gramener'
output_json_file = os.path.join(output_directory,'rspb.json')
input_metadata_file = os.path.join(base_directory, 'StHelena_Detections.xlsx')
image_directory = os.path.join(base_directory, 'StHELENA_images/')
mapping_df = ''
filename_col = 'image_name'
load_width_and_height = True
annotation_fields_to_copy = ['Fortnight', 'Detector', 'datetime', 'site']

assert(os.path.isdir(image_directory))

# This is one time process
#%% Create Filenames and timestamps mapping CSV

image_full_paths = find_images(image_directory, bRecursive=True)
csv_file = os.path.join(output_directory, "mapping_names.csv")
if not os.path.exists(csv_file):
    map_list = []
    for img_ in image_full_paths:
        try:
            date_cr = Image.open(img_)._getexif()[306]
            _tmp = {}
            # import pdb;pdb.set_trace()
            img_path = img_.replace(image_directory, "")
            img_folder = img_path.split("/")[0]
            site = img_path.split("/")[1]
            detector = img_path.split("/")[2]
            _tmp["image_name"] = img_path
            _tmp["Fortnight"] = img_folder.replace("Fortnight", "")
            _tmp["site"] = site
            _tmp["Detector"] = detector
            _tmp["datetime"] = "-".join(date_cr.split(":")[:-1])
            map_list.append(_tmp)
        except Exception as e:
            print(e)
            print(img_)
    mapping_df = pd.DataFrame(map_list)
    mapping_df.to_csv(csv_file, index=False)
else:
    mapping_df = pd.read_csv(csv_file)

#%% To create CCT JSON for RSPB dataset

#%% Read source data
input_metadata = pd.read_excel(input_metadata_file)

print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns),
      len(input_metadata)))

# Original Excel file had timestamp in different columns
input_metadata['datetime'] = input_metadata[['DATUM', 'Hour', 'Mins']].apply(lambda x: '{0} {1}-{2}'.format(datetime.strftime(x[0], '%Y-%m-%d'),"{0:0=2d}".format(x[1]),"{0:0=2d}".format(x[2])), axis = 1)
input_metadata['Detector'] = "Detector"+input_metadata['Detector'].astype('str')
result = pd.merge(input_metadata, mapping_df, how='left', on=['datetime', "Fortnight", "site", "Detector"])


#%% Map filenames to rows, verify image existence
    
start_time = time.time()
filenames_to_rows = {}
image_filenames = result[filename_col]
image_filenames = list(set(image_filenames))

missing_files = []
duplicate_rows = []

# Build up a map from filenames to a list of rows, checking image existence as we go
for iFile, fn in enumerate(image_filenames):
    try:
        if fn == 'nan' or type(fn) == float:
            pass
        else:
            if (fn in filenames_to_rows):
                duplicate_rows.append(iFile)
                filenames_to_rows[fn].append(iFile)
            else:
                filenames_to_rows[fn] = [iFile]
                image_path = os.path.join(image_directory, fn)
                if not os.path.isfile(image_path):
                    missing_files.append(fn)
    except Exception as e:
        pass

elapsed = time.time() - start_time

print('Finished verifying image existence in {}, found {} missing files (of {})'.format(
    humanfriendly.format_timespan(elapsed), 
    len(missing_files),len(image_filenames)))

#%% Skipping this check because one image has multiple species
# assert len(duplicate_rows) == 0

#%% Check for images that aren't included in the metadata file

images_missing_from_metadata = []

for iImage, image_path in enumerate(image_full_paths):

    relative_path = os.path.relpath(image_path, image_directory)
    if relative_path not in filenames_to_rows:
        images_missing_from_metadata.append(relative_path)
    
print('{} of {} files are not in metadata'.format(len(images_missing_from_metadata),len(image_full_paths)))

#%% Create CCT dictionaries

images = []
annotations = []

# Map categories to integer IDs
#
# The category '0' is reserved for 'empty'

categories_to_category_id = {}
categories_to_counts = {}
categories_to_category_id['empty'] = 0
categories_to_counts['empty'] = 0

next_category_id = 1

# For each image

start_time = time.time()
for image_name in image_filenames:

    if type(image_name) != str:
        continue

    image_path = os.path.join(image_directory, image_name)
    # Don't include images that don't exist on disk
    if not os.path.isfile(image_path):
        continue
    
    im = {}
    im['id'] = image_name.split('.')[0]
    im['file_name'] = image_name

    if load_width_and_height:
        pilImage = Image.open(image_path)
        width, height = pilImage.size
        im['width'] = width
        im['height'] = height
    else:
        im['width'] = -1
        im['height'] = -1

    images.append(im)

    rows = filenames_to_rows[image_name]

    # Some filenames will match to multiple rows
    # assert(len(rows) == 1)

    # iRow = rows[0]
    for iRow in rows:
        row = result.iloc[iRow]

        category = row['Species']

        # Have we seen this category before?
        if category in categories_to_category_id:
            categoryID = categories_to_category_id[category]
            categories_to_counts[category] += 1
        else:
            categoryID = next_category_id
            categories_to_category_id[category] = categoryID
            categories_to_counts[category] = 1
            next_category_id += 1

        # Create an annotation
        ann = {}

        # The Internet tells me this guarantees uniqueness to a reasonable extent, even
        # beyond the sheer improbability of collisions.
        ann['id'] = str(uuid.uuid1())
        ann['image_id'] = im['id']
        ann['category_id'] = categoryID
        # ann['datetime'] = row['datetime']
        # ann['site'] = row['site']

        for fieldname in annotation_fields_to_copy:
            ann[fieldname] = row[fieldname]
            if ann[fieldname] is np.nan:
                ann[fieldname] = ''
            ann[fieldname] = str(ann[fieldname])

        annotations.append(ann)

# ...for each image

# Convert categories to a CCT-style dictionary
categories = []

for category in categories_to_counts:
    print('Category {}, count {}'.format(
        category, categories_to_counts[category]))
    categoryID = categories_to_category_id[category]
    cat = {}
    cat['name'] = category
    cat['id'] = categoryID
    categories.append(cat)

elapsed = time.time() - start_time
print('Finished creating CCT dictionaries in {}'.format(
      humanfriendly.format_timespan(elapsed)))


#%% Create info struct

info = {}
info['year'] = 2012
info['version'] = 1
info['description'] = 'RSPB Dataset'
info['contributor'] = 'Helena Detection'


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_json_file, 'w'), indent=4)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
    len(images), len(annotations), len(categories)))

#%% Validate output

from data_management.databases import sanity_check_json_db

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = image_directory
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False
data = sanity_check_json_db.sanity_check_json_db(output_json_file,options)


#%% Preview labels

from visualization import visualize_db
from data_management.databases import sanity_check_json_db

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = None
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.classes_to_exclude = ['empty']
html_output_file,image_db = visualize_db.process_images(db_path=output_json_file,
                                                        output_dir=os.path.join(
                                                        output_directory, 'RSPB/preview'),
                                                        image_base_dir=image_directory,
                                                        options=viz_options)
os.startfile(html_output_file))


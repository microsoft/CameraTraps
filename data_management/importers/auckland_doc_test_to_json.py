#
# auckland_doc_test_to_json.py
#
# Convert Auckland DOC data set to COCO camera traps format.  This was
# for a testing data set where a .csv file was provided with class
# information.
#

#%% Constants and imports

import json
import os
import uuid
import pandas as pd
import datetime
import ntpath
import re
import numpy as np
from tqdm import tqdm

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
from path_utils import find_images, insert_before_extension

input_base_dir = r'e:\auckland-test\2_Testing'

input_metadata_file = r'G:\auckland-doc\Maukahuka - Auckland Island - Cat camera data Master April 2019 - DOC-5924483.xlsx'

# Filenames will be stored in the output .json relative to this base dir
output_base_dir = r'g:\auckland-doc'
output_json_filename = os.path.join(output_base_dir, 'auckland-doc-test.json')

assert os.path.isdir(input_base_dir)
os.makedirs(output_base_dir,exist_ok=True)

output_encoding = 'utf-8'
read_image_sizes = True

info = {}
info['year'] = 2020
info['version'] = '1.0'
info['description'] = 'Auckaland DOC Camera Traps (test)'
info['contributor'] = 'Auckland DOC'
info['date_created'] = str(datetime.date.today())


#%% Enumerate files

print('Enumerating files from {}'.format(input_base_dir))
absolute_image_paths = find_images(input_base_dir, recursive=True)
print('Enumerated {} images'.format(len(absolute_image_paths)))

relative_image_paths = []
for fn in absolute_image_paths:
    relative_image_paths.append(os.path.relpath(fn,input_base_dir).replace('\\','/'))

relative_image_paths_set = set(relative_image_paths)

assert len(relative_image_paths_set) == len(relative_image_paths)


#%% Create unique identifier for each image

# The ground truth doesn't have full paths in it; create unique identifiers for each image 
# based on the camera name and filename.
# 
# We store file identifiers as cameraname_filename.
file_identifier_to_relative_paths = {}
camera_names = set()

# relative_path = relative_image_paths[0]
for relative_path in relative_image_paths:
    
    # Example relative paths
    #
    # Summer_Trial_2019/A1_1_42_SD114_20190210/AucklandIsland_A1_1_42_SD114_20190210_01300001.jpg
    # Winter_Trial_2019/Installation/10_F4/10_F4_tmp_201908210001.JPG
    fn = ntpath.basename(relative_path)    
    
    # Find the camera name
    tokens = relative_path.split('/')
    
    if tokens[1] == 'Installation' or 'Rebait' in tokens[1]:
        camera_name = tokens[2]
    
    else:
        # E..g. "A1_1_42_SD114_20190210" in the above example
        camera_token = tokens[1]
        camera_name = None
        m = re.search('^(.+)_SD',camera_token)
        if m:
            camera_name = m.group(1)
        else:
            # For camera tokens like C1_5_D_190207
            m = re.search('^(.+_.+_.+)',camera_token)
            camera_name = m.group(1)
    
    assert camera_name
    camera_names.add(camera_name)
    
    file_identifier = camera_name + '_' + fn
    if file_identifier not in file_identifier_to_relative_paths:
        file_identifier_to_relative_paths[file_identifier] = [relative_path]
    else:
        file_identifier_to_relative_paths[file_identifier].append(relative_path)
    
print('Found {} unique camera names'.format(len(camera_names)))
      
      
#%% Load input data

input_metadata = pd.read_excel(input_metadata_file)

print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns),
      len(input_metadata)))

# The spreadsheet has a space after "Camera"
input_metadata = input_metadata.rename(columns={'Camera ':'Camera'})
    

#%% Assemble dictionaries

image_id_to_image = {}
category_name_to_category = {}
annotations = []

# Force the empty category to be ID 0
empty_category = {}
empty_category['name'] = 'empty'
empty_category['id'] = 0
category_name_to_category['empty'] = empty_category

rows_not_found_in_folder = []
rows_ambiguous = []
rows_no_filename = []
rows_no_annotation = []

image_id_to_rows = {}

next_id = 1

category_names = ['cat','mouse','unknown','human','pig','sealion','penguin','dog','openadjusted']

# array([nan, 'Blackbird', 'Bellbird', 'Tomtit', 'Song thrush', 'Pippit',
#       'Pippet', '?', 'Dunnock', 'Song thursh', 'Kakariki', 'Tui', ' ',
#       'Silvereye', 'NZ Pipit', 'Blackbird and Dunnock', 'Unknown',
#       'Pipit', 'Songthrush'], dtype=object)

def bird_name_to_category_name(bird_name):
    bird_name = bird_name.lower().strip().replace(' ','_').replace('song_thursh','song_thrush')
    bird_name = bird_name.replace('pippet','pipt').replace('pippit','pipit').replace('nz_pipit','pipit')
    if bird_name == '?' or bird_name == '' or bird_name == 'unknown':
        category_name = 'unknown_bird'
    else:
        category_name = bird_name
    return category_name
    
bird_names = input_metadata.Bird_ID.unique()
for bird_name in bird_names:
    if isinstance(bird_name,float):
        continue
    category_name = bird_name_to_category_name(bird_name)
    if category_name not in category_names:
        category_names.append(category_name)
        
for category_name in category_names:
    cat = {}
    cat['name'] = category_name
    cat['id'] = next_id
    next_id = next_id +1
    category_name_to_category[category_name] = cat
    
def create_annotation(image_id,category_name,count):
    assert isinstance(image_id,str)
    assert isinstance(category_name,str)
    assert isinstance(count,int) or isinstance(count,float)
    if isinstance(count,float):
        count = int(count)
    ann = {}    
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = image_id
    category = category_name_to_category[category_name]
    category_id = category['id']
    ann['category_id'] = category_id
    ann['count'] = count
    return ann
    
# i_row = 0; row = input_metadata.iloc[i_row]
for i_row,row in tqdm(input_metadata.iterrows(),total=len(input_metadata)):

    # E.g.: AucklandIsland_A1_1_42_SD114_20190210_01300009.jpg
    filename = row['File']
    if isinstance(filename,float):
        rows_no_filename.append(i_row)
        continue
    
    camera_name = row['Camera']
    file_identifier = camera_name + '_' + filename
    if not file_identifier in file_identifier_to_relative_paths:
        rows_not_found_in_folder.append(i_row)
        continue        

    relative_paths_this_file_id = file_identifier_to_relative_paths[file_identifier]
    
    if len(relative_paths_this_file_id) == 1:
        relative_path = relative_paths_this_file_id[0]
    else:
        
        # We have multiple files matching this identifier, can we uniquely resolve this
        # to one of those files based on the camera ID?
        matches = [s for s in relative_paths_this_file_id if camera_name in s]
        assert len(matches) > 0
        if len(matches) > 1:
            rows_ambiguous.append(i_row)
            continue
        relative_path = matches[0]

    assert filename.endswith('.jpg') or filename.endswith('.JPG')
    image_id = filename.lower().replace('.jpg','')

    if image_id in image_id_to_rows:
        image_id_to_rows[image_id].append(i_row)
        continue
        
    image_id_to_rows[image_id] = [i_row]
        
    im = {}
    im['id'] = image_id 
    im['file_name'] = relative_path
    im['datetime'] = str(row['Date and time'])
    im['camera'] = row['Camera']
    im['sd_card'] = row['SD_Card']
    im['sd_change'] = row['SD_Change']
    im['comments'] = row['Comments']
    
    image_id_to_image[im['id']] = im
    
    # create_annotation(image_id,category_name,count)
    
    # 'SD_Change', 'Cat', 'Mouse', 'Bird', 'Bird_ID', 'False_trig', 'Unknown',
    # 'Human', 'Collared_cat', 'Cat_ID', 'Pig', 'Sea_lion', 'Open_adjusted',
    # 'Penguin', 'Dog', 'Comments', 'Unnamed: 22']
    
    # Each of these categories is handled a little differently...
    
    annotations_this_image = []
    if (not np.isnan(row['Cat'])):
        assert np.isnan(row['Collared_cat'] )
        annotations_this_image.append(create_annotation(im['id'],'cat',row['Cat']))
    
    if (not np.isnan(row['Collared_cat'])):
        assert np.isnan(row['Cat'] )
        annotations_this_image.append(create_annotation(im['id'],'cat',row['Collared_cat']))
    
    if (not np.isnan(row['Bird'])):
        if isinstance(row['Bird_ID'],str):
            category_name = bird_name_to_category_name(row['Bird_ID'])
        else:
            assert np.isnan(row['Bird_ID'])            
            category_name = 'unknown_bird'
        annotations_this_image.append(create_annotation(im['id'],category_name,row['Bird']))
    
    if (not np.isnan(row['False_trig'])):
        annotations_this_image.append(create_annotation(im['id'],'empty',-1))
    
    # These are straightforward
    for s in ['Mouse','Unknown','Pig','Human','Sea_lion','Penguin','Dog','Open_adjusted']:
        if isinstance(row[s],float) or isinstance(row[s],int):
            if not np.isnan(row[s]):
                 annotations_this_image.append(create_annotation(im['id'],s.lower().replace('_',''),row[s]))
        elif isinstance(row[s],str):
            print('File {}, label {}, value {}'.format(filename,s,row[s]))
        else:
            raise ValueError('Error handling count value {}'.format(row[s]))
        
    if len(annotations_this_image) > 1:
        print('Multiple annotations for filename {}'.format(filename))
       
    if len(annotations_this_image) == 0:
        rows_no_annotation.append(i_row)
        
    annotations.extend(annotations_this_image)

# ...for each image


#%% Summarize errors

print('Of {} rows:\n'.format(len(input_metadata)))

print('{} images not found in folder'.format(len(rows_not_found_in_folder)))
print('{} images ambiguously mapped'.format(len(rows_ambiguous)))
print('{} images no filename'.format(len(rows_no_filename)))
print('{} images no annotation'.format(len(rows_no_annotation)))
print('{} images handled successfully, {} total annotations'.format(len(image_id_to_image),len(annotations)))
    

#%% Write output .json

images = list(image_id_to_image.values())
categories = list(category_name_to_category.values())

data = {}
data['info'] = info
data['images'] = images
data['annotations'] = annotations
data['categories'] = categories

json.dump(data, open(output_json_filename, 'w'), indent=1)
print('Finished writing json to {}'.format(output_json_filename))


#%% Validate .json file

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = input_base_dir
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False

sortedCategories, data, _ = sanity_check_json_db.sanity_check_json_db(output_json_filename, options)


#%% Preview labels

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 2000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = False
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
viz_options.classes_to_exclude = ['empty']
html_output_file, image_db = visualize_db.process_images(db_path=output_json_filename,
                                                         output_dir=os.path.join(
                                                         output_base_dir, 'preview'),
                                                         image_base_dir=input_base_dir,
                                                         options=viz_options)
os.startfile(html_output_file)


#%% Precision-recall analysis

from api.batch_processing.postprocessing.postprocess_batch_results import PostProcessingOptions
from api.batch_processing.postprocessing.postprocess_batch_results import process_batch_results

api_output_file = r'g:\auckland-doc\auckland-doc_20200801\combined_api_outputs\auckland-doc_202008012020.08.01_reformatMaukahuka_Auckland_Island2_TestingSummer_Trial_2019_detections.filtered_rde_0.60_0.85_5_0.05.json'
postprocessing_output_folder = r'G:\auckland-doc\auckland-doc_20200801\postprocessing'
image_base = r'E:\auckland-test\2_Testing'
ground_truth_json_file = output_json_filename

output_base = os.path.join(postprocessing_output_folder,'pr_analysis')
os.makedirs(output_base,exist_ok=True)

options = PostProcessingOptions()
options.unlabeled_classes.append('openadjusted')
options.image_base_dir = image_base
options.parallelize_rendering = True
options.include_almost_detections = True
options.num_images_to_sample = 2500
options.confidence_threshold = 0.75
options.almost_detection_confidence_threshold = 0.7
options.ground_truth_json_file = ground_truth_json_file
options.allow_missing_images = True
options.ground_truth_filename_replacements = {} 
options.api_output_filename_replacements = {'2020.08.01_reformat\\Maukahuka_Auckland_Island\\2_Testing\\':''}
options.api_output_file = api_output_file
options.output_dir = output_base
ppresults = process_batch_results(options)
os.startfile(ppresults.output_html_file)


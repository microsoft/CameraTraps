#
# get_lila_category_counts.py
#
# Example of how to create a csv of category counts per dataset (at '<output_dir> + 
# category_counts.xlsx') and a coco_file (at'<output_dir> + compiled_coco.json') for
# all identified images of the given category over the given datasets.
#
# You can constrain the category and datasets looked at in code below.
#
# Columns in the csv are:
#   number of images labeled with given category(im_cnt)
#   number of images with a bbox labeled with given category (bbox_im_cnt)
#   number of bbox's labeled with given category (bbox_cnt)
#
# This script also outputs two pickle files which contain dictionaries.
#
# The 'total_category_counts_by_set.pkl' file contains a dictionary of dictionaries of the form:
#
# {<dataset-with-image-level-annotations>:
#            {<category> : { 'image_urls': [<image-urls-of-images-with-category>],
#                           'im_cnt :<number>},
#            <category2>: ...},
#  <dataset-with-bbox-level-annotations>:
#            {<category> : { 'image_urls': [<image-urls-of-images-with-category>],
#                           'bbox_im_cnt :<number>},
#                           'bbox_cnt':<number>},
#            <category2>: ...},
#  <dataset3>: ...}
#
# The 'total_category_counts.pkl' file contains a dictionary of dictionaries of the form:
#
# {<category> : { 'image_urls': [<image-urls-of-images-with-category>],
#                'im_cnt :<number,
#                'bbox_im_cnt :<number>},
#                'bbox_cnt':<number>},
#  <category2>: ...}
#

#%% Constants and imports

import json
import urllib.request
import tempfile
import zipfile
import os

import pickle
import pandas as pd

from urllib.parse import urlparse

# LILA camera trap master metadata file
metadata_url = 'http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt'

## dictionaries to fill for output

# category count over all datasets
total_category_counts = {} 

# category count seperated by datasets
total_category_counts_by_set = {} 

## datasets and categories to look at

# if False, will only collect data from datasets_of_interest
use_all_datasets = True 

# only meaningful if use_all_datasets is false
datasets_of_interest = [] 

# if False, will only collect data for categories in categories_of_interest
use_all_categories = False 

# only meaningly if use_all_categories is False
categories_of_interest = ['aardvark', 'aardvarkantbear', 'orycteropus afer']  

# How the categories should be labeled in the csv. key is label in lila dataset,
# value is label to use in csv
category_mapping = {'aardvark': 'antelope, aardvark',
                    'aardvarkantbear': 'antelope, aardvark',
                    'orycteropus afer': 'antelope, aardvark'}

# We'll write images, metadata downloads, and temporary files here
lila_local_base = r'g:\temp\lila'

output_dir = os.path.join(lila_local_base,'lila_category_counts')
os.makedirs(output_dir,exist_ok=True)

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)


#%% Support functions

def download_url(url, destination_filename=None, force_download=False, verbose=True):
    """
    Download a URL (defaulting to a temporary file)
    """
    
    if destination_filename is None:
        temp_dir = os.path.join(tempfile.gettempdir(),'lila')
        os.makedirs(temp_dir,exist_ok=True)
        url_as_filename = url.replace('://', '_').replace('.', '_').replace('/', '_')
        destination_filename =             os.path.join(temp_dir,url_as_filename)
            
    if (not force_download) and (os.path.isfile(destination_filename)):
        print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url)))
        return destination_filename
    
    if verbose:
        print('Downloading file {} to {}'.format(os.path.basename(url),destination_filename),end='')
    
    os.makedirs(os.path.dirname(destination_filename),exist_ok=True)
    urllib.request.urlretrieve(url, destination_filename)  
    assert(os.path.isfile(destination_filename))
    
    if verbose:
        nBytes = os.path.getsize(destination_filename)    
        print('...done, {} bytes.'.format(nBytes))
        
    return destination_filename


def unzip_file(input_file, output_folder=None):
    """
    Unzip a zipfile to the specified output folder, defaulting to the same location as
    the input file    
    """
    
    if output_folder is None:
        output_folder = os.path.dirname(input_file)
        
    with zipfile.ZipFile(input_file, 'r') as zf:
        zf.extractall(output_folder)


#%% Download and parse the metadata file

# Put the master metadata file in the same folder where we're putting images
p = urlparse(metadata_url)
metadata_filename = os.path.join(metadata_dir,os.path.basename(p.path))
download_url(metadata_url, metadata_filename)

# Read lines from the master metadata file
with open(metadata_filename,'r') as f:
    metadata_lines = f.readlines()
metadata_lines = [s.strip() for s in metadata_lines]

# Parse those lines into a table
metadata_table = {}

for s in metadata_lines:
    
    if len(s) == 0 or s[0] == '#':
        continue
    
    # Each line in this file is name/sas_url/json_url/[bbox_json_url]
    tokens = s.split(',')
    assert len(tokens) == 4
    ds_name = tokens[0].strip()
    url_mapping = {'sas_url':tokens[1],'json_url':tokens[2]}
    metadata_table[ds_name] = url_mapping
    
    # Create a separate entry for bounding boxes if they exist
    if len(tokens[3].strip()) > 0:
        print('Adding bounding box dataset for {}'.format(ds_name))
        bbox_url_mapping = {'sas_url':tokens[1],'json_url':tokens[3]}
        metadata_table[tokens[0]+'_bbox'] = bbox_url_mapping
        assert 'https' in bbox_url_mapping['json_url']

    assert 'https' not in tokens[0]
    assert 'https' in url_mapping['sas_url']
    assert 'https' in url_mapping['json_url']


#%% Download and extract metadata for the datasets we're interested in

if use_all_datasets: datasets_of_interest = list(metadata_table.keys())

for ds_name in datasets_of_interest:
    
    assert ds_name in metadata_table
    json_url = metadata_table[ds_name]['json_url']
    
    p = urlparse(json_url)
    json_filename = os.path.join(metadata_dir,os.path.basename(p.path))
    download_url(json_url, json_filename)
    
    # Unzip if necessary
    if json_filename.endswith('.zip'):
        
        with zipfile.ZipFile(json_filename,'r') as z:
            files = z.namelist()
        assert len(files) == 1
        unzipped_json_filename = os.path.join(metadata_dir,files[0])
        if not os.path.isfile(unzipped_json_filename):
            unzip_file(json_filename,metadata_dir)        
        else:
            print('{} already unzipped'.format(unzipped_json_filename))
        json_filename = unzipped_json_filename
    
    metadata_table[ds_name]['json_filename'] = json_filename
    
# ...for each dataset of interest


#%% Count categories

coco_annotations = []
coco_images = []
coco_category_names = {}
coco_category_id = 0

for ds_name in datasets_of_interest:
    
    print('counting categories in: ' + ds_name)
    
    json_filename = metadata_table[ds_name]['json_filename']
    sas_url = metadata_table[ds_name]['sas_url']
    
    base_url = sas_url.split('?')[0]    
    assert not base_url.endswith('/')
    
    ## Open the metadata file
    
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    categories = data['categories']
    category_ids = [c['id'] for c in categories]
    for c in categories:
        c['name'] = c['name'].lower()
        category_id_to_name = {c['id']:c['name'] for c in categories}
    annotations = data['annotations']
    images = data['images']
    
    # Double check the annotations url provided corresponds to that implied by ds_name, or else fix
    
    # Only need to look at first entry b/c json files with image-level annotations
    # are seperated from those with box-level annotations.
    if 'bbox' in annotations[0]: 
        if ds_name.split('_')[-1] != 'bbox': 
            ds_name = ds_name + '_bbox'
    else:
        if ds_name.split('_')[-1] == 'bbox': 
            ds_name = ds_name.split('_')[0]
            
    # Build a list of image files (relative path names) that match the target categories
    if ds_name not in total_category_counts_by_set.keys():
        total_category_counts_by_set[ds_name] = {}
        
    for category_id in category_ids:
        
        categories = category_id_to_name[category_id]
        if not use_all_categories: 
            if categories not in categories_of_interest: 
                continue
        category_name = category_mapping[categories]    
        
        # Add categories entry to total_category_counts if not already present
        if categories not in total_category_counts.keys(): 
            coco_category_names[category_name] = coco_category_id
            coco_category_id += 1
            total_category_counts[category_name] = {}
            total_category_counts[category_name]['im_cnt'] = 0
            total_category_counts[category_name]['bbox_im_cnt'] = 0
            total_category_counts[category_name]['bbox_cnt'] = 0
            total_category_counts[category_name]['image_urls'] = []
            
        if category_name not in total_category_counts_by_set[ds_name].keys():
            total_category_counts_by_set[ds_name][category_name] = {}
            total_category_counts_by_set[ds_name][category_name]['image_urls'] = []
        
        # Retrieve all the images that match that category
        annotations_of_interest = [ann for ann in annotations if ann['category_id'] == category_id]
        image_ids_of_interest = [ann['image_id'] for ann in annotations_of_interest]
        image_ids_of_interest_set = set(image_ids_of_interest)
        images_of_interest = [im for im in images if im['id'] in image_ids_of_interest_set]
        filenames = [im['file_name'] for im in images_of_interest]
        assert len(filenames) == len(image_ids_of_interest_set)
        
        # Convert to URLs and add to category_counts dicts
        for fn in filenames:        
            url = base_url + '/' + fn
            total_category_counts[category_name]['image_urls'].append(url)
            total_category_counts_by_set[ds_name][category_name]['image_urls'].append(url)
        
        # Record total category count in dataset
        im_category_cnt = len(image_ids_of_interest_set) # count number unique images with this category
        if ds_name.split('_')[-1] == 'bbox': # only need to look at first entry b/c json files with image-level annotations are seperated from those with box-level
            if 'bbox' not in annotations[0]:  print(ds_name, category_name, 'bad1')
            bbox_category_cnt = len(image_ids_of_interest) # count number bounding boxes with this category
            total_category_counts[category_name]['bbox_im_cnt'] += im_category_cnt
            total_category_counts[category_name]['bbox_cnt'] += bbox_category_cnt 
            total_category_counts_by_set[ds_name][category_name]['bbox_im_cnt'] = im_category_cnt 
            total_category_counts_by_set[ds_name][category_name]['bbox_cnt'] = bbox_category_cnt
        else: 
            if 'bbox' in annotations[0]: print(ds_name, category_name, 'bad2')
            total_category_counts[category_name]['im_cnt'] += im_category_cnt
            total_category_counts_by_set[ds_name][category_name]['im_cnt'] = im_category_cnt
            
        # Add relevant annotations to custom coco file
        for annotation in annotations_of_interest:
            new_annotation = annotation.copy()
            new_annotation['image_id'] = ds_name + '_' + annotation['image_id']
            new_annotation['category_id'] = coco_category_names[category_name]
            coco_annotations.append(new_annotation)
        for im in images_of_interest:
            new_image = im.copy()
            new_image['id'] = ds_name + '_' + im['id']
            coco_images.append(new_image)


#%% Save output COCO files

coco_categories = [{'id':v, 'name':k} for k,v in coco_category_names.items()]
coco = {'categories':coco_categories, 'annotations':coco_annotations, 'images':coco_images}

json_data = json.dumps(coco)
with open(os.path.join(output_dir,'compiled_coco.json'),'w') as f:
    f.write(json_data)


#%% Save category counts to csv

writer = pd.ExcelWriter(os.path.join(output_dir,'category_counts.xlsx'), engine='xlsxwriter')

for category in total_category_counts.keys():
    
    col0, col1, col2, col3 = [], [], [], []
    for dataset in total_category_counts_by_set.keys():
        if category in total_category_counts_by_set[dataset]:
            col0.append(dataset)
            if dataset.split('_')[-1] == 'bbox':
                col1.append(0)
                col2.append(total_category_counts_by_set[dataset][category]['bbox_im_cnt'])
                col3.append(total_category_counts_by_set[dataset][category]['bbox_cnt'])
            else:
                col1.append(total_category_counts_by_set[dataset][category]['im_cnt'])
                col2.append(0)
                col3.append(0)
            
    df = pd.DataFrame({'dataname': col0, 'im_cnt': col1, 'bbox_im_cnt': col2, 'bbox_cnt':col3})
    df.to_excel(writer, sheet_name=category)
    
writer.save()


#%% Save dictionary of category counts and image urls to file

with open(os.path.join(output_dir,'total_category_counts_by_set.pkl'),'wb') as f:
    pickle.dump(total_category_counts_by_set,f)

with open(os.path.join(output_dir,'total_category_counts.pkl'),'wb') as f:
    pickle.dump(total_category_counts,f)

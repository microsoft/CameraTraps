#
# get_lila_species_counts.py
#
# Example of how to create a csv of species counts per dataset (at '<output_dir> + 
# species_counts.xlsx') and a coco_file (at'<output_dir> + compiled_coco.json') for
# all identified images of the given  species over the given datasets.
#
# You can constrain the species and datasets looked at in code below.
#
# Columns in the csv are:
#   number of images labeled with given species(im_cnt)
#   number of images with a bbox labeled with given species (bbox_im_cnt)
#   number of bbox's labeled with given species (bbox_cnt)
#
# This script also outputs two pickle files which contain dictionaries.
#
# The 'total_species_counts_by_set.pkl' file contains a dictionary of dictionaries of the form:
#
# {<dataset-with-image-level-annotations>:
#            {<species> : { 'image_urls': [<image-urls-of-images-with-species>],
#                           'im_cnt :<number>},
#            <species2>: ...},
#  <dataset-with-bbox-level-annotations>:
#            {<species> : { 'image_urls': [<image-urls-of-images-with-species>],
#                           'bbox_im_cnt :<number>},
#                           'bbox_cnt':<number>},
#            <species2>: ...},
#  <dataset3>: ...}
#
# The 'total_species_counts.pkl' file contains a dictionary of dictionaries of the form:
#
# {<species> : { 'image_urls': [<image-urls-of-images-with-species>],
#                'im_cnt :<number,
#                'bbox_im_cnt :<number>},
#                'bbox_cnt':<number>},
#  <species2>: ...}
#
#
# Data set names with '_bbox' appended are supposed to have bounding box level annotations
# while those without are to have image-level annotations. The mapping is not 
# guarauteed, however, so it's most likely best to include both versions
# and let the csv and pickle outputs seperate the images for you correctly.
#
# Options:
#
#'Caltech Camera Traps', 'Caltech Camera Traps_bbox', 'ENA24_bbox',
# 'Missouri Camera Traps_bbox', 'NACTI', 'NACTI_bbox', 'WCS Camera Traps', 'WCS Camera Traps_bbox',
# 'Wellington Camera Traps', 'Island Conservation Camera Traps_bbox', 'Channel Islands Camera Traps_bbox',
# 'Snapshot Serengeti', 'Snapshot Serengeti_bbox', 'Snapshot Karoo', 'Snapshot Kgalagadi',
# 'Snapshot Enonkishu', 'Snapshot Camdeboo', 'Snapshot Mountain Zebra', 'Snapshot Kruger'
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

# species count over all datasets
total_species_counts = {} 

# species count seperated by datasets
total_species_counts_by_set = {} 

## datasets and species to look at

# if False, will only collect data for species in species_of_interest
use_all_datasets = True 

# only meaningful if restrict_species is false
datasets_of_interest = [] 

# if False, will only collect data for species in species_of_interest
use_all_species = False 

# only need if restrict_species is false
species_of_interest = ['aardvark', 'aardvarkantbear', 'orycteropus afer']  

# how the species should be labeled in the csv. key is label in lila dataset, value is label to use in csv
species_mapping = {'aardvark': 'antelope, aardvark',
                    'aardvarkantbear': 'antelope, aardvark',
                    'orycteropus afer': 'antelope, aardvark'}

# We'll write images, metadata downloads, and temporary files here
output_dir = r'c:\temp\lila_downloads_by_species'
os.makedirs(output_dir,exist_ok=True)


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
metadata_filename = os.path.join(output_dir,os.path.basename(p.path))
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
    
    # Each line in this file is name/sas_url/json_url/bbox_json_url
    tokens = s.split(',')
    assert len(tokens) == 4 or len(tokens) == 3
    url_mapping = {'sas_url':tokens[1],'json_url':tokens[2]}
    metadata_table[tokens[0]] = url_mapping
    if len(tokens) == 4:
        if tokens[3] != 'NA':
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
    json_filename = os.path.join(output_dir,os.path.basename(p.path))
    download_url(json_url, json_filename)
    
    # Unzip if necessary
    if json_filename.endswith('.zip'):
        
        with zipfile.ZipFile(json_filename,'r') as z:
            files = z.namelist()
        assert len(files) == 1
        unzipped_json_filename = os.path.join(output_dir,files[0])
        if not os.path.isfile(unzipped_json_filename):
            unzip_file(json_filename,output_dir)        
        else:
            print('{} already unzipped'.format(unzipped_json_filename))
        json_filename = unzipped_json_filename
    
    metadata_table[ds_name]['json_filename'] = json_filename
    
# ...for each dataset of interest


#%% Count species

coco_annotations = []
coco_images = []
coco_category_names = {}
coco_category_id = 0

for ds_name in datasets_of_interest:
    
    print('counting species in: ' + ds_name)
    
    json_filename = metadata_table[ds_name]['json_filename']
    sas_url = metadata_table[ds_name]['sas_url']
    
    base_url = sas_url.split('?')[0]    
    assert not base_url.endswith('/')
    
    sas_token = sas_url.split('?')[1]
    assert not sas_token.startswith('?')
    
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
    
    # only need to look at first entry b/c json files with image-level annotations
    # are seperated from those with box-level
    if 'bbox' in annotations[0]: 
        if ds_name.split('_')[-1] != 'bbox': 
            ds_name = ds_name + '_bbox'
    else:
        if ds_name.split('_')[-1] == 'bbox': 
            ds_name = ds_name.split('_')[0]
            
    # Build a list of image files (relative path names) that match the target species
    if ds_name not in total_species_counts_by_set.keys():
        total_species_counts_by_set[ds_name] = {}
        
    for category_id in category_ids:
        
        species = category_id_to_name[category_id]
        if not use_all_species: 
            if species not in species_of_interest: 
                continue
        species = species_mapping[species]    
        
        # add species entry to total_species_counts if not already present
        if species not in total_species_counts.keys(): 
            coco_category_names[species] = coco_category_id
            coco_category_id += 1
            total_species_counts[species] = {}
            total_species_counts[species]['im_cnt'] = 0
            total_species_counts[species]['bbox_im_cnt'] = 0
            total_species_counts[species]['bbox_cnt'] = 0
            total_species_counts[species]['image_urls'] = []
            
        if species not in total_species_counts_by_set[ds_name].keys():
            total_species_counts_by_set[ds_name][species] = {}
            total_species_counts_by_set[ds_name][species]['image_urls'] = []
        
        # Retrieve all the images that match that category
        annotations_of_interest = [ann for ann in annotations if ann['category_id'] == category_id]
        image_ids_of_interest = [ann['image_id'] for ann in annotations_of_interest]
        image_ids_of_interest_set = set(image_ids_of_interest)
        images_of_interest = [im for im in images if im['id'] in image_ids_of_interest_set]
        filenames = [im['file_name'] for im in images_of_interest]
        assert len(filenames) == len(image_ids_of_interest_set)
        
        # Convert to URLs and add to species_counts dicts
        for fn in filenames:        
            url = base_url + '/' + fn
            total_species_counts[species]['image_urls'].append(url)
            total_species_counts_by_set[ds_name][species]['image_urls'].append(url)
        
        # Record total species count in dataset
        im_species_cnt = len(image_ids_of_interest_set) # count number unique images with this species
        if ds_name.split('_')[-1] == 'bbox': # only need to look at first entry b/c json files with image-level annotations are seperated from those with box-level
            if 'bbox' not in annotations[0]:  print(ds_name, species, 'bad1')
            bbox_species_cnt = len(image_ids_of_interest) # count number bounding boxes with this species
            total_species_counts[species]['bbox_im_cnt'] += im_species_cnt
            total_species_counts[species]['bbox_cnt'] += bbox_species_cnt 
            total_species_counts_by_set[ds_name][species]['bbox_im_cnt'] = im_species_cnt 
            total_species_counts_by_set[ds_name][species]['bbox_cnt'] = bbox_species_cnt
        else: 
            if 'bbox' in annotations[0]: print(ds_name, species, 'bad2')
            total_species_counts[species]['im_cnt'] += im_species_cnt
            total_species_counts_by_set[ds_name][species]['im_cnt'] = im_species_cnt
            
        # Add relevant annotations to custom coco file
        for annotation in annotations_of_interest:
            new_annotation = annotation.copy()
            new_annotation['image_id'] = ds_name + '_' + annotation['image_id']
            new_annotation['category_id'] = coco_category_names[species]
            coco_annotations.append(new_annotation)
        for im in images_of_interest:
            new_image = im.copy()
            new_image['id'] = ds_name + '_' + im['id']
            coco_images.append(new_image)


#%% Save output coco files


coco_categories = [{'id':v, 'name':k} for k,v in coco_category_names.items()]
coco = {'categories':coco_categories, 'annotations':coco_annotations, 'images':coco_images}

json_data = json.dumps(coco)
with open(os.path.join(output_dir,'compiled_coco.json'),'w') as f:
    f.write(json_data)


#%% Save species counts to csv

writer = pd.ExcelWriter(os.path.join(output_dir,'species_counts.xlsx'), engine='xlsxwriter')

for species in total_species_counts.keys():
    
    col0, col1, col2, col3 = [], [], [], []
    for dataset in total_species_counts_by_set.keys():
        if species in total_species_counts_by_set[dataset]:
            col0.append(dataset)
            if dataset.split('_')[-1] == 'bbox':
                col1.append(0)
                col2.append(total_species_counts_by_set[dataset][species]['bbox_im_cnt'])
                col3.append(total_species_counts_by_set[dataset][species]['bbox_cnt'])
            else:
                col1.append(total_species_counts_by_set[dataset][species]['im_cnt'])
                col2.append(0)
                col3.append(0)
            
    df = pd.DataFrame({'dataname': col0, 'im_cnt': col1, 'bbox_im_cnt': col2, 'bbox_cnt':col3})
    df.to_excel(writer, sheet_name=species)
    
writer.save()


#%% Save dictionary of species counts and image urls to file

with open(os.path.join(output_dir,'total_species_counts_by_set.pkl'),'wb') as f:
    pickle.dump(total_species_counts_by_set,f)

with open(os.path.join(output_dir,'total_species_counts.pkl'),'wb') as f:
    pickle.dump(total_species_counts,f)

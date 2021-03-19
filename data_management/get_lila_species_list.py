
#
# get_lila_species_list.py
#
# Example of making a text file listing all species names in given datasets
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

from urllib.parse import urlparse

# LILA camera trap master metadata file
metadata_url = 'http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt'

# array to fill for output
species_list = []

## datasets and species to look at

# if False, will only collect data for species in species_of_interest
use_all_datasets = True 

# only need if restrict_species is false
datasets_of_interest = [] 

# We'll write images, metadata downloads, and temporary files here
output_dir = r'c:\temp\lila_downloads_by_species'
os.makedirs(output_dir,exist_ok=True)
output_file = os.path.join(output_dir,'species_list.txt')


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


def download_relative_filename(url, output_base, verbose=False):
    """
    Download a URL to output_base, preserving relative path
    """
    
    p = urlparse(url)
    # remove the leading '/'
    assert p.path.startswith('/'); relative_filename = p.path[1:]
    destination_filename = os.path.join(output_base,relative_filename)
    download_url(url, destination_filename, verbose=verbose)
    

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


#%% Get species names

for ds_name in datasets_of_interest:
    
    json_filename = metadata_table[ds_name]['json_filename']
    sas_url = metadata_table[ds_name]['sas_url']
    
    base_url = sas_url.split('?')[0]    
    assert not base_url.endswith('/')
    
    sas_token = sas_url.split('?')[1]
    assert not sas_token.startswith('?')
    
    # Open the metadata file
    
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    # Collect list of categories and mappings to species name
    categories = data['categories']
    category_ids = [c['id'] for c in categories]
    for c in categories:
        c['name'] = c['name'].lower()
        category_id_to_name = {c['id']:c['name'] for c in categories}
    
    # Append species to species_list
    for category_id in category_ids:
        species = category_id_to_name[category_id]
        if species not in species_list: species_list.append(species)


#%% Save possible species to file

with open(output_file, 'w') as txt_file:
    for line in species_list:
        txt_file.write(line + '\n')

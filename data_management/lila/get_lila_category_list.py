#
# get_lila_category_list.py
#
# Generates a .json-formatted dictionary mapping each LILA dataset to all categories
# that exist for that dataset, with counts for the number of occurrences of each category.
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
category_list = []

## datasets and categories to look at

# if False, will only collect data for categories in datasets_of_interest
use_all_datasets = True 

# only need if restrict_category is false
datasets_of_interest = []

# We'll write images, metadata downloads, and temporary files here
lila_local_base = r'g:\temp\lila'

output_dir = os.path.join(lila_local_base,'lila_categories_list')
os.makedirs(output_dir,exist_ok=True)

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)

output_file = os.path.join(output_dir,'lila_dataset_to_categories.json')


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
metadata_filename = os.path.join(metadata_dir,os.path.basename(p.path))

# Download the metadata file if necessary
download_url(metadata_url,metadata_filename)

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
        bbox_url_mapping = {'sas_url':tokens[1],'json_url':tokens[3]}
        metadata_table[tokens[0]+'_bbox'] = bbox_url_mapping
        assert 'https' in bbox_url_mapping['json_url']

    assert 'https' not in tokens[0]
    assert 'https' in url_mapping['sas_url']
    assert 'https' in url_mapping['json_url']

print('Read {} entries from the metadata file (including bboxes)'.format(len(metadata_table)))


#%% Download and extract metadata for the datasets we're interested in

if use_all_datasets:
    
    datasets_of_interest = list(metadata_table.keys())

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


#%% Get category names for each dataset

from collections import defaultdict

dataset_to_categories = {}

# ds_name = datasets_of_interest[0]
# ds_name = 'NACTI'
for ds_name in datasets_of_interest:
    
    print('Finding categories in {}'.format(ds_name))
    
    json_filename = metadata_table[ds_name]['json_filename']
    sas_url = metadata_table[ds_name]['sas_url']
    
    base_url = sas_url.split('?')[0]    
    assert not base_url.endswith('/')
    
    # Open the metadata file    
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    # Collect list of categories and mappings to category name
    categories = data['categories']
    
    category_id_to_count = defaultdict(int)
    annotations = data['annotations']    
    
    # ann = annotations[0]
    for ann in annotations:
        category_id_to_count[ann['category_id']] = category_id_to_count[ann['category_id']] + 1
    
    # c = categories[0]
    for c in categories:
       count = category_id_to_count[c['id']] 
       if 'count' in c:
           assert 'bbox' in ds_name or c['count'] == count       
       c['count'] = count
    
    dataset_to_categories[ds_name] = categories

# ...for each dataset    


#%% Save dict

with open(output_file, 'w') as f:
    json.dump(dataset_to_categories,f,indent=2)

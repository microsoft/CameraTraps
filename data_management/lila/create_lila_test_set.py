#
# create_lila_test_set.py
#
# Create a test set of camera trap images, containing N empty and N non-empty 
# images from each LILA data set.
#

#%% Constants and imports

import json
import urllib.request
import tempfile
import zipfile
import os
import random

from urllib.parse import urlparse

# LILA camera trap master metadata file
metadata_url = 'http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt'

n_empty_images_per_dataset = 1
n_non_empty_images_per_dataset = 1

# We'll write images, metadata downloads, and temporary files here
lila_local_base = r'g:\temp\lila'

output_dir = os.path.join(lila_local_base,'lila_test_set')
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
        destination_filename = \
            os.path.join(temp_dir,url_as_filename)
            
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
    
    # Each line in this file is name/base_url/json_url/[box_url]
    tokens = s.split(',')
    assert len(tokens)==4
    url_mapping = {'sas_url':tokens[1],'json_url':tokens[2]}
    metadata_table[tokens[0]] = url_mapping
    
    assert 'https' not in tokens[0]
    assert 'https' in url_mapping['sas_url']
    assert 'https' in url_mapping['json_url']


#%% Download and extract metadata for every dataset

# ds_name = (list(metadata_table.keys()))[0]
for ds_name in metadata_table.keys():
    
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


#%% Choose images from each dataset

# ds_name = (list(metadata_table.keys()))[0]
for ds_name in metadata_table.keys():

    print('Choosing images for {}'.format(ds_name))
    
    json_filename = metadata_table[ds_name]['json_filename']
    
    with open(json_filename,'r') as f:
        d = json.load(f)
    
    category_id_to_name = {c['id']:c['name'] for c in d['categories']}
    category_name_to_id = {c['name']:c['id'] for c in d['categories']}
    
    if 'empty' not in category_name_to_id:
        print('Warning: no empty images available for {}'.format(ds_name))
        empty_category_id = -1
        empty_annotations = []
        empty_annotations_to_download = []
    else:
        empty_category_id = category_name_to_id['empty']        
        empty_annotations = [ann for ann in d['annotations'] if ann['category_id'] == empty_category_id]
        empty_annotations_to_download = random.sample(empty_annotations,n_empty_images_per_dataset)        
        
    non_empty_annotations = [ann for ann in d['annotations'] if ann['category_id'] != empty_category_id]
    
    non_empty_annotations_to_download = random.sample(non_empty_annotations,n_non_empty_images_per_dataset)
    annotations_to_download = empty_annotations_to_download + non_empty_annotations_to_download
    image_ids_to_download = set([ann['image_id'] for ann in annotations_to_download])
    assert len(image_ids_to_download) == len(set(image_ids_to_download))
    
    images_to_download = []
    for im in d['images']:
        if im['id'] in image_ids_to_download:
            images_to_download.append(im)
    assert len(images_to_download) == len(image_ids_to_download)
    
    metadata_table[ds_name]['images_to_download'] = images_to_download

# ...for each dataset


#%% Convert to URLs

# ds_name = (list(metadata_table.keys()))[0]
for ds_name in metadata_table.keys():

    sas_url = metadata_table[ds_name]['sas_url']
    
    base_url = sas_url.split('?')[0]    
    assert not base_url.endswith('/')
    
    # Retrieve image file names
    filenames = [im['file_name'] for im in metadata_table[ds_name]['images_to_download']]
    
    urls_to_download = []
    
    # Convert to URLs
    for fn in filenames:        
        url = base_url + '/' + fn
        urls_to_download.append(url)

    metadata_table[ds_name]['urls_to_download'] = urls_to_download
    
# ...for each dataset


#%% Download those image files

# ds_name = (list(metadata_table.keys()))[0]
for ds_name in metadata_table.keys():

    # This URL may not be a SAS URL, we will remove a SAS token if it's present
    sas_url = metadata_table[ds_name]['sas_url']
    
    base_url = sas_url.split('?')[0]    
    assert not base_url.endswith('/')
    base_url += '/'
        
    urls_to_download = metadata_table[ds_name]['urls_to_download']
    
    # url = urls_to_download[0]
    for url in urls_to_download:
        
        assert base_url in url
        output_file_relative = ds_name.lower().replace(' ','_') + '_' + url.replace(base_url,'').replace('/','_').replace('\\','_')
        output_file_absolute = os.path.join(output_dir,output_file_relative)
        try:
            download_url(url, destination_filename=output_file_absolute, force_download=False, verbose=True)
        except Exception as e:
            print('\n*** Error downloading {} ***\n{}'.format(url,str(e)))
        
    # ...for each url
    
# ...for each dataset


#
# create_lila_test_set.py
#
# Create a test set of camera trap images, containing N empty and N non-empty 
# images from each LILA data set.
#

#%% Constants and imports

import json
import os
import random

from data_management.lila.lila_common import read_lila_metadata, get_json_file_for_dataset

# from ai4eutils
from url_utils import download_url

n_empty_images_per_dataset = 1
n_non_empty_images_per_dataset = 1

# We'll write images, metadata downloads, and temporary files here
lila_local_base = os.path.expanduser('~/lila')

output_dir = os.path.join(lila_local_base,'lila_test_set')
os.makedirs(output_dir,exist_ok=True)

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)


#%% Download and parse the metadata file

metadata_table = read_lila_metadata(metadata_dir)


#%% Download and extract metadata for every dataset

for ds_name in metadata_table.keys():
    metadata_table[ds_name]['json_filename'] = get_json_file_for_dataset(ds_name=ds_name,
                                                                         metadata_dir=metadata_dir,
                                                                         metadata_table=metadata_table)


#%% Choose images from each dataset

# ds_name = (list(metadata_table.keys()))[0]
for ds_name in metadata_table.keys():

    print('Choosing images for {}'.format(ds_name))
    
    json_filename = metadata_table[ds_name]['json_filename']
    
    with open(json_filename,'r') as f:
        d = json.load(f)
    
    category_id_to_name = {c['id']:c['name'] for c in d['categories']}
    category_name_to_id = {c['name']:c['id'] for c in d['categories']}
    
    ## Find empty images
    
    if 'empty' not in category_name_to_id:
        empty_annotations_to_download = []
    else:
        empty_category_id = category_name_to_id['empty']        
        empty_annotations = [ann for ann in d['annotations'] if ann['category_id'] == empty_category_id]
        try:
            empty_annotations_to_download = random.sample(empty_annotations,n_empty_images_per_dataset)        
        except ValueError:
            print('No empty images available for dataset {}'.format(ds_name))
            empty_annotations_to_download = []
        
    ## Find non-empty images
    
    non_empty_annotations = [ann for ann in d['annotations'] if ann['category_id'] != empty_category_id]    
    try:
        non_empty_annotations_to_download = random.sample(non_empty_annotations,n_non_empty_images_per_dataset)
    except ValueError:
        print('No non-empty images available for dataset {}'.format(ds_name))
        non_empty_annotations_to_download = []

    
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


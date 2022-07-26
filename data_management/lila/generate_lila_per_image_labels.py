#
# generate_lila_per_image_labels.py
# 
# Generate a .csv file with one row per annotation, containing full URLs to every
# camera trap image on LILA, with taxonomically expanded labels.
#
# Typically there will be one row per image, though images with multiple annotations
# will have multiple rows.
#
# Some images may not physically exist, particularly images that are labeled as "human".
# This script does not validate image URLs.
#
# Does not include bounding box annotations.
#

#%% Constants and imports

import os
import json
import pandas as pd
import numpy as np

from collections import defaultdict
from tqdm import tqdm

from data_management.lila.lila_common import read_lila_metadata, \
    get_json_file_for_dataset, \
    read_lila_taxonomy_mapping

from url_utils import download_url

# array to fill for output
category_list = []

# We'll write images, metadata downloads, and temporary files here
lila_local_base = os.path.expanduser('~/lila')

os.makedirs(lila_local_base,exist_ok=True)

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)

output_file = os.path.join(lila_local_base,'lila_image_urls_and_labels.csv')


#%% Download and parse the metadata file

metadata_table = read_lila_metadata(metadata_dir)


#%% Download and extract metadata for the datasets we're interested in

for ds_name in metadata_table.keys():    
    metadata_table[ds_name]['json_filename'] = get_json_file_for_dataset(ds_name=ds_name,
                                                                         metadata_dir=metadata_dir,
                                                                         metadata_table=metadata_table)
    
#%% Load taxonomy data

taxonomy_df = read_lila_taxonomy_mapping(metadata_dir)


#%% Build a dictionary that maps each [dataset,query] pair to the full taxonomic label set

ds_label_to_taxonomy = {}

# i_row = 0; row = taxonomy_df.iloc[i_row]
for i_row,row in taxonomy_df.iterrows():
    
    ds_label = row['dataset_name'] + ':' + row['query']
    assert ds_label.strip() == ds_label
    assert ds_label not in ds_label_to_taxonomy
    ds_label_to_taxonomy[ds_label] = row.to_dict()
    

#%% Process annotations for each dataset

import csv

header = ['dataset_name','url','image_id','sequence_id','location_id','frame_num','original_label','scientific_name','common_name']

taxonomy_levels_to_include = \
    ['kingdom','phylum','subphylum','superclass','class','subclass','infraclass','superorder','order',
     'suborder','infraorder','superfamily','family','subfamily','tribe','genus','species','subspecies','variety']
    
header.extend(taxonomy_levels_to_include)

missing_annotations = set()

def clearnan(v):
    if isinstance(v,float):
        assert np.isnan(v)
        v = ''
    assert isinstance(v,str)
    return v

with open(output_file,'w') as f:
    
    csv_writer = csv.writer(f)
    csv_writer.writerow(header)
    
    # ds_name = list(metadata_table.keys())[0]
    for ds_name in metadata_table.keys():
        
        if 'bbox' in ds_name:
            print('Skipping bbox dataset {}'.format(ds_name))
            continue
    
        print('Processing dataset {}'.format(ds_name))
        
        json_filename = metadata_table[ds_name]['json_filename']
        with open(json_filename, 'r') as f:
            data = json.load(f)
        
        categories = data['categories']
        category_ids = [c['id'] for c in categories]
        for c in categories:
            category_id_to_name = {c['id']:c['name'] for c in categories}
            
        annotations = data['annotations']
        images = data['images']
        
        image_id_to_category_names = defaultdict(set)
        
        # Go through annotations, marking each image with the categories that are present
        #
        # ann = annotations[0]
        for ann in annotations:
            
            category_name = category_id_to_name[ann['category_id']]
            image_id_to_category_names[ann['image_id']].add(category_name)
    
        unannotated_images = []
        
        # im = images[10]
        for im in images:
            
            file_name = im['file_name'].replace('\\','/')
            base_url = metadata_table[ds_name]['sas_url']
            assert not base_url.endswith('/')
            url = base_url + '/' + file_name
            
            # Location, sequence, and image IDs are only guaranteed to be unique within
            # a dataset, so for the output .csv file, include both
            if 'location' in im:
                location_id = ds_name + ' : ' + str(im['location'])
            else:
                location_id = ds_name
                
            image_id = ds_name + ' : ' + str(im['id'])
            
            if 'seq_id' in im:
                sequence_id = ds_name + ' : ' + str(im['seq_id'])
            else:
                sequence_id = ds_name + ' : ' + 'unknown'
                
            if 'frame_num' in im:
                frame_num = im['frame_num']
            else:
                frame_num = -1
            
            categories_this_image = image_id_to_category_names[im['id']]
            
            if len(categories_this_image) == 0:
                unannotated_images.append(im)
                continue
            
            # category_name = list(categories_this_image)[0]
            for category_name in categories_this_image:
                
                ds_label = ds_name + ':' + category_name.lower()
                
                if ds_label not in ds_label_to_taxonomy:
                    
                    # Only print a warning the first time we see an unmapped label
                    if ds_label not in missing_annotations:
                        print('Warning: {} not in taxonomy file'.format(ds_label))
                    missing_annotations.add(ds_label)
                    continue
                
                taxonomy_labels = ds_label_to_taxonomy[ds_label]
                
                row = []
                row.append(ds_name)
                row.append(url)
                row.append(image_id)
                row.append(location_id)
                row.append(sequence_id)
                row.append(frame_num)
                row.append(taxonomy_labels['query'])
                row.append(clearnan(taxonomy_labels['scientific_name']))
                row.append(clearnan(taxonomy_labels['common_name']))
                
                for s in taxonomy_levels_to_include:
                    row.append(clearnan(taxonomy_labels[s]))
                    
                csv_writer.writerow(row)
                        
            # ...for each category that was applied at least once to this image
            
        # ...for each image
        print('{} of {} images are un-annotated\n'.format(len(unannotated_images),len(images)))
        
    # ...for each dataset

# ...with open()    


#%% Preview a sample of files to make sure everything worked

df = pd.read_csv(output_file)

print('Read {} lines from {}'.format(len(df),output_file))

n_empty_images_per_dataset = 3
n_non_empty_images_per_dataset = 3

preview_folder = os.path.join(lila_local_base,'csv_preview')
os.makedirs(preview_folder,exist_ok=True)


#%% Choose images to download

np.random.seed(0)

images_to_download = []

# ds_name = list(metadata_table.keys())[2]
for ds_name in metadata_table.keys():
    
    if 'bbox' in ds_name:
        continue
    
    # Find all rows for this dataset
    ds_rows = df.loc[df['dataset_name'] == ds_name]
    
    print('{} rows available for {}'.format(len(ds_rows),ds_name))
    assert len(ds_rows) > 0
    
    empty_rows = ds_rows[ds_rows['scientific_name'].isnull()]
    non_empty_rows = ds_rows[~ds_rows['scientific_name'].isnull()]
    
    if len(empty_rows) == 0:
        print('No empty images available for {}'.format(ds_name))
    else:
        empty_rows_to_download = empty_rows.sample(n=n_empty_images_per_dataset)
        images_to_download.extend(empty_rows_to_download.to_dict('records'))

    if len(non_empty_rows) == 0:
        print('No non-empty images available for {}'.format(ds_name))
    else:
        non_empty_rows_to_download = non_empty_rows.sample(n=n_non_empty_images_per_dataset)
        images_to_download.extend(non_empty_rows_to_download.to_dict('records'))
    
 # ...for each dataset

print('Selected {} total images'.format(len(images_to_download)))


#%% Download images

import urllib.request

# i_image = 0; image = images_to_download[i_image]
for i_image,image in tqdm(enumerate(images_to_download),total=len(images_to_download)):
    
    url = image['url']
    ext = os.path.splitext(url)[1]
    output_file = os.path.join(preview_folder,'image_{}'.format(str(i_image).zfill(4)) + ext)
    relative_file = os.path.relpath(output_file,preview_folder)
    try:
        download_url(url,output_file,verbose=False)
        image['relative_file'] = relative_file
    except urllib.error.HTTPError:
        print('Image {} does not exist ({}:{})'.format(
            i_image,image['dataset_name'],image['original_label']))
        image['relative_file'] = None


#%% Write HTML

import write_html_image_list

"""
filename: the output file

image: a list of image filenames or dictionaries with one or more of the following fields:
    
    filename
    imageStyle
    textStyle
    title
    linkTarget
"""

html_filename = os.path.join(preview_folder,'index.html')

html_images = []
for im in images_to_download:
    
    if im['relative_file'] is None:
        continue
    
    output_im = {}
    output_im['filename'] = im['relative_file']
    output_im['linkTarget'] = im['url']
    output_im['title'] = str(im)
    output_im['imageStyle'] = 'width:600px;'
    output_im['textStyle'] = 'font-weight:normal;font-size:100%;'
    html_images.append(output_im)
    
write_html_image_list.write_html_image_list(html_filename,html_images)

from path_utils import open_file
open_file(html_filename)

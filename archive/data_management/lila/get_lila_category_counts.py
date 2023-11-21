#
# get_lila_category_counts.py
#
# Count the number of images and bounding boxes with each label in one or more LILA datasets.
#
# This script doesn't write these counts out anywhere other than the console, it's just intended
# as a template for doing operations like this on LILA data.  get_lila_category_list.py writes 
# information out to a .json file, but it counts *annotations*, not *images*, for each category.
#

#%% Constants and imports

import json
import os

from collections import defaultdict

from data_management.lila.lila_common import read_lila_metadata, get_json_file_for_dataset

# If None, will use all datasets
datasets_of_interest = None

# We'll write images, metadata downloads, and temporary files here
lila_local_base = os.path.expanduser('~/lila')

metadata_dir = os.path.join(lila_local_base,'metadata')
os.makedirs(metadata_dir,exist_ok=True)


#%% Download and parse the metadata file

metadata_table = read_lila_metadata(metadata_dir)


#%% Download and extract metadata for the datasets we're interested in

if datasets_of_interest is None:
    datasets_of_interest = list(metadata_table.keys())

for ds_name in datasets_of_interest:    
    metadata_table[ds_name]['json_filename'] = get_json_file_for_dataset(ds_name=ds_name,
                                                                         metadata_dir=metadata_dir,
                                                                         metadata_table=metadata_table)
    
    
#%% Count categories

ds_name_to_category_counts = {}

# ds_name = datasets_of_interest[0]
for ds_name in datasets_of_interest:
    
    category_to_image_count = {}
    category_to_bbox_count = {}
    
    print('Counting categories in: ' + ds_name)
    
    json_filename = metadata_table[ds_name]['json_filename']
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    categories = data['categories']
    category_ids = [c['id'] for c in categories]
    for c in categories:
        category_id_to_name = {c['id']:c['name'] for c in categories}
    annotations = data['annotations']
    images = data['images']
    
    for category_id in category_ids:        
        category_name = category_id_to_name[category_id]        
        category_to_image_count[category_name] = 0
        category_to_bbox_count[category_name] = 0
        
    image_id_to_category_names = defaultdict(set)
    
    # Go through annotations, marking each image with the categories that are present
    #
    # ann = annotations[0]
    for ann in annotations:
        
        category_name = category_id_to_name[ann['category_id']]
        image_id_to_category_names[ann['image_id']].add(category_name)

    # Now go through images and count categories
    category_to_count = defaultdict(int)
    
    # im = images[0]
    for im in images:
        categories_this_image = image_id_to_category_names[im['id']]
        for category_name in categories_this_image:
            category_to_count[category_name] += 1

    ds_name_to_category_counts[ds_name] = category_to_count
    
# ...for each dataset
    

#%% Print the results

for ds_name in ds_name_to_category_counts:
    
    print('\n** Category counts for {} **\n'.format(ds_name))
    
    category_to_count = ds_name_to_category_counts[ds_name]
    
    for category_name in category_to_count.keys():        
        print('{}: {}'.format(category_name,category_to_count[category_name]))
        
# ...for each dataset


        
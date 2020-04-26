#
# download_lila_subset.py
#
# Example of how to download a list of files from LILA, e.g. all the files
# in a data set corresponding to a particular species.
#

#%% Constants and imports

import json
import urllib.request
import os
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

# SAS URLs come from:
#
# http://lila.science/?attachment_id=792
#
# In this example, we're using the Missouri Camera Traps data set
sas_url = 'https://lilablobssc.blob.core.windows.net/missouricameratraps/images?st=2020-01-01T00%3A00%3A00Z&se=2034-01-01T00%3A00%3A00Z&sp=rl&sv=2019-07-07&sr=c&sig=zf5Vb3BmlGgBKBM1ZtAZsEd1vZvD6EbN%2BNDzWddJsUI%3D'

# This assumes you've downloaded the metadata file from LILA
json_filename = r'd:\temp\missouri_camera_traps_set1.json'

output_dir = r'd:\temp\missouri_camera_traps_subset'

species_of_interest = 'red_fox'

# We will demonstrate two approaches to downloading, one that loops over files
# and downloads directly in Python, another that uses AzCopy.
#
# AzCopy will generally be more performant and supports resuming if the 
# transfers are interrupted.  It assumes that azcopy is on the system path.
use_azcopy_for_download = False

overwrite_files = False

# Number of concurrent download threads (when not using AzCopy)
n_download_threads = 50


#%% Environment prep and derived constants

base_url = sas_url.split('?')[0]
sas_token = sas_url.split('?')[1]
os.makedirs(output_dir,exist_ok=True)


#%% Open the metadata file

with open(json_filename, 'r') as f:
    data = json.load(f)

categories = data['categories']
annotations = data['annotations']
images = data['images']


#%% Build a list of image files (relative path names) that match the target species

# Retrieve the category ID we're interested in
category_of_interest = list(filter(lambda x: x['name'] == species_of_interest, categories))
assert len(category_of_interest) == 1
category_of_interest = category_of_interest[0]
category_id_of_interest = category_of_interest['id']

# Retrieve all the images that match that category
image_ids_of_interest = set([ann['image_id'] for ann in annotations if ann['category_id'] == category_id_of_interest])

print('Selected {} of {} images'.format(len(image_ids_of_interest),len(images)))

# Retrieve image file names
filenames = [im['file_name'] for im in images if im['id'] in image_ids_of_interest]
assert len(filenames) == len(image_ids_of_interest)


#%% Support functions

def download_image(fn):
    
    url = base_url + '/' + fn
    target_file = os.path.join(output_dir,fn)
    if ((not overwrite_files) and (os.path.isfile(target_file))):
        # print('Skipping file {}'.format(fn))
        return
    
    # print('Downloading {} to {}'.format(url,target_file))        
    os.makedirs(os.path.dirname(target_file),exist_ok=True)
    urllib.request.urlretrieve(
        url,target_file)

    
#%% Download those image files

if use_azcopy_for_download:
    
    print('Downloading images for {0} with azcopy'.format(species_of_interest))
    
    # Write out a list of files, and use the azcopy "list-of-files" option to download those files
    # this azcopy feature is unofficially documented at https://github.com/Azure/azure-storage-azcopy/wiki/Listing-specific-files-to-transfer
    az_filename = os.path.join(output_dir, 'filenames.txt')
    with open(az_filename, 'w') as f:
        for fn in filenames:
            f.write(fn.replace('\\','/') + '\n')
    cmd = 'azcopy cp "{0}" "{1}" --list-of-files "{2}"'.format(
            sas_url, output_dir, az_filename)            
    os.system(cmd)
    
else:
    
    # Loop over files
    print('Downloading images for {0} without azcopy'.format(species_of_interest))
    
    if n_download_threads <= 1:
    
        for fn in tqdm(filenames):        
            download_image(fn)
        
    else:
    
        pool = ThreadPool(n_download_threads)        
        tqdm(pool.imap(download_image, filenames), total=len(filenames))
    
print('Done!')

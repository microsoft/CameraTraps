#
# make_eMammal_json.py
#
# Produces the COCO-formatted json database for an eMammal dataset, i.e. a 
# collection of folders, each of which contains a deployment_manifest.xml file.
#
# In this process, each image needs to be loaded to size it.
#
# To add bounding box annotations to the resulting database, use 
# add_annotations_to_eMammal_json.py.
#

#%% Constants and imports

# Either add the eMammal directory to your path, or run from there
# os.chdir(r'd:\git\CameraTraps\database_tools\eMammal')

import json
import multiprocessing
import os
# import warnings
import eMammal_helpers as helpers

from datetime import datetime, date
from multiprocessing.dummy import Pool as ThreadPool
from lxml import etree
from tqdm import tqdm

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
# warnings.filterwarnings('ignore')

# Should we run the image size retrieval in parallel?
run_parallel = False

output_dir_path = r'd:\path'
deployments_path = r'd:\other_path'
db_filename = 'apr.json'
corrupt_images_db_filename = 'apr_corrupt.json'

description = 'description'
version = '1.0'
contributor = 'contributor'
curator = '.json created by curator'


#%% Support functions

def _add_image(entry, full_img_path):    
    """ 
    Open the image to get size information and add height and width to the image entry. 
    """
    
    img_width, img_height = helpers.get_img_size(full_img_path)
    if img_width == -1 or img_height == -1:
        corrupt_images.append(full_img_path)
        return None
    entry['width'] = img_width
    entry['height'] = img_height
    pbar.update(1)
    return entry


#%% Main loop (metadata processing; image sizes are retrieved later)
    
print('Creating tasks to get all images...')
start_time = datetime.now()
tasks = []
folders = os.listdir(deployments_path)

all_species_strings = set()

# deployment = folders[0]
for deployment in tqdm(folders):
    
    deployment_path = os.path.join(deployments_path, deployment)
    manifest_path = os.path.join(deployment_path, 'deployment_manifest.xml')

    assert os.path.isfile(manifest_path)
    
    with open(manifest_path, 'r') as f:
        tree = etree.parse(f)

    root = tree.getroot()
    project_id = root.findtext('ProjectId')
    deployment_id = root.findtext('CameraDeploymentID')
    deployment_location = root.findtext('CameraSiteName')

    image_sequences = root.findall('ImageSequence')
    
    # sequence = image_sequences[0]
    for sequence in image_sequences:
        
        seq_id = sequence.findtext('ImageSequenceId')

        # get species info for this sequence
        researcher_identifications = sequence.findall('ResearcherIdentifications')
        species = set()

        for researcher_id in researcher_identifications:
            identifications = researcher_id.findall('Identification')
            for id in identifications:
                species_common_name = helpers.clean_species_name(id.findtext('SpeciesCommonName'))
                species.add(species_common_name)

        species_str = ';'.join(sorted(list(species)))
        all_species_strings.add(species_str)
        
        # add each image's info to database
        images = sequence.findall('Image')
        seq_num_frames = len(images) # total number of frames in this sequence
        assert isinstance(seq_num_frames,int) and seq_num_frames > 0 # historical quirk
        
        # img = images[0]
        for img in images:
            
            img_id = img.findtext('ImageId')
            img_file_name = img.findtext('ImageFileName')
            assert img_file_name.lower().endswith('.jpg')

            img_frame = img.findtext('ImageOrder')
            if img_frame == '' or img_frame is None:
                # some manifests don't have the ImageOrder info, but the info is in the file name
                img_frame = img_file_name.split('i')[1].split('.')[0]

            # full_img_id has no frame info
            #
            # frame number only used in requests to iMerit for ordering
            full_img_id = 'datasetemammal.project{}.deployment{}.seq{}.img{}'.format(
                project_id, deployment_id, seq_id, img_id)            
            full_img_path = os.path.join(deployment_path, img_file_name)

            img_datetime, datetime_err = helpers.parse_timestamp(img.findtext('ImageDateTime'))
            if datetime_err:
                print('WARNING datetime parsing error for image {}. Error: {}'.format(
                    full_img_path, datetime_err))

            entry = {
                'id': full_img_id,
                'width': 0,  # place holders
                'height': 0,
                'file_name': os.path.join(deployment, img_file_name),
                'location': deployment_location,
                'datetime': str(img_datetime),
                'seq_id': seq_id,
                'frame_num': int(img_frame),
                'seq_num_frames': seq_num_frames,
                'label': species_str  # extra field for eMammal
            }

            tasks.append((entry, full_img_path))
        
        # ...for each image
        
    # ...for each sequence
    
# ...for each deployment
    
print('Finished creating tasks to get images.')


#%% Get image sizes

# 'tasks' is currently a list of 2-tuples, with each entry as [image dictionary,path].
# 
# Go through that and copy just the image dictionaries to 'db_images', adding size
# information to each entry.  Takes a couple hours.

db_images = []
corrupt_images = []
pbar = tqdm(total=len(tasks))

if run_parallel:
    # opening each image seems too fast for this multi-threaded version to be faster than sequential code.
    num_workers = multiprocessing.cpu_count()
    pool = ThreadPool(num_workers)
    db_images = pool.starmap(_add_image, tasks)
    print('Waiting for image processes to finish...')
    pool.close()
    pool.join()
else:
    print('Finding image size sequentially')
    for entry, full_img_path in tasks:
        db_images.append(_add_image(entry, full_img_path))

db_images = [i for i in db_images if i is not None]

print('{} images could not be opened:'.format(len(corrupt_images)))
print(corrupt_images)
print('Done getting image sizes')


#%% Assemble top-level dictionaries

db_info = {
    'year': 'unkown',
    'version': version,
    'description': description,
    'contributor': contributor,
    'curator': curator,
    'date_created': str(date.today())
}

coco_formatted_json = {
    'info': db_info,
    'images': db_images
}


#%% Write out .json

print('Saving the json database to disk...')
with open(os.path.join(output_dir_path, db_filename), 'w') as f:
    json.dump(coco_formatted_json, f, indent=4, sort_keys=True)
print('...done')

print('Saving list of corrupt images...')
with open(os.path.join(output_dir_path, corrupt_images_db_filename), 'w') as f:
    json.dump(corrupt_images, f, indent=4)    
print('...done')

print('Running the script took {}.'.format(datetime.now() - start_time))



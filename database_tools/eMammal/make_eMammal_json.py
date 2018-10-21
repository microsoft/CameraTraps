import json
import multiprocessing
import os
import warnings
from datetime import datetime, date
from multiprocessing.dummy import Pool as ThreadPool  # this functions like threading

import eMammal_helpers as helpers
from lxml import etree
from tqdm import tqdm

warnings.filterwarnings('ignore')
# ignoring all PIL cannot read EXIF metainfo for the images warnings


# make_eMammal_json.py
#
# Produces the COCO formatted json database for the eMammal dataset, with only image information.
# This contains all the images whether they were annotated with bounding boxes or not.
# In this process, each image needs to be loaded to size it.
#
# To add annotations to the resulting database with only images to produce the complete COCO formatted
# json, use add_annotations_to_eMammal_json.py.
#
# At DEPLOYMENTS_PATH, the top level folders are one for each deployment, since the folder
# name has both projectID and deploymentID, uniquely identifying a deployment.
#
# This script is analogous to make_per_season_SS_json.py, but without adding any annotations.


# configurations and paths
run_parallel = False

output_dir_path = '/home/yasiyu/yasiyu_temp/eMammal_db'
deployments_path = '/datadrive/emammal'


def _add_image(entry, full_img_path):
    """ Open the image to get size information and add height and width to the image entry. """
    img_width, img_height = helpers.get_img_size(full_img_path)
    if img_width == -1 or img_height == -1:
        corrupt_images.append(full_img_path)
        return None
    entry['width'] = img_width
    entry['height'] = img_height
    pbar.update(1)
    return entry


print('Creating tasks to get all images...')
start_time = datetime.now()
tasks = []

for deployment in tqdm(os.listdir(deployments_path)):
    deployment_path = os.path.join(deployments_path, deployment)
    manifest_path = os.path.join(deployment_path, 'deployment_manifest.xml')

    with open(manifest_path, 'r') as f:
        tree = etree.parse(f)

    root = tree.getroot()
    project_id = root.findtext('ProjectId')
    deployment_id = root.findtext('CameraDeploymentID')
    deployment_location = root.findtext('CameraSiteName')

    image_sequences = root.findall('ImageSequence')
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

        # add each image's info to database
        images = sequence.findall('Image')
        seq_num_frames = len(images)[0]  # total number of frames in this sequence
        for img in images:
            img_id = img.findtext('ImageId')
            img_file_name = img.findtext('ImageFileName')
            assert img_file_name.lower().endswith('.jpg')  # some are .JPG and some are .jpg

            # img_frame info added to DB for potential motion based studies; it is obtained best-effort
            img_frame = img.findtext('ImageOrder')
            if img_frame == '' or img_frame is None:
                # some Robert Long xml doesn't have the ImageOrder info, but the info is in the file name
                img_frame = img_file_name.split('i')[1].split('.')[0]

            # note that the full_img_id has no frame info.
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
print('Finished creating tasks to get images.')

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
print('Done with images.')

db_info = {
    'year': 2018,
    'version': '0.0.1',
    'description': 'eMammal dataset containing 3140 deployments, in COCO format.',
    'contributor': 'eMammal',
    'date_created': str(date.today())
}

coco_formatted_json = {
    'info': db_info,
    'images': db_images
}

print('Saving the json database to disk...')
with open(os.path.join(output_dir_path, 'eMammal_images.json'), 'w') as f:
    json.dump(coco_formatted_json, f, indent=4, sort_keys=True)

with open(os.path.join(output_dir_path, 'eMammal_corrupt_images.json'), 'w') as f:
    json.dump(corrupt_images, f, indent=4)

print('Running the script took {}.'.format(datetime.now() - start_time))



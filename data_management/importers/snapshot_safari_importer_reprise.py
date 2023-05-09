#
# snapshot_safar_importer_reprise.py
#
# This is a 2023 update to snapshot_safari_importer.py.  We do a bunch of things now that
# we didn't do the last time we imported Snapshot data (like updating the big taxonomy)
# file, and we skip a bunch of things now that we used to do (like generating massive
# zipfiles).  So, new year, new importer.
#

#%% Constants and imports

import os
import glob
import json
import shutil
import random

import pandas as pd

from tqdm import tqdm
from collections import defaultdict

input_base = '/media/user/Elements'
output_base = os.path.expanduser('~/data/snapshot-safari-metadata')
file_list_cache_file = os.path.join(output_base,'file_list.json')

assert os.path.isdir(input_base)
os.makedirs(output_base,exist_ok=True)

# We're going to copy all the .csv files to a faster location
annotation_cache_dir = os.path.join(output_base,'csv_files')
os.makedirs(annotation_cache_dir,exist_ok=True)


#%% List files

# Do a one-time enumeration of the entire drive; this will take a long time,
# but will save a lot of hassle later.

if os.path.isfile(file_list_cache_file):
    print('Loading file list from {}'.format(file_list_cache_file))
    with open(file_list_cache_file,'r') as f:
        all_files = json.load(f)    
else:
    all_files = glob.glob(os.path.join(input_base,'**','*.*'),recursive=True)
    all_files = [fn for fn in all_files if '$RECYCLE.BIN' not in fn]
    all_files = [fn for fn in all_files if 'System Volume Information' not in fn]
    print('Enumerated {} files'.format(len(all_files)))
    with open(file_list_cache_file,'w') as f:
        json.dump(all_files,f,indent=1)
    print('Wrote file list to {}'.format(file_list_cache_file))


#%% Create derived lists

# Takes about 60 seconds

all_files_relative = [os.path.relpath(fn,input_base) for fn in all_files]
all_files_relative = [fn.replace('\\','/') for fn in all_files_relative]
all_files_relative_set = set(all_files_relative)

# CSV files are one of:
#
# _report_lila.csv (this is the one we want to use, with the species/count/etc. for each sequence)
# _report_lila_image_inventory.csv (maps captures to images)
# _report_lila_overview.csv (distrubution of species)
csv_files = [fn for fn in all_files_relative if fn.endswith('.csv')]


#%% Copy all csv files to the annotation cache folder

# fn = csv_files[0]
for fn in csv_files:
    target_file = os.path.join(annotation_cache_dir,os.path.basename(fn))
    source_file = os.path.join(input_base,fn)
    shutil.copyfile(source_file,target_file)    

def read_cached_csv_file(fn):
    cached_csv_file = os.path.join(annotation_cache_dir,os.path.basename(fn))
    df = pd.read_csv(cached_csv_file)
    return df


#%% List project folders

# Project folders look like one of these:
#
# APN
# Snapshot Cameo/DEB
project_code_to_project_folder = {}

folders = os.listdir(input_base)
folders = [fn for fn in folders if (not fn.startswith('$') and \
                                    not 'System Volume' in fn)]

for fn in folders:
    if len(fn) == 3:
        assert fn not in project_code_to_project_folder
        project_code_to_project_folder[fn] = fn
    else:
        assert 'Snapshot' in fn
        subfolders = os.listdir('/'.join([input_base,fn]))
        for subfn in subfolders:
            assert len(subfn) == 3
            assert subfn not in project_code_to_project_folder
            project_code_to_project_folder[subfn] = '/'.join([fn,subfn])

project_folder_to_project_code = {v: k for k, v in project_code_to_project_folder.items()}
project_codes = sorted(list(project_code_to_project_folder.keys()))
project_folders = sorted(list(project_code_to_project_folder.values()))

def file_to_project_folder(fn):
    
    tokens = fn.split('/')
    if len(tokens[0]) == 3:
        project_folder = tokens[0]
    else:
        assert 'Snapshot' in tokens[0]
        project_folder = '/'.join(tokens[0:2])
    assert project_folder in project_folders
    return project_folder

def file_to_project_code(fn):
    
    return project_folder_to_project_code[file_to_project_folder(fn)]


#%% Map report and inventory files to codes

project_code_to_report_files = defaultdict(list)

# fn = csv_files[0]
for fn in csv_files:
    if 'report_lila.csv' not in fn:
        continue
    project_code = project_folder_to_project_code[file_to_project_folder(fn)]
    project_code_to_report_files[project_code].append(fn)

project_codes_with_no_reports = set()

for project_code in project_code_to_project_folder.keys():
    if project_code not in project_code_to_report_files:
        project_codes_with_no_reports.add(project_code)
        print('Warning: no report files available for {}'.format(project_code))


#%% Make sure that every report has a corresponding inventory file

all_report_files = [item for sublist in project_code_to_report_files.values() \
                    for item in sublist]
for fn in all_report_files:
    inventory_file = fn.replace('.csv','_image_inventory.csv')
    assert inventory_file in csv_files
    
    
#%% Count species based on overview and report files

# The overview and report files should produce the same counts; we'll verify this
# in the next cell.

species_to_count_overview = defaultdict(int)
species_to_count_report = defaultdict(int)

for report_file in all_report_files:
        
    overview_file = report_file.replace('.csv','_overview.csv')    
    
    df = read_cached_csv_file(overview_file)
    
    for i_row,row in df.iterrows():
        
        if row['question'] == 'question__species':
            
            assert isinstance(row['answer'],str)
            assert isinstance(row['count'],int)
            species = row['answer']
            
            if len(species) < 3:
                assert species == '0' or species == '1'
                
            species_to_count_overview[species] += row['count']
    
    # ...for each capture in the overview file
    
    df = read_cached_csv_file(report_file)
    
    for i_row,row in df.iterrows():
        
        species = row['question__species']
        assert isinstance(species,str)
        
        # Ignore results from the blank/non-blank workflow
        if len(species) < 3:
            assert species == '0' or species == '1'                
        species_to_count_report[species] += 1
                    
    # ...for each capture in the report file
    
# ...for each report file
    

#%% Print counts

species_to_count_overview_sorted = \
    {k: v for k, v in sorted(species_to_count_overview.items(), 
                             key=lambda item: item[1], reverse=True)}
species_to_count_report_sorted = \
    {k: v for k, v in sorted(species_to_count_report.items(), 
                             key=lambda item: item[1], reverse=True)}

string_count = 0
non_blank_count = 0

for species in species_to_count_overview_sorted.keys():        
    
    # The overview and report files should produce the same counts
    assert species_to_count_overview_sorted[species] == \
        species_to_count_report[species]    
    count = species_to_count_overview_sorted[species]
    if species not in ('0','1'):
        string_count += count
        if species != 'blank':
            non_blank_count += count
            
    print('{}{}'.format(species.ljust(25),count))

n_images = len(all_files)
n_sequences = sum(species_to_count_overview_sorted.values())

print('\n{} total images\n{} total sequences'.format(n_images,n_sequences))

print('\nString count: {}'.format(string_count))
print('Non-blank count: {}'.format(non_blank_count))


#%% Make sure that capture IDs in the reports/inventory files match

# ...and confirm that (almost) all the images in the inventory tables are 
# present on disk.

all_relative_paths_in_inventory = set()
files_missing_on_disk = []

for report_file in all_report_files:
        
    project_base = file_to_project_folder(report_file)                
    inventory_file = report_file.replace('.csv','_image_inventory.csv')    
    
    inventory_df = read_cached_csv_file(inventory_file)
    report_df = read_cached_csv_file(report_file)
    
    capture_ids_in_report = set()
    for i_row,row in report_df.iterrows():
        capture_ids_in_report.add(row['capture_id'])
    
    capture_ids_in_inventory = set()
    for i_row,row in inventory_df.iterrows():
        
        capture_ids_in_inventory.add(row['capture_id'])
        image_path_relative = project_base + '/' + row['image_path_rel']
        
        # assert image_path_relative in all_files_relative_set
        if image_path_relative not in all_files_relative_set:
            
            # Make sure this isn't just a case issue
            assert image_path_relative.replace('.JPG','.jpg') \
                not in all_files_relative_set
            assert image_path_relative.replace('.jpg','.JPG') \
                not in all_files_relative_set
            files_missing_on_disk.append(image_path_relative)
            
        assert image_path_relative not in all_relative_paths_in_inventory
        all_relative_paths_in_inventory.add(image_path_relative)
                
    # Make sure the set of capture IDs appearing in this report is
    # the same as the set of capture IDs appearing in the corresponding
    # inventory file.
    assert capture_ids_in_report == capture_ids_in_inventory

# ...for each report file

print('\n{} missing files (of {})'.format(
    len(files_missing_on_disk),len(all_relative_paths_in_inventory)))

    
#%% For all the files we have on disk, see which are and aren't in the inventory files

# There aren't any capital-P .PNG files, but if I don't include .PNG
# in this list, I'll look at this in a year and wonder whether I forgot
# to include it.
image_extensions = set(['.JPG','.jpg','.PNG','.png'])

images_not_in_inventory = []
n_images_in_inventoried_projects = 0

# fn = all_files_relative[0]
for fn in tqdm(all_files_relative):

    if os.path.splitext(fn)[1] not in image_extensions:
        continue
    project_code = file_to_project_code(fn)
    if project_code in project_codes_with_no_reports:
        # print('Skipping project {}'.format(project_code))
        continue
    n_images_in_inventoried_projects += 1
    if fn not in all_relative_paths_in_inventory:
        images_not_in_inventory.append(fn)

print('\n{} images on disk are not in inventory (of {} in eligible projects)'.format(
    len(images_not_in_inventory),n_images_in_inventoried_projects))


#%% Map captures to images, and vice-versa

capture_id_to_images = defaultdict(list)
image_to_capture_id = {}

# report_file = all_report_files[0]
for report_file in tqdm(all_report_files):
        
    inventory_file = report_file.replace('.csv','_image_inventory.csv')    
    inventory_df = read_cached_csv_file(inventory_file)
    
    project_folder = file_to_project_folder(inventory_file)
    
    # row = inventory_df.iloc[0]
    for i_row,row in inventory_df.iterrows():
    
        capture_id = row['capture_id']
        image_file_relative = os.path.join(project_folder,row['image_path_rel'])
        capture_id_to_images[capture_id].append(image_file_relative)
        assert image_file_relative not in image_to_capture_id
        image_to_capture_id[image_file_relative] = capture_id
        
    # ...for each row (one image per row)
        
# ...for each report file
    

#%% Map captures to species (just species for now, we'll go back and get other metadata later)

capture_id_to_species = defaultdict(list)

for project_code in tqdm(project_codes):
    
    report_files = project_code_to_report_files[project_code]
    
    for report_file in report_files:
        
        report_df = read_cached_csv_file(report_file)

        for i_row,row in report_df.iterrows():
        
            capture_id = row['capture_id']
            species = row['question__species']
            capture_id_to_species[capture_id].append(species)

        # ...for each row
        
    # ...for each report file in this project
    
# ...for each project


#%% Take a look at the annotations "0" and "1"

captures_0 = []
captures_1 = []
captures_1_alone = []
captures_1_with_species = []

for capture_id in tqdm(capture_id_to_species):
    
    species_this_capture_id = capture_id_to_species[capture_id]
    
    # Multiple rows may be present for a capture, but they should be unique
    assert len(species_this_capture_id) == len(set(species_this_capture_id))
    
    if '0' in species_this_capture_id:
        captures_0.append(capture_id)
        # '0' should always appear alone
        assert len(species_this_capture_id) == 1

    if '1' in species_this_capture_id:
        captures_1.append(capture_id)
        assert '0' not in species_this_capture_id
        # '1' should never appear alone
        # assert len(species_this_capture_id) > 1
        if len(species_this_capture_id) == 1:
            captures_1_alone.append(capture_id)
        else:
            captures_1_with_species.append(capture_id)

# ...for each capture ID

print('')
print('Number of captures with "0" as the species (always appears alone): {}'.format(len(captures_0)))
print('Number of captures with "1" as the species: {}'.format(len(captures_1)))
print('Number of captures with "1" as the species, with no other species: {}'.format(
    len(captures_1_alone)))
print('Number of captures with "1" as the species, with other species: {}'.format(
    len(captures_1_with_species)))


#%% Sample some of those captures with mysterious "0" and "1" annotations

random.seed(0)
n_to_sample = 500
captures_0_samples = random.sample(captures_0,n_to_sample)
captures_1_samples = random.sample(captures_1,n_to_sample)

capture_0_sample_output_folder = os.path.join(output_base,'capture_0_samples')
capture_1_sample_output_folder = os.path.join(output_base,'capture_1_samples')
os.makedirs(capture_0_sample_output_folder,exist_ok=True)
os.makedirs(capture_1_sample_output_folder,exist_ok=True)

def copy_sampled_captures(sampled_captures,sample_capture_output_folder):

    for capture_id in tqdm(sampled_captures):    
        images_this_capture = capture_id_to_images[capture_id]
        for fn in images_this_capture:            
            # assert fn in all_files_relative_set
            if fn not in all_files_relative_set:
                print('Warning: missing file {}'.format(fn))
                continue
            source_image = os.path.join(input_base,fn)
            target_image = os.path.join(sample_capture_output_folder,os.path.basename(fn))
            shutil.copyfile(source_image,target_image)
        # ....for each image
    # ...for each capture
        
copy_sampled_captures(captures_0_samples,capture_0_sample_output_folder)    
copy_sampled_captures(captures_1_samples,capture_1_sample_output_folder)


#%% Find images that MD thinks contain people

md_results_folder = os.path.expanduser(
    '~/postprocessing/snapshot-safari/snapshot-safari-2023-04-21-v5a.0.0/json_subsets')
md_results_files = os.listdir(md_results_folder)

md_human_detection_threshold = 0.2
md_vehicle_detection_threshold = 0.2

# We'll make sure this is actually correct for all the files we load
md_human_category = '2'
md_vehicle_category = '3'

md_human_images = set()
md_vehicle_images = set()

# project_code = project_codes[0]
for project_code in project_codes:
    
    print('Finding human images for {}'.format(project_code))
    
    project_folder = project_code_to_project_folder[project_code]
    
    md_results_file = [fn for fn in md_results_files if project_code in fn]
    assert len(md_results_file) == 1
    md_results_file = os.path.join(md_results_folder,md_results_file[0])
    
    with open(md_results_file,'r') as f:
        md_results = json.load(f)
    assert md_results['detection_categories'][md_human_category] == 'person'
    assert md_results['detection_categories'][md_vehicle_category] == 'vehicle'
    
    # im = md_results['images'][0]
    for im in tqdm(md_results['images']):
        
        if 'detections' not in im:
            continue
        
        # MD results files are each relative to their own projects, we want
        # filenames to be relative to the base of the drive
        fn = os.path.join(project_folder,im['file'])
        for det in im['detections']:
            if det['category'] == md_human_category and \
                det['conf'] >= md_human_detection_threshold:
                    md_human_images.add(fn)
            if det['category'] == md_vehicle_category and \
                det['conf'] >= md_vehicle_detection_threshold:
                    md_vehicle_images.add(fn)
                    
        # ...for each detection
        
    # ...for each image
                    
# ...for each project

print('MD found {} human images, {} vehicle images'.format(
    len(md_human_images),len(md_vehicle_images)))

md_human_or_vehicle_images = \
    set(md_human_images).union(set(md_vehicle_images))
    
# next(iter(md_human_or_vehicle_images))


#%% Find images where the ground truth says humans or vehicles are present

human_species_id = 'human'
vehicle_species_id = 'humanvehicle'

gt_human_capture_ids = set()
gt_vehicle_capture_ids = set()

for capture_id in capture_id_to_species:
    
    species_this_capture_id = capture_id_to_species[capture_id]
    
    for species in species_this_capture_id:
        if species == human_species_id:
            gt_human_capture_ids.add(capture_id)
        elif species == vehicle_species_id:
            gt_vehicle_capture_ids.add(capture_id)
    
# ...for each capture ID

gt_human_images = []
gt_vehicle_images = []

for capture_id in gt_human_capture_ids:
    images_this_capture_id = capture_id_to_images[capture_id]
    gt_human_images.extend(images_this_capture_id)
for capture_id in gt_vehicle_capture_ids:
    images_this_capture_id = capture_id_to_images[capture_id]
    gt_vehicle_images.extend(images_this_capture_id)    
    
print('Ground truth includes {} human images ({} captures), {} vehicle images ({} captures)'.format(
    len(gt_human_images),len(gt_human_capture_ids),
    len(gt_vehicle_images),len(gt_vehicle_capture_ids)))

ground_truth_human_or_vehicle_images = \
    set(gt_human_images).union(set(gt_vehicle_images))
    
# next(iter(ground_truth_human_or_vehicle_images))


#%% Find mismatches

gt_missing_human_images = []
gt_missing_vehicle_images = []

for fn in md_human_images:
    if fn not in ground_truth_human_or_vehicle_images:
        gt_missing_human_images.append(fn)

for fn in md_vehicle_images:
    if fn not in ground_truth_human_or_vehicle_images:
        gt_missing_vehicle_images.append(fn)
        
print('Of {} images where MD found a human, {} are not in the ground truth'.format(
    len(md_human_images),len(gt_missing_human_images)))

print('Of {} images where MD found a vehicle, {} are not in the ground truth'.format(
    len(md_vehicle_images),len(gt_missing_vehicle_images)))


#%% Sample mismatches

random.seed(0)
n_to_sample = 1000
sampled_human_mismatches = random.sample(gt_missing_human_images,n_to_sample)
sampled_vehicle_mismatches = random.sample(gt_missing_vehicle_images,n_to_sample)

human_mismatch_output_folder = os.path.join(output_base,'mismatches_human')
vehicle_mismatch_output_folder = os.path.join(output_base,'mismatches_vehicle')
os.makedirs(human_mismatch_output_folder,exist_ok=True)
os.makedirs(vehicle_mismatch_output_folder,exist_ok=True)

def copy_sampled_images(sampled_images,sampled_images_output_folder):

    for fn in tqdm(sampled_images): 
        if fn not in all_files_relative_set:
            print('Warning: missing file {}'.format(fn))
            continue
        source_image = os.path.join(input_base,fn)
        target_image = os.path.join(sampled_images_output_folder,os.path.basename(fn))
        shutil.copyfile(source_image,target_image)
    
copy_sampled_images(sampled_human_mismatches,human_mismatch_output_folder)
copy_sampled_images(sampled_vehicle_mismatches,vehicle_mismatch_output_folder)


#%% See what's up with some of the mismatches

filename_base_to_filename = {}

from path_utils import is_image_file

# fn = all_files_relative[0]
for fn in tqdm(all_files_relative):

    if not is_image_file(fn):
        continue
    if 'Indiv_Recognition' in fn:
        continue
    bn = os.path.basename(fn)
    assert bn not in filename_base_to_filename
    filename_base_to_filename[bn] = fn


if False:

    bn = 'TSW_S2_KA02_R3_IMAG0002.JPG'
    fn = filename_base_to_filename[bn]
    capture_id = image_to_capture_id[fn]
    species = capture_id_to_species[capture_id]
    
    
#%% Look at the distribution of labels for these mismatched images    

gt_missing_images = set(gt_missing_human_images).union(set(gt_missing_vehicle_images))

missing_image_species_to_count = defaultdict(int)

for fn in gt_missing_images:
    if fn not in image_to_capture_id:
        continue
    capture_id = image_to_capture_id[fn]
    species = capture_id_to_species[capture_id]
    for s in species:
        missing_image_species_to_count[s] += 1

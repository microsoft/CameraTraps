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

import pandas as pd

from tqdm import tqdm
from collections import defaultdict

input_base = 'f:\\'
output_base = r'g:\temp\snapshot-safari-metadata'
file_list_cache_file = os.path.join(output_base,'file_list.json')

assert os.path.isdir(input_base)
os.makedirs(output_base,exist_ok=True)


#%% List files

# Do a one-time enumeration of the entire drive; this will take a long time,
# but will save a lot of hassle later.

if os.path.isfile(file_list_cache_file):
    print('Loading file list from {}'.format(file_list_cache_file))
    with open(file_list_cache_file,'r') as f:
        all_files = json.load(f)    
else:
    all_files = glob.glob(os.path.join(input_base,'**','*.*'),recursive=True)
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


#%% List project folders

# Project folders look like one of these:
#
# APN
# Snapshot Cameo/DEB
project_code_to_project_folder = {}

folders = os.listdir(input_base)
folders = [fn for fn in folders if (not fn.startswith('$') and not 'System Volume' in fn)]

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

all_report_files = [item for sublist in project_code_to_report_files.values() for item in sublist]
for fn in all_report_files:
    inventory_file = fn.replace('.csv','_image_inventory.csv')
    assert inventory_file in csv_files
    
    
#%% Count species based on overview and report files

species_to_count_overview = defaultdict(int)
species_to_count_report = defaultdict(int)

for project_code in tqdm(project_codes):
    
    report_files = project_code_to_report_files[project_code]
    
    for report_file in report_files:
        
        overview_file = report_file.replace('.csv','_overview.csv')    
        
        df = pd.read_csv(os.path.join(input_base,overview_file))
        for i_row,row in df.iterrows():
            if row['question'] == 'question__species':
                assert isinstance(row['answer'],str)
                assert isinstance(row['count'],int)
                species = row['answer']
                species_to_count_overview[species] += row['count']
                
        df = pd.read_csv(os.path.join(input_base,report_file))
        for i_row,row in df.iterrows():
            species = row['question__species']
            assert isinstance(species,str)
            species_to_count_report[species] += 1

            
#%% Print counts

species_to_count_overview_sorted = \
    {k: v for k, v in sorted(species_to_count_overview.items(), key=lambda item: item[1], reverse=True)}
species_to_count_report_sorted = \
    {k: v for k, v in sorted(species_to_count_report.items(), key=lambda item: item[1], reverse=True)}

string_count = 0
non_blank_count = 0

for species in species_to_count_overview_sorted.keys():        
    assert species_to_count_overview_sorted[species] == species_to_count_report[species]    
    count = species_to_count_overview_sorted[species]
    if species not in ('0','1'):
        string_count += count
        if species != 'blank':
            non_blank_count += count
            
    print('{}{}'.format(species.ljust(25),count))

n_images = len(all_files)
n_sequences = sum(species_to_count_overview_sorted.values())
print('There are a total of {} images in {} sequences ({:.2f} images per sequence)'.format(
    n_images,n_sequences,n_images/n_sequences))

print('\nString count: {}'.format(string_count))
print('Non-blank count: {}'.format(non_blank_count))

#%% Make sure that capture IDs in the reports/inventory files match

# ...and that all the images in the inventory tables are actually present on disk.

all_relative_paths_in_inventory = set()
files_missing_on_disk = []

for project_code in tqdm(project_codes):
    
    report_files = project_code_to_report_files[project_code]
    
    for report_file in report_files:
        
        project_base = file_to_project_folder(report_file)                
        inventory_file = report_file.replace('.csv','_image_inventory.csv')    
        
        inventory_df = pd.read_csv(os.path.join(input_base,inventory_file))
        report_df = pd.read_csv(os.path.join(input_base,report_file))
        
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
                assert image_path_relative.replace('.JPG','.jpg') not in all_files_relative_set
                assert image_path_relative.replace('.jpg','.JPG') not in all_files_relative_set
                files_missing_on_disk.append(image_path_relative)
                
            assert image_path_relative not in all_relative_paths_in_inventory
            all_relative_paths_in_inventory.add(image_path_relative)
                        
        assert capture_ids_in_report == capture_ids_in_inventory
    
    # ...for each report on this project

# ...for each project    

print('\n{} missing files (of {})'.format(len(files_missing_on_disk),len(all_relative_paths_in_inventory)))

    
#%% For all the files we have on disk, see which are and aren't in the inventory files

# There aren't any capital-P .PNG files, but if I don't include that
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


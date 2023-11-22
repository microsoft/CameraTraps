#
# cct_to_wi.py
#
# Converts COCO Camera Traps .json files to the Wildlife Insights
# batch upload format
#
# Also see:
#
# https://github.com/ConservationInternational/Wildlife-Insights----Data-Migration
#
# https://data.naturalsciences.org/wildlife-insights/taxonomy/search
#

#%% Imports

import os
import json
import pandas as pd
from collections import defaultdict 


#%% Paths

# A COCO camera traps file with information about this dataset
input_file = r'c:\temp\camera_trap_images_no_people\bellevue_camera_traps.2020-12-26.json'
assert os.path.isfile(input_file)

# A .json dictionary mapping common names in this dataset to dictionaries with the 
# WI taxonomy fields: common_name, wi_taxon_id, class, orer, family, genus, species
taxonomy_file = r'c:\temp\camera_trap_images_no_people\belleve_camera_traps_to_wi.json'
assert os.path.isfile(taxonomy_file)

templates_dir = r'c:\temp\wi_batch_upload_templates'
assert os.path.isdir(templates_dir)

output_base = r'c:\temp\wi_output'
os.makedirs(output_base,exist_ok = True)

    
#%% Constants


projects_file_name = 'Template Wildlife Insights Batch Upload - Projectv1.0.csv'
deployments_file_name = 'Template Wildlife Insights Batch Upload - Deploymentv1.0.csv'
images_file_name = 'Template Wildlife Insights Batch Upload - Imagev1.0.csv'
cameras_file_name = 'Template Wildlife Insights Batch Upload - Camerav1.0.csv'

assert all([os.path.isfile(os.path.join(templates_dir,fn)) for fn in \
            [projects_file_name,deployments_file_name,images_file_name,cameras_file_name]])


#%% Project information

project_info = {}
project_info['project_name'] = 'Bellevue Camera Traps'
project_info['project_id'] = 'bct_001'
project_info['project_short_name'] = 'BCT'
project_info['project_objectives'] = 'none'
project_info['project_species'] = 'Multiple'
project_info['project_species_individual'] = ''
project_info['project_sensor_layout'] = 'Convenience'
project_info['project_sensor_layout_targeted_type'] = ''
project_info['project_bait_use'] = 'No'
project_info['project_bait_type'] = 'None'
project_info['project_stratification'] = 'No'
project_info['project_stratification_type'] = ''
project_info['project_sensor_method'] = 'Sensor Detection'
project_info['project_individual_animals'] = 'No'
project_info['project_admin'] = 'Dan Morris'
project_info['project_admin_email'] = 'cameratraps@lila.science'
project_info['country_code'] = 'USA'
project_info['embargo'] = str(0)
project_info['initiative_id'] = ''
project_info['metadata_license'] = 'CC0'
project_info['image_license'] = 'CC0'

project_info['project_blank_images'] = 'No'
project_info['project_sensor_cluster'] = 'No'

camera_info = {}
camera_info['project_id'] = project_info['project_id'] 
camera_info['camera_id'] = '0000'
camera_info['make'] = ''
camera_info['model'] = ''
camera_info['serial_number'] = ''
camera_info['year_purchased'] = ''

deployment_info = {}

deployment_info['project_id'] = project_info['project_id'] 
deployment_info['deployment_id'] = 'test_deployment'
deployment_info['subproject_name'] = 'test_subproject'
deployment_info['subproject_design'] = ''
deployment_info['placename'] = 'yard'
deployment_info['longitude'] = '47.6101'
deployment_info['latitude'] = '-122.2015'
deployment_info['start_date'] = '2016-01-01 00:00:00'
deployment_info['end_date'] = '2026-01-01 00:00:00'
deployment_info['event_name'] = ''
deployment_info['event_description'] = ''
deployment_info['event_type'] = ''
deployment_info['bait_type'] = ''
deployment_info['bait_description'] = ''
deployment_info['feature_type'] = 'None'
deployment_info['feature_type_methodology'] = ''
deployment_info['camera_id'] = camera_info['camera_id']
deployment_info['quiet_period'] = str(60)
deployment_info['camera_functioning'] = 'Camera Functioning'
deployment_info['sensor_height'] = 'Chest height'
deployment_info['height_other'] = ''
deployment_info['sensor_orientation'] = 'Parallel'
deployment_info['orientation_other'] = ''
deployment_info['recorded_by'] = 'Dan Morris'

image_info = {}
image_info['identified_by'] = 'Dan Morris'


#%% Read templates

def parse_fields(templates_dir,file_name):
    
    with open(os.path.join(templates_dir,file_name),'r') as f:
        lines = f.readlines()
        lines = [s.strip() for s in lines if len(s.strip().replace(',','')) > 0]
        assert len(lines) == 1, 'Error processing template {}'.format(file_name)
        fields = lines[0].split(',')
        print('Parsed {} columns from {}'.format(len(fields),file_name))
    return fields

projects_fields = parse_fields(templates_dir,projects_file_name)
deployments_fields = parse_fields(templates_dir,deployments_file_name)
images_fields = parse_fields(templates_dir,images_file_name)
cameras_fields = parse_fields(templates_dir,cameras_file_name)


#%% Compare dictionary to template lists

def compare_info_to_template(info,template_fields,name):
    
    for s in info.keys():
        assert s in template_fields,'Field {} not specified in {}_fields'.format(s,name)
    for s in template_fields:
        assert s in info.keys(),'Field {} not specified in {}_info'.format(s,name)


def write_table(file_name,info,template_fields):
    
    assert len(info) == len(template_fields)
    
    project_output_file = os.path.join(output_base,file_name)
    with open(project_output_file,'w') as f:
    
        # Write the header
        for i_field,s in enumerate(template_fields):
            f.write(s)
            if i_field != len(template_fields)-1:
                f.write(',')
        f.write('\n')
        
        # Write values
        for i_field,s in enumerate(template_fields):
            f.write(info[s])
            if i_field != len(template_fields)-1:
                f.write(',')
        f.write('\n')
    

#%% Project file

compare_info_to_template(project_info,projects_fields,'project')
write_table(projects_file_name,project_info,projects_fields)


#%% Camera file

compare_info_to_template(camera_info,cameras_fields,'camera')
write_table(cameras_file_name,camera_info,cameras_fields)


#%% Deployment file

compare_info_to_template(deployment_info,deployments_fields,'deployment')
write_table(deployments_file_name,deployment_info,deployments_fields)


#%% Images file

# Read .json file with image information
with open(input_file,'r') as f:
    input_data = json.load(f)

# Read taxonomy dictionary
with open(taxonomy_file,'r') as f:
    taxonomy_mapping = json.load(f)
    
url_base = taxonomy_mapping['url_base']
taxonomy_mapping = taxonomy_mapping['taxonomy']

# Populate output information
# df = pd.DataFrame(columns = images_fields)

category_id_to_name = {cat['id']:cat['name'] for cat in input_data['categories']}

image_id_to_annotations = defaultdict(list)

annotations = input_data['annotations']
                         
# annotation = annotations[0]
for annotation in annotations:
    image_id_to_annotations[annotation['image_id']].append(
        category_id_to_name[annotation['category_id']])

rows = []

# im = input_data['images'][0]
for im in input_data['images']:

    row = {}
    
    url = url_base + im['file_name'].replace('\\','/')
    row['project_id'] = project_info['project_id']
    row['deployment_id'] = deployment_info['deployment_id']
    row['image_id'] = im['id']
    row['location'] = url
    row['identified_by'] = image_info['identified_by']
    
    category_names = image_id_to_annotations[im['id']]
    assert len(category_names) == 1
    category_name = category_names[0]
    
    taxon_info = taxonomy_mapping[category_name]
    
    assert len(taxon_info.keys()) == 7
    
    for s in taxon_info.keys():
        row[s] = taxon_info[s]    
    
    # We don't have counts, but we can differentiate between zero and 1
    if category_name == 'empty':
        row['number_of_objects'] = 0
    else:
        row['number_of_objects'] = 1
        
    row['uncertainty'] = None
    row['timestamp'] = im['datetime']; assert isinstance(im['datetime'],str)
    row['highlighted'] = 0
    row['age'] = None
    row['sex'] = None
    row['animal_recognizable'] = 'No'
    row['individual_id'] = None
    row['individual_animal_notes'] = None
    row['markings'] = None
    
    assert len(row) == len(images_fields)
    rows.append(row)
    
df = pd.DataFrame(rows)

df.to_csv(os.path.join(output_base,images_file_name),index=False)

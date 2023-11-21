#
# prepare_classification_script.py
#
# Notebook-y script used to prepare a series of shell commands to run a classifier
# (other than MegaClassifier) on a MegaDetector result set.
#
# Differs from prepare_classification_script_mc.py only in the final class mapping step.
#

#%% Job options

import os

organization_name = 'idfg'
job_name = 'idfg-2022-01-27-EOE2021S_Group6'
input_filename = 'idfg-2022-01-27-EOE2021S_Group6_detections.filtered_rde_0.60_0.85_30_0.20.json'
image_base = '/datadrive/idfg/EOE2021S_Group6'
crop_path = os.path.join(os.path.expanduser('~/crops'),job_name + '_crops')
device_id = 1

working_dir_base = os.path.join(os.path.expanduser('~/postprocessing'),
                                                   organization_name,
                                                   job_name)

output_base = os.path.join(working_dir_base,'combined_api_outputs')

assert os.path.isdir(working_dir_base)
assert os.path.isdir(output_base)

output_file = os.path.join(working_dir_base,'run_idfgclassifier_' + job_name +  '.sh')

input_files = [
    os.path.join(
        os.path.expanduser('~/postprocessing'),
                           organization_name,
                           job_name,
                           'combined_api_outputs',
                           input_filename
        )
    ]

for fn in input_files:
    assert os.path.isfile(fn)
    

#%% Constants

include_cropping = False

classifier_base = os.path.expanduser('~/models/camera_traps/idfg_classifier/idfg_classifier_20200905_042558')
assert os.path.isdir(classifier_base)

checkpoint_path = os.path.join(classifier_base,'idfg_classifier_ckpt_14_compiled.pt')
assert os.path.isfile(checkpoint_path)

classifier_categories_path = os.path.join(classifier_base,'label_index.json')
assert os.path.isfile(classifier_categories_path)

classifier_output_suffix = '_idfg_classifier_output.csv.gz'
final_output_suffix = '_idfgclassifier.json'

threshold_str = '0.65'
n_threads_str = '50'
image_size_str = '300'
batch_size_str = '64'
num_workers_str = '8'
logdir = working_dir_base

classification_threshold_str = '0.05'

# This is just passed along to the metadata in the output file, it has no impact
# on how the classification scripts run.
typical_classification_threshold_str = '0.75'

classifier_name = 'idfg4'
        

#%% Set up environment

commands = []
# commands.append('cd CameraTraps/classification\n')
# commands.append('conda activate cameratraps-classifier\n')


#%% Crop images

if include_cropping:
    
    commands.append('\n### Cropping ###\n')
    
    # fn = input_files[0]
    for fn in input_files:
    
        input_file_path = fn
        crop_cmd = ''
        
        crop_comment = '\n# Cropping {}\n'.format(fn)
        crop_cmd += crop_comment
        
        crop_cmd += "python crop_detections.py \\\n" + \
        	 input_file_path + ' \\\n' + \
             crop_path + ' \\\n' + \
             '--images-dir "' + image_base + '"' + ' \\\n' + \
             '--threshold "' + threshold_str + '"' + ' \\\n' + \
             '--square-crops ' + ' \\\n' + \
             '--threads "' + n_threads_str + '"' + ' \\\n' + \
             '--logdir "' + logdir + '"' + ' \\\n' + \
             '\n'
        crop_cmd = '{}'.format(crop_cmd)
        commands.append(crop_cmd)


#%% Run classifier

commands.append('\n### Classifying ###\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    
    classify_cmd = ''
    
    classify_comment = '\n# Classifying {}\n'.format(fn)
    classify_cmd += classify_comment
    
    classify_cmd += "python run_classifier.py \\\n" + \
    	 checkpoint_path + ' \\\n' + \
         crop_path + ' \\\n' + \
         classifier_output_path + ' \\\n' + \
         '--detections-json "' + input_file_path + '"' + ' \\\n' + \
         '--classifier-categories "' + classifier_categories_path + '"' + ' \\\n' + \
         '--image-size "' + image_size_str + '"' + ' \\\n' + \
         '--batch-size "' + batch_size_str + '"' + ' \\\n' + \
         '--num-workers "' + num_workers_str + '"' + ' \\\n'
    
    if device_id is not None:
        classify_cmd += '--device {}'.format(device_id)
        
    classify_cmd += '\n\n'    
    classify_cmd = '{}'.format(classify_cmd)
    commands.append(classify_cmd)
		

#%% Merge classification and detection outputs

commands.append('\n### Merging ###\n')

# fn = input_files[0]
for fn in input_files:

    input_file_path = fn
    classifier_output_path = crop_path + classifier_output_suffix
    final_output_path = os.path.join(output_base,
                                     os.path.basename(classifier_output_path)).\
                                     replace(classifier_output_suffix,
                                     final_output_suffix)
    final_output_path = final_output_path.replace('_detections','')
    final_output_path = final_output_path.replace('_crops','')
    
    merge_cmd = ''
    
    merge_comment = '\n# Merging {}\n'.format(fn)
    merge_cmd += merge_comment
    
    merge_cmd += "python merge_classification_detection_output.py \\\n" + \
    	 classifier_output_path + ' \\\n' + \
         classifier_categories_path + ' \\\n' + \
         '--output-json "' + final_output_path + '"' + ' \\\n' + \
         '--detection-json "' + input_file_path + '"' + ' \\\n' + \
         '--classifier-name "' + classifier_name + '"' + ' \\\n' + \
         '--threshold "' + classification_threshold_str + '"' + ' \\\n' + \
         '--typical-confidence-threshold "' + typical_classification_threshold_str + '"' + ' \\\n' + \
         '\n'
    merge_cmd = '{}'.format(merge_cmd)
    commands.append(merge_cmd)


#%% Write everything out

with open(output_file,'w') as f:
    for s in commands:
        f.write('{}'.format(s))

import stat
st = os.stat(output_file)
os.chmod(output_file, st.st_mode | stat.S_IEXEC)
        
#%% Imports and constants

import json
import os

from tqdm import tqdm
from operator import itemgetter
from shutil import copyfile

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
from data_management.cct_json_utils import IndexedJsonDb

annotation_list_filename = r'd:\wildlife_data\zsl_borneo\all_img_ids_to_bbox.json'
image_json = r'd:\wildlife_data\zsl_borneo\201906cameratraps\0.5\zsl_camera_traps_201906.json'
image_base = r'd:\wildlife_data\zsl_borneo\201906cameratraps\0.5'
output_base = r'd:\wildlife_data\zsl_borneo'

human_classes = ['human','hunter']


#%% Load data

with open(annotation_list_filename,'r') as f:
    annotation_list = json.load(f)
    
# with open(image_json,'r') as f:
#    data = json.load(f)
indexedData = IndexedJsonDb(image_json)

print('Done loading data')    
    

#%% Sanity-check data

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = image_base
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = False
    
sortedCategories = sanity_check_json_db.sanity_check_json_db(indexedData.db,options)


#%% Label previews

viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 500
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
html_output_file,image_db = visualize_db.process_images(indexedData.db,
                                                        os.path.join(output_base,'preview'),
                                                        image_base,viz_options)
os.startfile(html_output_file)


#%% Collect images to annotate

images_to_annotate = []

annotation_list = set(annotation_list)
n_humans = 0

for im in tqdm(indexedData.db['images']):
    
    class_names = indexedData.get_classes_for_image(im)
    b_human = False
    for cn in class_names:
        if cn.lower() in human_classes:
            b_human = True
            n_humans += 1
            break
        
    if b_human or im['id'] in annotation_list:
        images_to_annotate.append(im)
    

print('Found {} of {} images ({} humans)'.format(len(images_to_annotate),len(annotation_list),n_humans))
assert len(images_to_annotate) >= len(annotation_list)



#%% Sort by sequence and frame

images_to_annotate = sorted(images_to_annotate, key=itemgetter('seq_id', 'frame_num')) 


#%% Copy to a folder by GUID

# dataset[dataset_id].seq[sequence_id].frame[frame_number].img[img_id].extension

imerit_output_base = os.path.join(output_base,'imerit_batch_9')
os.makedirs(imerit_output_base,exist_ok=True)

# im = images_to_annotate[0]
for im in tqdm(images_to_annotate):
    
    relative_path = im['file_name']
    extension = os.path.splitext(relative_path)[1]
    frame_num = im['frame_num']
    seq_id = im['seq_id']
    id = im['id']
    assert '.' not in id
    input_full_path = os.path.join(image_base,relative_path)
    assert os.path.isfile(input_full_path)
    output_filename = 'datasetzslborneo.seq' + '{0:0>8d}'.format(seq_id) + '.frame' + \
        '{0:0>4d}'.format(frame_num) + '.img' + id + extension
    im['imerit_filename'] = output_filename
    output_full_path = os.path.join(imerit_output_base,output_filename)
    assert not os.path.isfile(output_full_path)
    copyfile(input_full_path,output_full_path)

# ...for each image
    

#%% Write out the annotation list

imerit_batch9_json_filename = os.path.join(imerit_output_base,'imerit_batch_9.json')
with open(imerit_batch9_json_filename,'w') as f:
    json.dump(images_to_annotate, f, indent=2)
    
    
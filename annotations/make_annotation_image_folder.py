import json
import os
from shutil import copyfile
import csv
import os


# Filename convention
# dataset[dataset_id].seq[sequence_id].frame[frame_number].img[img_id].extension
file_folder = 'D:/snapshot_serengeti/'
new_file_folder = 'D:/imerit_annotation_images_ss_1/images/'
if not os.path.exists(new_file_folder):
    os.makedirs(new_file_folder)
db_file = 'C:/Users/t-sabeer/Documents/databases/imerit_annotation_images_ss_1.json'
with open(db_file,'r') as f:
    data = json.load(f)

print(len(data['images']))
not_found = []
for im in data['images']:
    old_filename = im['file_name'].split('/')
    old_image_name = old_filename[-1].split('_')[-1]
    filename = ''
    for chunk in old_filename[:-1]:
        filename += chunk + '/'
    filename += old_image_name
    if os.path.isfile(file_folder+filename):
        new_filename = im['seq_id']+'_'+str(im['frame_num'])+'_'+im['file_name'].split('/')[-1]
        #copyfile(file_folder+filename,new_file_folder+new_filename)
        im['file_name'] = new_filename
    else: 
        not_found.append(im['id'])
print(len(not_found))

data['images'] = [im for im in data['images'] if im['id'] not in not_found]
data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] not in not_found]

json.dump(data, open('C:/Users/t-sabeer/Documents/databases/imerit_annotation_images_ss_1_new_filenames.json','w'))

print(data['images'][0])

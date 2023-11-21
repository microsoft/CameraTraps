#
# create_classification_tfrecords_from_json.py
#
# Called from make_tfrecords_cis_trans.py if you're running a classification experiment.
#

import json
import codecs
import pickle
from PIL import Image

#datafolder = '/teamscratch/findinganimals/data/iWildCam2018/'
#datafile = 'eccv_18_annotation_files_oneclass/CaltechCameraTrapsECCV18'
#image_file_root = datafolder+'eccv_18_all_images/'

#datafolder = '/teamscratch/findinganimals/data/iWildCam2018/'
#datafolder = '/data/iwildcam/'
datafolder = '/datadrive/snapshotserengeti/'
#datafile = 'combined_iwildcam_annotations_oneclass/eccv_train_and_imerit_2'
database_file = datafolder+'databases/oneclass/imerit_ss_annotations_1.json'
image_file_root = datafolder+'images/'

def create_classification_tfrecords_format(database_file,image_file_root):
    with open(database_file,'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    print('Images: ', len(images))
    print('Annotations: ', len(annotations))
    print('Categories: ', len(categories))
    print(categories)
    
    vis_data = []

    im_id_to_im = {im['id']:im for im in images}
    #need consecutive category ids
    #old_cat_id_to_new_cat_id = {categories[idx]['id']:idx+1 for idx in range(len(categories))}
    #print(old_cat_id_to_new_cat_id)
    cat_id_to_cat_name = {cat['id']:cat['name'] for cat in categories}
    im_id_to_anns = {im['id']:[] for im in images}
    for ann in annotations: 
        im_id_to_anns[ann['image_id']].append(ann)
   
    for im in images:
        #remove multiclass images
        if len(im_id_to_anns[im['id']]) > 1:
            continue
        image_data = {}
        image_data['filename'] = image_file_root+im['file_name']
        
        image_data['id'] = im['id']
        image_data['seq_id'] = im['seq_id']
        image_data['seq_num_frames'] = im['seq_num_frames']
        image_data['frame_num'] = im['frame_num']
        image_data['location'] = im['location']
        if 'date_captured' in image_data:
            image_data['date_captured'] = im['date_captured']
        image_data['class'] = {}
        
        for ann in im_id_to_anns[im['id']]:
            image_data['class']['label'] = ann['category_id']
            image_data['class']['text'] = cat_id_to_cat_name[ann['category_id']]
        vis_data.append(image_data)

    print('New images: ', len(vis_data))
    #print(images[0])
    #print(vis_data[0])

            
    return vis_data

if __name__ == '__main__':

    vis_data = create_tfrecords_format(database_file,image_file_root)
    output_file = database_file.split('.')[0] + '_tfrecord_format.json'
    with open(output_file,'w') as f:
        json.dump(vis_data, f, ensure_ascii=False)


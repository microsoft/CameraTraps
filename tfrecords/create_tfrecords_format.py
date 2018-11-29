#
# create_tfrecords_format.py
#
# This script converts a COCO formatted json database to another json file that would be the
# input to create_tfrecords.py or make_tfrecords_with_train_test_split.py or similar scripts
# to create tf_records
#

#%% Imports and environment

import json
import os
from PIL import Image
from tqdm import tqdm


#%% Main tfrecord generation function

def create_tfrecords_format(database_file, image_file_root, cats_to_ignore = [],
                            exclude_images_without_bbox = False):
    
    print('Loading database...')
    with open(database_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    print('Images: ', len(images))
    print('Annotations: ', len(annotations))
    print('Categories: ', len(categories))

    no_bbox_count = 0
    vis_data = []

    # Remap category IDs; TF needs consecutive category ids
    valid_cats = [cat for cat in categories if cat['name'] not in cats_to_ignore]
    old_cat_id_to_new_cat_id = {valid_cats[idx]['id']:idx+1 for idx in range(len(valid_cats))}
    cat_id_to_cat_name = {cat['id']: cat['name'] for cat in categories}

    im_id_to_anns = {im['id']: [] for im in images}

    # Sanity-check number of empty annotations and annotation-less images
    for ann in annotations:
        if 'bbox' in ann:
            assert(len(ann['bbox']) > 0)
            im_id_to_anns[ann['image_id']].append(ann)            
        else:
            no_bbox_count += 1
            
    print('Annotations with no bbox: ', no_bbox_count)
    
    emptyImages = []
    for imageID in im_id_to_anns:
        imAnns = im_id_to_anns[imageID]
        if (len(imAnns) == 0):
            emptyImages.append(imageID)
            
    print('Images with no annotations: ', len(emptyImages))
    
    num_img_with_anno = 0
    num_bboxes_skipped = 0
    
    for im in tqdm(images):
        
        if exclude_images_without_bbox:  
            
            # Images without annotations don't have bounding boxes
            if len(im_id_to_anns[im['id']]) < 1:
                continue
            else:
                # Images with annotations *may* have bounding boxes
                bHasBbox = False
                for imAnn in im_id_to_anns:
                    if 'bbox' in imAnn:
                        bHasBbox = True
                    break
                if not bHasBbox:
                    continue            
            
        num_img_with_anno += 1

        image_data = {}
        image_data['filename'] = os.path.join(image_file_root, im['file_name'])
        image_data['id'] = im['id']

        # Propagate optional metadata to tfrecords
        if 'seq_id' in im:
            image_data['seq_id'] = im['seq_id']
            image_data['seq_num_frames'] = im['seq_num_frames']
            image_data['frame_num'] = im['frame_num']
        if 'location' in im:
            image_data['location'] = im['location']
        if 'date_captured' in image_data:
            image_data['date_captured'] = im['date_captured']
        if 'datetime' in image_data:
            image_data['datetime'] = im['datetime']
        if 'label' in im:
            image_data['text'] = im['label']

        if 'height' in im:
            im_h = im['height']
            im_w = im['width']
        else:
            im_w, im_h = Image.open(image_data['filename']).size

        image_data['height'] = im_h
        image_data['width'] = im_w

        image_data['object'] = {}
        image_data['object']['count'] = 0
        image_data['object']['id'] = []
        image_data['object']['bbox'] = {}
        image_data['object']['bbox']['xmin'] = []
        image_data['object']['bbox']['xmax'] = []
        image_data['object']['bbox']['ymin'] = []
        image_data['object']['bbox']['ymax'] = []
        image_data['object']['bbox']['label'] = []
        image_data['object']['bbox']['text'] = []

        for ann in im_id_to_anns[im['id']]:
            
            if len(ann['bbox']) == 0:
                continue

            # Do not include bboxes of categories in the cat_to_ignore list
            if ann['category_id'] not in old_cat_id_to_new_cat_id:
                num_bboxes_skipped += 1
                continue

            image_data['object']['count'] += 1
            image_data['object']['id'].append(ann['id'])
            image_data['object']['bbox']['label'].append(
                old_cat_id_to_new_cat_id[ann['category_id']])
            image_data['object']['bbox']['text'].append(
                cat_id_to_cat_name[ann['category_id']])

            xmin = ann['bbox'][0] / float(im_w)
            xmax = (ann['bbox'][0] + ann['bbox'][2]) / float(im_w)
            ymin = ann['bbox'][1] / float(im_h)
            ymax = (ann['bbox'][1] + ann['bbox'][3]) / float(im_h)

            image_data['object']['bbox']['xmin'].append(xmin)
            image_data['object']['bbox']['xmax'].append(xmax)
            image_data['object']['bbox']['ymin'].append(ymin)
            image_data['object']['bbox']['ymax'].append(ymax)

        # ...for each annotation for the current image
        
        vis_data.append(image_data)

    # ...for each image
    
    if exclude_images_without_bbox:
        print('Number of images with annotations: ', num_img_with_anno)
        
    print('Number of image entries: ', len(vis_data))
    print('num_bboxes_skipped: ', num_bboxes_skipped)
    
    return vis_data


#%% Driver
    
if __name__ == '__main__':
    
    cats_to_ignore = ['empty'] # ['person', 'empty', 'car']
    
    # Should we exclude images without bbox annotations in the resulting json?
    #
    # For eMammal, since all images whether sent for annotation or not are included in the DB,
    # this flag needs to be True. Images that have been determined to be empty will be dealt
    # with in the for loop over annotations    
    exclude_images_without_bbox = True  
        
    image_file_root = '/datadrive/emammal/'
    database_file = '/home/yasiyu/yasiyu_temp/eMammal_db/eMammal_20180929.json'
    
    vis_data = create_tfrecords_format(database_file, image_file_root)
    output_file = database_file.split('.')[0] + '_tfrecord_format.json'
    print('Saving the tfrecord_format json...')
    with open(output_file, 'w') as f:
        json.dump(vis_data, f, ensure_ascii=False, indent=4, sort_keys=True)


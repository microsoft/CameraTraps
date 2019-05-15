#
# create_tfrecords_format.py
#
# This script converts a COCO formatted json database to another json file that would be the
# input to create_tfrecords.py or similar scripts to create tf_records
#

#%% Imports and environment

import json
import os
from PIL import Image
from tqdm import tqdm


#%% Main tfrecord generation function

def create_tfrecords_format(dataset_name, database_file, image_file_root, cats_to_include=[],
                            exclude_images_without_bbox=False):
    
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
    valid_cats = [cat for cat in categories if cat['name'] in cats_to_include]
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
            
    print('Annotations with no bbox of any category: ', no_bbox_count)
    
    empty_images = []
    for image_id in im_id_to_anns:
        im_anns = im_id_to_anns[image_id]
        if (len(im_anns) == 0):
            empty_images.append(image_id)
            
    print('Images with no annotations: ', len(empty_images))
    
    num_img_with_anno = 0
    num_bboxes_skipped = 0
    num_images_skipped_for_group = 0  # we exclude images with any 'group' annotations to avoid confusion
    
    for im in tqdm(images):
        
        if exclude_images_without_bbox:
            # Images without annotations don't have bounding boxes
            if len(im_id_to_anns[im['id']]) < 1:
                continue
            else:
                # Images with annotations *may* have bounding boxes
                has_bbox = False
                for ann in im_id_to_anns[im['id']]:
                    if 'bbox' in ann:
                        has_bbox = True
                        break
                if not has_bbox:
                    continue            
            
        num_img_with_anno += 1

        image_data = {}

        db_file_name = im['file_name']

        image_data['filename'] = os.path.join(image_file_root, db_file_name)  # .replace('/', '~')

        # prepend the dataset name to image_id because after inference on val set, records from
        # different datasets are stored in one tfrecord
        image_data['id'] = dataset_name + '~' + im['id']

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

        image_contains_group = False

        for ann in im_id_to_anns[im['id']]:
            
            if len(ann['bbox']) == 0:
                continue

            # checking to ignore any images that contain 'group' needs to happen before ignoring non-valid categories!
            if cat_id_to_cat_name[ann['category_id']] == 'group':
                image_contains_group = True
                continue

            # Only include valid categories
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

        if image_contains_group:
            num_images_skipped_for_group += 1
            continue

        vis_data.append(image_data)

    # ...for each image
    
    if exclude_images_without_bbox:
        print('Number of images with annotations: ', num_img_with_anno)
        
    print('Number of image entries: ', len(vis_data))
    print('num_bboxes_skipped: ', num_bboxes_skipped)
    print('num_images_skipped_for_group: ', num_images_skipped_for_group)
    
    return vis_data

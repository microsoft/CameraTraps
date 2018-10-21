import json
import os

from PIL import Image
from tqdm import tqdm

# create_tfrecords_format.py
#
# This script converts a COCO formatted json database to another json file that would be the
# input to create_tfrecords.py or make_tfrecords_with_train_test_split.py or similar scripts
# to create tf_records
#

# configurations and paths

# eMammal
coordinates_relative = True  # are the coordinates in bbox specification relative
cat_to_ignore = ['empty'] #['person', 'empty']
# iMerit label for human is 'person'; 'empty' is not valid either as a bbox category,
# but we actually rely on the image level label 'empty' to determine if an image is empty

exclude_images_without_bbox = True  # to exclude images without bbox annotations in the resulting json?
# for eMammal, since all images whether sent for annotation or not are included in the DB,
# this flag needs to be True. Images that have been determined to be empty will be dealt
# with in the for loop over annotations

image_file_root = '/datadrive/emammal/'
database_file = '/home/yasiyu/yasiyu_temp/eMammal_db/eMammal_20180929.json'


# Snapshot Serengeti and iWildCam
# coordinate_relative = False
# cat_to_exclude = ['empty', 'car']
# exclude_images_without_bbox = False
# is_one_class = False

# datafolder = '/teamscratch/findinganimals/data/iWildCam2018/'
# datafile = 'eccv_18_annotation_files_oneclass/CaltechCameraTrapsECCV18'
# image_file_root = datafolder+'eccv_18_all_images/'

# datafolder = '/teamscratch/findinganimals/data/iWildCam2018/'
# datafolder = '/data/iwildcam/'
# datafolder = '/datadrive/snapshotserengeti/'
# datafile = 'combined_iwildcam_annotations_oneclass/eccv_train_and_imerit_2'
# database_file = datafolder+'databases/oneclass/imerit_ss_annotations_1.json'
# image_file_root = datafolder+'images/'
# cat_to_ignore = ['empty', 'car']


def create_tfrecords_format(database_file, image_file_root):
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
    empty_images_count = 0
    vis_data = []

    # need consecutive category ids
    valid_cats = [cat for cat in categories if cat['name'] not in cats_to_ignore]
    print(valid_cats)
    old_cat_id_to_new_cat_id = {valid_cats[idx]['id']:idx+1 for idx in range(len(valid_cats))}
    print(old_cat_id_to_new_cat_id)
    cat_id_to_cat_name = {cat['id']: cat['name'] for cat in categories}

    im_id_to_anns = {im['id']: [] for im in images}

    for ann in annotations:
        if 'bbox' in ann:  # and cat_id_to_cat_name[ann['category_id']] != 'car':
            im_id_to_anns[ann['image_id']].append(ann)
            if len(ann['bbox']) == 0:
                empty_images_count += 1
        else:
            no_bbox_count += 1
    print('Anns with no bbox: ', no_bbox_count)
    print('eMammal - number of empty images from the labeled set: ', empty_images_count)

    num_img_with_anno = 0
    num_bboxes_skipped = 0
    for im in tqdm(images):
        if exclude_images_without_bbox:
            if len(im_id_to_anns[im['id']]) < 1:
                # but do include all images that are labeled "empty" (whether annotated or not)
                if im['label'] != 'empty':
                    continue
            else:
                num_img_with_anno += 1

        image_data = {}
        image_data['filename'] = os.path.join(image_file_root, im['file_name'])
        # image_data['filename'] = image_data['filename'].encode('utf-8')
        image_data['id'] = im['id']

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
        if 'height' in im:
            im_h = im['height']
            im_w = im['width']
        else:
            im_w, im_h = Image.open(image_data['filename']).size

        image_data['height'] = im_h
        image_data['width'] = im_w

        # no finalized numerical label on the image for eMammal
        if 'label' in im:
            image_data['text'] = im['label']

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

            # do not include bboxes of categories in the cat_to_ignore list
            if ann['category_id'] not in old_cat_id_to_new_cat_id:
                num_bboxes_skipped += 1
                continue

            image_data['object']['count'] += 1
            image_data['object']['id'].append(ann['id'])
            image_data['object']['bbox']['label'].append(
                old_cat_id_to_new_cat_id[ann['category_id']])
            image_data['object']['bbox']['text'].append(
                cat_id_to_cat_name[ann['category_id']])

            if coordinates_relative:
                xmin = ann['bbox'][0]
                xmax = ann['bbox'][0] + ann['bbox'][2]
                ymin = ann['bbox'][1]
                ymax = ann['bbox'][1] + ann['bbox'][3]
            else:
                xmin = ann['bbox'][0] / float(im_w)
                xmax = (ann['bbox'][0] + ann['bbox'][2]) / float(im_w)
                ymin = ann['bbox'][1] / float(im_h)
                ymax = (ann['bbox'][1] + ann['bbox'][3]) / float(im_h)

            image_data['object']['bbox']['xmin'].append(xmin)
            image_data['object']['bbox']['xmax'].append(xmax)
            image_data['object']['bbox']['ymin'].append(ymin)
            image_data['object']['bbox']['ymax'].append(ymax)

        vis_data.append(image_data)

    if exclude_images_without_bbox:
        print('Number of images with annotations: ', num_img_with_anno)
    print('Number of image entries: ', len(vis_data))
    print('num_bboxes_skipped: ', num_bboxes_skipped)
    # print(images[0])
    # print(vis_data[0])

    return vis_data


if __name__ == '__main__':
    vis_data = create_tfrecords_format(database_file, image_file_root)
    output_file = database_file.split('.')[0] + '_tfrecord_format.json'
    print('Saving the tfrecord_format json...')
    with open(output_file, 'w') as f:
        json.dump(vis_data, f, ensure_ascii=False, indent=4, sort_keys=True)


#
# make_embedded_db.py
#
# Given an image json and/or a bounding box json in the COCO Camera Trap format, output
# an embedded/denormalized version of it.
#
# Check carefully that the dataset_name parameter is set correctly!!
#

import argparse
import json
import os

from data_management.cct_json_utils import IndexedJsonDb
from ct_utils import truncate_float


def write_json(p, content, indent=1):
    with open(p, 'w') as f:
        json.dump(content, f, indent=indent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str,
                        help='a short string representing the dataset to be used as a partition key in the DB')
    parser.add_argument('--image_db', type=str, help='path to the json containing the image DB in CCT format')
    parser.add_argument('--bbox_db', type=str, help='path to the json containing the bbox DB in CCT format')
    parser.add_argument('--embedded_db', type=str, required=True, help='path to store the resulting json')
    args = parser.parse_args()

    assert len(args.dataset_name) > 0, 'dataset name cannot be an empty string'
    assert os.path.exists(args.image_db), 'image_db file path provided does not point to a file'
    assert os.path.exists(args.bbox_db), 'bbox_db file path provided does not point to a file'

    #%% integrate the image DB

    # at first a dict of image_id: image_obj with annotations embedded,
    # then its items becomes the array of documents that will get uploaded to Cosmos DB
    docs = {}

    if args.image_db:
        cct_json_db = IndexedJsonDb(args.image_db)
        docs = cct_json_db.image_id_to_image

        # takes in image entries and species annotation in the image DB
        num_images_with_more_than_1_species = 0
        for image_id, annotations in cct_json_db.image_id_to_annotations.items():
            docs[image_id]['annotations'] = {
                'species': []
            }
            if len(annotations) > 1:
                num_images_with_more_than_1_species += 1
            for anno in annotations:
                cat_name = cct_json_db.cat_id_to_name[anno['category_id']]
                docs[image_id]['annotations']['species'].append(cat_name)

        print('Number of items from the image DB:', len(docs))
        print('Number of images with more than 1 species: {} ({}% of image DB)'.format(
            num_images_with_more_than_1_species, round(100 * num_images_with_more_than_1_species / len(docs), 2)))

    #%% integrate the bbox DB
    if args.bbox_db:
        cct_bbox_json_db = IndexedJsonDb(args.bbox_db)

        # add any images that are not in the image DB
        # also add any fields in the image object that are not present already
        num_added = 0
        num_amended = 0
        for image_id, image_obj in cct_bbox_json_db.image_id_to_image.items():
            if image_id not in docs:
                docs[image_id] = image_obj
                num_added += 1

            amended = False
            for field_name, val in image_obj.items():
                if field_name not in docs[image_id]:
                    docs[image_id][field_name] = val
                    amended = True
            if amended:
                num_amended += 1

        print('Number of images added from bbox DB entries: ', num_added)
        print('Number of images amended: ', num_amended)
        print('Number of items in total: ', len(docs))

        # add bbox to the annotations field
        num_more_than_1_bbox = 0

        for image_id, bbox_annotations in cct_bbox_json_db.image_id_to_annotations.items():

            # for any newly added images
            if 'annotations' not in docs[image_id]:
                docs[image_id]['annotations'] = {}

            docs[image_id]['annotations']['bbox'] = []

            if len(bbox_annotations) > 1:
                num_more_than_1_bbox += 1

            for bbox_anno in bbox_annotations:
                item_bbox = {
                    'category': cct_bbox_json_db.cat_id_to_name[bbox_anno['category_id']],
                    'bbox_abs': bbox_anno['bbox'],
                }

                if 'width' in docs[image_id]:
                    image_w = docs[image_id]['width']
                    image_h = docs[image_id]['height']
                    x, y, w, h = bbox_anno['bbox']
                    item_bbox['bbox_rel'] = [
                        truncate_float(x / image_w),
                        truncate_float(y / image_h),
                        truncate_float(w / image_w),
                        truncate_float(h / image_h)
                    ]

                docs[image_id]['annotations']['bbox'].append(item_bbox)

        print('Number of images with more than one bounding box: {} ({}% of all entries)'.format(
            num_more_than_1_bbox, 100 * num_more_than_1_bbox / len(docs), 2))
    else:
        print('No bbox DB provided.')

    assert len(docs) > 0, 'No image entries found in the image or bbox DB jsons provided.'
    
    docs = list(docs.values())

    #%% processing
    # get rid of any trailing '.JPG' for the id field
    # insert the 'dataset' attribute used as the partition key
    # replace illegal chars (for Cosmos DB) in the id field of the image
    # replace directory separator with tilde ~
    # rename the id field (reserved word) to image_id
    illegal_char_map = {
        '/': '~',
        '\\': '~',
        '?': '__qm__',
        '#': '__pound__'
    }

    for i in docs:
        i['id'] = i['id'].split('.JPG')[0].split('.jpg')[0]

        for illegal, replacement in illegal_char_map.items():
            i['id'] = i['id'].replace(illegal, replacement)

        i['dataset'] = args.dataset_name

        i['image_id'] = i['id']
        del i['id']

    #%% some validation
    print('Example item:')
    print(docs[-1])

    num_both_species_bbox = 0
    for item in docs:
        if 'annotations' in item:
            if 'species' in item['annotations'] and 'bbox' in item['annotations']:
                num_both_species_bbox += 1
    print('Number of images with both species and bbox annotations: {} ({}% of all entries)'.format(
        num_both_species_bbox, round(100 * num_both_species_bbox / len(docs), 2)))

    #%% save the embedded json database
    write_json(args.embedded_db, docs)


if __name__ == '__main__':
    main()


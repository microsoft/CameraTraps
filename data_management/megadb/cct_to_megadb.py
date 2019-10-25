#
# cct_to_megadb.py
#
# Given an image json and/or a bounding box json in the COCO Camera Trap format, output
# the equivalent database in the MegaDB format so that the entries can be ingested into MegaDB.
#
# All fields in the original image json would be carried over, and any fields in the
# bounding box json but not in the corresponding entry in the image json will be added.
#
# Check carefully that the dataset_name parameter is set correctly!!

import argparse
import json
import os
import uuid
from collections import defaultdict
import sys

from data_management.megadb import sequences_schema_check
from data_management.cct_json_utils import IndexedJsonDb
from ct_utils import truncate_float


def write_json(p, content, indent=1):
    with open(p, 'w') as f:
        json.dump(content, f, indent=indent)


def process_sequences(docs, dataset_name):
    print('Putting into sequences...')
    img_level_properties = set()
    sequences = defaultdict(list)

    # a dummy sequence ID will be generated if the image entry does not have a seq_id field
    # seq_id only needs to be unique within this dataset; MegaDB does not rely on it as the _id field

    # "annotations" fields will be opened and have its sub-field surfaced one level up
    for im in docs:
        if 'seq_id' in im:
            seq_id = im['seq_id']
            del im['seq_id']
        else:
            seq_id = 'dummy_' + uuid.uuid4().hex  # if this will be sent for annotation, may need a sequence ID based on file name to group potential sequences together
            img_level_properties.add('file_name')
            img_level_properties.add('image_id')

        for obsolete_prop in ['seq_num_frames', 'width', 'height']:
            if obsolete_prop in im:
                del im[obsolete_prop]

        if 'annotations' in im:
            for prop, prop_val in im['annotations'].items():
                im[prop] = prop_val
                if prop == 'bbox':
                    for bbox_item in im['bbox']:
                        if 'bbox_rel' not in bbox_item:
                            print('Missing relative coordinates for bbox! Exiting...')
                            print(im)
                            sys.exit(1)
                        else:
                            bbox_item['bbox'] = bbox_item['bbox_rel']
                            del bbox_item['bbox_rel']

                            if 'bbox_abs' in bbox_item:
                                del bbox_item['bbox_abs']

                if prop == 'species':
                    im['class'] = im['species']
                    del im['species']

            del im['annotations']

        sequences[seq_id].append(im)

    new_sequences = []
    for seq_id, images in sequences.items():
        new_sequences.append({
            'seq_id': seq_id,
            'dataset': dataset_name,
            'images': images
        })

    sequences = new_sequences
    print(len(sequences))
    print(sequences[100])

    # check that the location field is the same for all images in a sequence
    print('Checking the location field...')
    for seq in sequences:
        locations = []
        for im in seq['images']:
            locations.append(im.get('location', ''))  # empty string if no location provided
        assert len(set(locations)) == 1, 'Location fields in images of the sequence {} are different.'.format(seq['seq_id'])

    # check which fields in a CCT image entry are sequence-level
    all_img_properties = set()
    for seq in sequences:
        if 'images' not in seq:
            continue

        image_properties = defaultdict(set)
        for im in seq['images']:
            for prop_name, prop_value in im.items():
                image_properties[prop_name].add(str(prop_value))  # make all hashable
                all_img_properties.add(prop_name)

        for prop_name, prop_values in image_properties.items():
            if len(prop_values) > 1:
                img_level_properties.add(prop_name)

    # image-level properties that really should be sequence-level
    seq_level_properties = all_img_properties - img_level_properties
    print('all_img_properties')
    print(all_img_properties)
    print('img_level_properties')
    print(img_level_properties)
    print('image-level properties that really should be sequence-level')
    print(seq_level_properties)

    for seq in sequences:
        if 'images' not in seq:
            continue

        for seq_property in seq_level_properties:
            # get the value of this sequence-level property from the first image entry
            seq[seq_property] = seq['images'][0][seq_property]
            for im in seq['images']:
                del im[seq_property]  # and remove it from the image level

    # check which fields are really dataset-level and should be included in the dataset table instead.
    seq_level_prop_values = defaultdict(set)

    for seq in sequences:
        for prop_name in seq:
            if prop_name not in ['dataset', 'seq_id', 'class', 'images', 'location']:
                seq_level_prop_values[prop_name].add(seq[prop_name])

    dataset_props = []
    for prop_name, values in seq_level_prop_values.items():
        if len(values) == 1:
            dataset_props.append(prop_name)
            print('   Sequence-level property {} with value {} should be a dataset-level property. Removed from sequences.'.format(prop_name, list(values)[0]))

    sequences_neat = []
    for seq in sequences:
        for dataset_prop in dataset_props:
            del seq[dataset_prop]
        sequences_neat.append(sequences_schema_check.order_seq_properties(seq))

    print('Finished processing sequences.')
    return sequences_neat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str,
                        help='a short string representing the dataset to be used as a partition key in MegaDB')
    parser.add_argument('--image_db', type=str, help='path to the json containing the image DB in CCT format')
    parser.add_argument('--bbox_db', type=str, help='path to the json containing the bbox DB in CCT format')
    parser.add_argument('--docs', type=str, help='embedded CCT format json to use instead of image_db or bbox_db')
    parser.add_argument('--partial_mega_db', type=str, required=True, help='path to store the resulting json')
    args = parser.parse_args()

    assert len(args.dataset_name) > 0, 'dataset name cannot be an empty string'
    print('The dataset_name is set to {}. Please make sure this is correct!'.format(args.dataset_name))

    if args.image_db:
        assert os.path.exists(args.image_db), 'image_db file path provided does not point to a file'
    if args.bbox_db:
        assert os.path.exists(args.bbox_db), 'bbox_db file path provided does not point to a file'

    # at first a dict of image_id: image_obj with annotations embedded,
    docs = {}

    # %% integrate the image DB
    if args.image_db:
        print('Loading image DB...')
        cct_json_db = IndexedJsonDb(args.image_db)
        docs = cct_json_db.image_id_to_image  # each image entry is first assigned the image object

        # takes in image entries and species and other annotations in the image DB
        num_images_with_more_than_1_species = 0
        for image_id, annotations in cct_json_db.image_id_to_annotations.items():
            docs[image_id]['annotations'] = {
                'species': []
            }
            if len(annotations) > 1:
                num_images_with_more_than_1_species += 1
            for anno in annotations:
                # convert the species category to explicit string name
                cat_name = cct_json_db.cat_id_to_name[anno['category_id']]
                docs[image_id]['annotations']['species'].append(cat_name)

                # there may be other fields in the annotation object
                for anno_field_name, anno_field_val in anno.items():
                    # these fields should already be gotten from the image object
                    if anno_field_name not in ['category_id', 'id', 'image_id', 'datetime', 'location', 'sequence_level_annotation', 'seq_id', 'seq_num_frames', 'frame_num']:
                        docs[image_id]['annotations'][anno_field_name] = anno_field_val

        print('Number of items from the image DB:', len(docs))
        print('Number of images with more than 1 species: {} ({}% of image DB)'.format(
            num_images_with_more_than_1_species, round(100 * num_images_with_more_than_1_species / len(docs), 2)))

    #%% integrate the bbox DB
    if args.bbox_db:
        print('Loading bbox DB...')
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
                    # 'bbox_abs': bbox_anno['bbox'],
                }

                if 'width' in docs[image_id]:
                    image_w = docs[image_id]['width']
                    image_h = docs[image_id]['height']
                    x, y, w, h = bbox_anno['bbox']
                    item_bbox['bbox'] = [
                        truncate_float(x / image_w),
                        truncate_float(y / image_h),
                        truncate_float(w / image_w),
                        truncate_float(h / image_h)
                    ]

                docs[image_id]['annotations']['bbox'].append(item_bbox)

            # not keeping height and width
            del docs[image_id]['width']
            del docs[image_id]['height']

        print('Number of images with more than one bounding box: {} ({}% of all entries)'.format(
            num_more_than_1_bbox, 100 * num_more_than_1_bbox / len(docs), 2))
    else:
        print('No bbox DB provided.')

    assert len(docs) > 0, 'No image entries found in the image or bbox DB jsons provided.'

    sequences = process_sequences(docs)

    #%% validation
    print('Example sequence items:')
    print()
    print(sequences[0])
    print()
    print(sequences[-1])
    print()

    sequences_schema_check.sequences_schema_check(sequences, args.dataset)

    write_json(args.partial_mega_db, sequences)


if __name__ == '__main__':
    main()

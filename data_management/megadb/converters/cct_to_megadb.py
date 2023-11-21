"""
cct_to_megadb.py

Given an image json and/or a bounding box json in the COCO Camera Trap format, output
the equivalent database in the MegaDB format so that the entries can be ingested into MegaDB.

All fields in the original image json would be carried over, and any fields in the
bounding box json but not in the corresponding entry in the image json will be added.

Check carefully that the dataset_name parameter is set correctly!!
"""

import argparse
import json
import os
import sys
import uuid
from collections import defaultdict
from copy import deepcopy
from random import sample

import numpy as np
from tqdm import tqdm

from ct_utils import truncate_float, write_json
from data_management.cct_json_utils import IndexedJsonDb
from data_management.megadb.schema import sequences_schema_check

# some property names have changed in the new schema
old_to_new_prop_name_mapping = {
    'file_name': 'file'
}


def process_sequences(embedded_image_objects, dataset_name, deepcopy_embedded=True):
    """
    Combine the image entries in an embedded COCO Camera Trap json from make_cct_embedded()
    into sequence objects that can be ingested to the `sequences` table in MegaDB.

    Image-level properties that have the same value are moved to the sequence level;
    sequence-level properties that have the same value are removed with a print-out
    describing what should be added to the `datasets` table instead.

    All strings in the array for the `class` property are lower-cased.

    Args:
        embedded_image_objects: array of image objects returned by make_cct_embedded()
        dataset_name: Make sure this is the desired name for the dataset
        deepcopy_embedded: True if to make a deep copy of `docs`; otherwise the `docs` object passed in will be modified!

    Returns:
        an array of sequence objects
    """
    print('The dataset_name is set to {}. Please make sure this is correct!'.format(dataset_name))

    if deepcopy_embedded:
        print('Making a deep copy of docs...')
        docs = deepcopy(embedded_image_objects)
    else:
        docs = embedded_image_objects

    print('Putting {} images into sequences...'.format(
        len(docs)))
    img_level_properties = set()
    sequences = defaultdict(list)

    # a dummy sequence ID will be generated if the image entry does not have a seq_id field
    # seq_id only needs to be unique within this dataset; MegaDB does not rely on it as the _id field

    # "annotations" fields are opened and have its sub-field surfaced one level up
    for im in tqdm(docs):
        if 'seq_id' in im:
            seq_id = im['seq_id']
            del im['seq_id']
        else:
            seq_id = 'dummy_' + uuid.uuid4().hex  # if this will be sent for annotation, may need a sequence ID based on file name to group potential sequences together
            img_level_properties.add('file')
            img_level_properties.add('image_id')

        for old_name, new_name in old_to_new_prop_name_mapping.items():
            if old_name in im:
                im[new_name] = im[old_name]
                del im[old_name]

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

    # set the `dataset` property on each sequence to the provided dataset_name
    new_sequences = []
    for seq_id, images in sequences.items():
        new_sequences.append({
            'seq_id': seq_id,
            'dataset': dataset_name,
            'images': images
        })
    sequences = new_sequences
    print('Number of sequences: {}'.format(len(sequences)))

    # check that the location field is the same for all images in a sequence
    print('Checking the location field...')
    for seq in sequences:
        locations = []
        for im in seq['images']:
            locations.append(im.get('location', ''))  # empty string if no location provided
        assert len(set(locations)) == 1, 'Location fields in images of the sequence {} are different.'.format(seq['seq_id'])

    # check which fields in a CCT image entry are sequence-level
    print('Checking which fields in a CCT image entry are sequence-level...')
    all_img_properties = set()
    for seq in sequences:
        if 'images' not in seq:
            continue

        image_properties = defaultdict(set)  # property name to stringfied property value

        for im in seq['images']:
            for prop_name, prop_value in im.items():
                image_properties[prop_name].add(str(prop_value))  # make all hashable
                all_img_properties.add(prop_name)

        for prop_name, prop_values in image_properties.items():
            if len(prop_values) > 1:
                img_level_properties.add(prop_name)

    # image-level properties that really should be sequence-level
    seq_level_properties = all_img_properties - img_level_properties

    # need to add (misidentified) seq properties not present for each image in a sequence to img_level_properties
    # (some properties act like flags - all have the same value, but not present on each img)
    bool_img_level_properties = set()
    for seq in sequences:
        if 'images' not in seq:
            continue
        for im in seq['images']:
            for seq_property in seq_level_properties:
                if seq_property not in im:
                    bool_img_level_properties.add(seq_property)
    seq_level_properties -= bool_img_level_properties
    img_level_properties |= bool_img_level_properties

    print('\nall_img_properties')
    print(all_img_properties)
    print('\nimg_level_properties')
    print(img_level_properties)
    print('\nimage-level properties that really should be sequence-level')
    print(seq_level_properties)
    print('')

    # add the sequence-level properties to the sequence objects
    for seq in sequences:
        if 'images' not in seq:
            continue

        for seq_property in seq_level_properties:
            # not every sequence have to have all the seq_level_properties
            if seq_property in seq['images'][0]:
                # get the value of this sequence-level property from the first image entry
                seq[seq_property] = seq['images'][0][seq_property]
                for im in seq['images']:
                    del im[seq_property]  # and remove it from the image level

    # check which fields are really dataset-level and should be included in the dataset table instead.
    seq_level_prop_values = defaultdict(set)
    for seq in sequences: 
        for prop_name in seq:
            if prop_name not in ['dataset', 'seq_id', 'class', 'images', 'location', 'bbox']:
                seq_level_prop_values[prop_name].add(seq[prop_name])
    dataset_props = []
    for prop_name, values in seq_level_prop_values.items():
        if prop_name == 'season':
            continue  # always keep 'season'
        if len(values) == 1:
            dataset_props.append(prop_name)
            print('! Sequence-level property {} with value {} should be a dataset-level property. Removed from sequences.'.format(prop_name, list(values)[0]))

    # delete sequence-level properties that should be dataset-level
    # make all `class` fields lower-case; cast `seq_id` to type string in case they're integers
    sequences_neat = []
    for seq in sequences:
        for dataset_prop in dataset_props:
            del seq[dataset_prop]

        seq['seq_id'] = str(seq['seq_id'])

        if 'class' in seq:
            seq['class'] = [c.lower() for c in set(seq['class'])]
        if 'images' in seq:
            for im in seq['images']:
                if 'class' in im:
                    im['class'] = [c.lower() for c in set(im['class'])]
        sequences_neat.append(sequences_schema_check.order_seq_properties(seq))

    # turn all float NaN values into None so it gets converted to null when serialized
    # this was an issue in the Snapshot Safari datasets
    for seq in sequences_neat:
        for seq_prop, seq_prop_value in seq.items():
            if isinstance(seq_prop_value, float) and np.isnan(seq_prop_value):
                seq[seq_prop] = None

            if seq_prop == 'images':
                for im_idx, im in enumerate(seq['images']):
                    for im_prop, im_prop_value in im.items():
                        if isinstance(im_prop_value, float) and np.isnan(im_prop_value):
                            seq['images'][im_idx][im_prop] = None

    print('Finished processing sequences.')
    #%% validation
    print('Example sequence items:')
    print()
    print(json.dumps(sequences_neat[0]))
    print()
    print(json.dumps(sample(sequences_neat, 1)[0]))
    print()

    return sequences_neat


def make_cct_embedded(image_db=None, bbox_db=None):
    """
    Takes in path to the COCO Camera Trap format jsons for images (species labels) and/or
    bboxes (animal/human/vehicle) labels and embed the class names and annotations into the image entries.

    Since IndexedJsonDb() can take either a path or a loaded json object as a dict, both
    arguments can be paths or loaded json objects

    Returns:
        an embedded version of the COCO Camera Trap format json database
    """


    # at first a dict of image_id: image_obj with annotations embedded, then it becomes
    # an array of image objects
    docs = {}

    # %% integrate the image DB
    if image_db:
        print('Loading image DB...')
        cct_json_db = IndexedJsonDb(image_db)
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
    if bbox_db:
        print('Loading bbox DB...')
        cct_bbox_json_db = IndexedJsonDb(bbox_db)

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
                if 'bbox' in bbox_anno:
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

            # not keeping height and width
            del docs[image_id]['width']
            del docs[image_id]['height']

        print('Number of images with more than one bounding box: {} ({}% of all entries)'.format(
            num_more_than_1_bbox, 100 * num_more_than_1_bbox / len(docs), 2))
    else:
        print('No bbox DB provided.')

    assert len(docs) > 0, 'No image entries found in the image or bbox DB jsons provided.'

    docs = list(docs.values())
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_name',
        help='A short string representing the dataset to be used as a partition key in MegaDB')
    parser.add_argument(
        '--image_db',
        help='Path to the json containing the image DB in CCT format')
    parser.add_argument(
        '--bbox_db',
        help='Path to the json containing the bbox DB in CCT format')
    parser.add_argument(
        '--docs',
        help='Embedded CCT format json to use instead of image_db or bbox_db')
    parser.add_argument(
        '--partial_mega_db',
        required=True,
        help='Path to store the resulting json')
    args = parser.parse_args()

    assert len(args.dataset_name) > 0, 'dataset_name cannot be an empty string'

    if args.image_db:
        assert os.path.exists(args.image_db), 'image_db file path provided does not point to a file'
    if args.bbox_db:
        assert os.path.exists(args.bbox_db), 'bbox_db file path provided does not point to a file'

    docs = make_cct_embedded(args.image_db, args.bbox_db)

    sequences = process_sequences(docs, args.dataset_name)

    sequences_schema_check.sequences_schema_check(sequences)

    write_json(args.partial_mega_db, sequences)


if __name__ == '__main__':
    main()

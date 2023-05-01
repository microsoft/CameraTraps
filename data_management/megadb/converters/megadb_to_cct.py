"""
Converts a JSON list of sequence items from the `sequences` table in the database to
an individual json database in the COCO Camera Trap (CCT) format.

! First use the `megadb_coords_to_abs.py` script to get a version of the list of sequence items with absolute
coordinates for bbox labels.

`sequences` table format: https://github.com/microsoft/CameraTraps/tree/master/data_management/megadb
COCO Camera Trap: https://github.com/microsoft/CameraTraps/tree/master/data_management#coco-camera-traps-format

If an image is included in a bbox JSON but does not have an annotation entry, it is confirmed empty. 
"""

import argparse
import json
import os
import uuid
from datetime import datetime

from tqdm import tqdm

from ct_utils import write_json
from data_management.cct_json_utils import CameraTrapJsonUtils


def break_into_images_annotations(mega_db, bbox_only):
    cct_images = []
    cct_annotations = []

    num_images_in_cct = 0
    num_images_skipped = 0

    for seq in tqdm(mega_db):
        assert 'seq_id' in seq and 'images' in seq
        for i in seq['images']:
            assert 'file' in i

        seq_level_classes = seq.get('class', [])

        seq_level_props = {}
        for prop_name, prop_val in seq.items():
            # `id` is from the database, as well as all attributes starting with _
            if prop_name in ['seq_id', 'images', 'class', 'id']:
                continue
            if prop_name.startswith('_'):
                continue
            seq_level_props[prop_name] = prop_val

        # if valuable sequence information is available, add them to the image
        seq_info_available = True if not seq['seq_id'].startswith('dummy_') else False
        if seq_info_available:
            seq_num_frames = len(seq['images'])

        for im in seq['images']:

            if 'bbox' not in im:
                num_images_skipped += 1
                continue

            num_images_in_cct += 1

            # required fields for an image object
            im_object = {
                'id': im['image_id'] if 'image_id' in im else str(uuid.uuid1()),
                'file_name': im['file']
            }

            if seq_info_available:
                im_object['seq_id'] = seq['seq_id']
                im_object['seq_num_frames'] = seq_num_frames
                if 'frame_num' in im:
                    im_object['frame_num'] = im['frame_num']

            # add seq-level class labels for this image
            if not bbox_only and len(seq_level_classes) > 0:
                for cls in seq_level_classes:
                    cct_annotations.append({
                        'id': str(uuid.uuid1()),
                        'image_id': im_object['id'],
                        'sequence_level_annotation': True,
                        'category_name': cls  # later converted to category_id
                    })
            # add other sequence-level properties to each image too
            for seq_prop, seq_prop_val in seq_level_props.items():
                im_object[seq_prop] = seq_prop_val

            # add other image-level properties
            for im_prop in im:
                if im_prop in ['file', 'frame_num', 'id', 'file_name']:
                    continue  # already added or need to leave out (e.g. 'id')
                elif im_prop == 'class':  # image-level "species" labels; not the bbox type labels

                    if bbox_only:
                        continue

                    for cls in im['class']:
                        if cls not in seq_level_classes:
                            cct_annotations.append({
                                'id': str(uuid.uuid1()),
                                'image_id': im_object['id'],
                                'category_name': cls  # later converted to category_id
                            })
                elif im_prop == 'bbox':
                        for b in im['bbox']:
                            cct_annotations.append({
                                'id': str(uuid.uuid1()),
                                'image_id': im_object['id'],
                                'category_name': b['category'],
                                'bbox': b['bbox']
                            })
                else:
                    im_object[im_prop] = im[im_prop]

            cct_images.append(im_object)
        # ... for im in seq['images']
    # ... for seq in mega_db

    print('Number of empty images: {}'.format(num_images_skipped))
    return cct_images, cct_annotations


def megadb_to_cct(dataset_name, mega_db, output_path, bbox_only):

    mega_db = [i for i in mega_db if i['dataset'] == dataset_name]
    assert len(mega_db) > 0, 'There are no entries from the dataset {}'.format(dataset_name)
    for i in mega_db:
        del i['dataset']  # all remaining fields will be added to the CCT database
    print('Number of entries belonging to dataset {}: {}'.format(dataset_name, len(mega_db)))

    cct_images, cct_annotations = break_into_images_annotations(mega_db, bbox_only)

    # consolidate categories
    category_names = set()
    for anno in cct_annotations:
        category_names.add(anno['category_name'])

    cat_name_to_id = {
        'empty': 0  # always set empty to 0 even for dataset without 'empty' labeled images
    }

    if bbox_only:
        cat_name_to_id['animal'] = 1
        cat_name_to_id['person'] = 2
        cat_name_to_id['group'] = 3
        cat_name_to_id['vehicle'] = 4

    for cat in category_names:
        if cat not in cat_name_to_id:
            cat_name_to_id[cat] = len(cat_name_to_id)

    for anno in cct_annotations:
        anno['category_id'] = cat_name_to_id[anno['category_name']]
        del anno['category_name']

    cct_categories = []
    for name, num_id in cat_name_to_id.items():
        cct_categories.append({
            'id': num_id,
            'name': name
        })

    print('Final CCT DB has {} image entries, and {} annotation entries.'.format(len(cct_images), len(cct_annotations)))
    cct_db = {
        'info': {
            'version': str(datetime.now()),
            'date_created': str(datetime.today().date()),
            'description': ''  # to be filled by main()
        },
        'images': cct_images,
        'categories': cct_categories,
        'annotations': cct_annotations
    }
    cct_db = CameraTrapJsonUtils.order_db_keys(cct_db)

    cct_db['info']['description'] = 'COCO Camera Traps database converted from sequences in dataset {}'.format(
        dataset_name)
    print('Writing to output file...')
    write_json(output_path, cct_db)
    print('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_name',
        help='The name of the dataset; only entries from this dataset will be used')
    parser.add_argument(
        'mega_db',
        help='A json containing a list of sequence objects. Should be the output of megadb_coords_to_abs.py')
    parser.add_argument(
        'out_cct_db',
        help='Path to store the resulting json')
    parser.add_argument(
        '--bbox_only',
        action='store_true',
        help='If flagged, only bbox labeled images will be included and no species labels will be included')
    args = parser.parse_args()

    assert len(args.dataset_name) > 0, 'dataset_name cannot be an empty string'
    assert os.path.exists(args.mega_db), 'File at mega_db path does not exist'

    with open(args.mega_db) as f:
        mega_db = json.load(f)
    print('Number of entries in the mega_db: {}'.format(len(mega_db)))

    megadb_to_cct(args.dataset_name, mega_db, args.out_cct_db, args.bbox_only)


if __name__ == '__main__':
    main()

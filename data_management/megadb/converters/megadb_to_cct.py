"""
Converts a JSON list of sequence items from the `sequences` table in the database to
an individual json database in the COCO Camera Trap (CCT) format.

!! Note that because we don't save the images' width and height in the database, we can't
convert the bounding box coordinates back to absolute coordinates (a CCT specification).
The coordinates are in the same order (x_min, y_min, width_box, height_box), just in relative coordinates.

`sequences` table format: https://github.com/microsoft/CameraTraps/tree/master/data_management/megadb
COCO Camera Trap: https://github.com/microsoft/CameraTraps/tree/master/data_management#coco-camera-traps-format
"""

import argparse
import json
import os
import uuid
from datetime import datetime
import sys

from tqdm import tqdm

from ct_utils import write_json
from data_management.cct_json_utils import CameraTrapJsonUtils


# TODO when we create the category IDs, we have to put the bbox IDs at the "top" so they can be distinguished from the species?


def break_into_images_annotations(mega_db):
    cct_images = []
    cct_annotations = []

    for seq in tqdm(mega_db):
        assert 'seq_id' in seq and 'images' in seq
        for i in seq['images']:
            assert 'file' in i

        seq_level_classes = seq.get('class', [])

        seq_level_props = {k: v for k, v in seq.items() if k not in ['seq_id', 'images', 'class']}

        # if valuable sequence information is available, add them to the image
        seq_info_available = True if not seq['seq_id'].startswith('dummy_') else False
        if seq_info_available:
            seq_num_frames = len(seq['images'])

        for im in seq['images']:
            # required fields for an image object
            im_object = {
                'id': str(uuid.uuid1()),
                'file_name': im['file']
            }

            if seq_info_available:
                im_object['seq_id'] = seq['seq_id']
                im_object['seq_num_frames'] = seq_num_frames
                if 'frame_num' in im:
                    im_object['frame_num'] = im['frame_num']

            # add seq-level class labels for this image
            if len(seq_level_classes) > 0:
                for cls in seq_level_classes:
                    cct_annotations.append({
                        'id': str(uuid.uuid1()),
                        'image_id': im_object['id'],
                        'sequence_level_annotation': True,
                        'category_name': cls  # later converted to category_id
                    })

            # add other image-level properties
            for im_prop in im:
                if im_prop in ['file', 'frame_num', 'id', 'file_name']:
                    continue  # already added or need to leave out (e.g. 'id')
                elif im_prop == 'class':  # image-level "species" labels; not the bbox type labels
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
                            'bbox': b['bbox']  # cannot convert to absolute coords, so leaving as is
                        })
                else:
                    im_object[im_prop] = im[im_prop]

            # add sequence-level properties to each image too
            for seq_prop, seq_prop_val in seq_level_props.items():
                im_object[seq_prop] = seq_prop_val

            cct_images.append(im_object)
        # ... for im in seq['images']
    # ... for seq in mega_db
    return cct_images, cct_annotations


def megadb_to_cct(dataset_name, mega_db, output_path):

    mega_db = [i for i in mega_db if i['dataset'] == dataset_name]
    assert len(mega_db) > 0, 'There are no entries from the dataset {}'.format(dataset_name)
    for i in mega_db:
        del i['dataset']  # all remaining fields will be added to the CCT database
    print('Number of entries belonging to dataset {}: {}'.format(dataset_name, len(mega_db)))

    cct_images, cct_annotations = break_into_images_annotations(mega_db)

    # consolidate categories
    category_names = set()
    for anno in cct_annotations:
        category_names.add(anno['category_name'])
        
    cat_name_to_id = {
        'empty': 0  # always set empty to 0 even for dataset without 'empty' labeled images
    }
    for cat in category_names:
        if cat != 'empty':
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
        help='A json containing a list of sequence objects')
    parser.add_argument(
        'out_cct_db',
        help='Path to store the resulting json')
    args = parser.parse_args()

    assert len(args.dataset_name) > 0, 'dataset_name cannot be an empty string'
    assert os.path.exists(args.mega_db), 'File at mega_db path does not exist'

    with open(args.mega_db) as f:
        mega_db = json.load(f)
    print('Number of entries in the mega_db: {}'.format(len(mega_db)))

    megadb_to_cct(args.dataset_name, mega_db, args.out_cct_db)


if __name__ == '__main__':
    main()

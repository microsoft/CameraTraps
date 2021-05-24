#
# add_bounding_boxes_to_megadb.py
#
# Given COCO-formatted JSONs containing manually labeled bounding box annotations, add them to
# MegaDB sequence entries, which can then be ingested into MegaDB.


import argparse
import os
import json
from collections import defaultdict
from typing import Dict, Tuple

from tqdm import tqdm
import path_utils  # ai4eutils

from data_management.cct_json_utils import IndexedJsonDb
import ct_utils


# the category map that comes in the COCO JSONs for iMerit batch 12 - to check that each
# JSON
bbox_categories = [
    {
        'id': 1,
        'name': 'Animal'
    },
    {
        'id': 2,
        'name': 'Group'
    },
    {
        'id': 3,
        'name': 'Human'
    },
    {
        'id': 4,
        'name': 'Vehicle'
    },
    {
        'id': 5,
        'name': 'No Object Visible'
    }
]
bbox_categories_str = json.dumps(bbox_categories).lower()  # cct_json_utils also lower()

bbox_cat_map = {c['id']: c['name'].lower() for c in bbox_categories[:4]}
bbox_cat_map[3] = 'person'  # MegaDB categories are "animal", "person" and "vehicle"
assert bbox_cat_map[4] == 'vehicle'


def file_name_to_parts(image_file_name) -> Tuple[str, str, int]:
    """
    Given the `file_name` field in an iMerit annotation, return the dataset name,
    sequence id and frame number.
    """
    parts = image_file_name.split('.')
    dataset_name = parts[0]
    seq_id = parts[1].split('seq')[1]
    frame_num = int(parts[2].split('frame')[1])
    return dataset_name, seq_id, frame_num


def add_annotations_to_sequences(annotations_dir: str, temp_sequences_dir: str, sequences_dir: str):
    """
    Extract the bounding box annotations from the COCO JSONs for all datasets labeled in this round.

    Args:
        annotations_dir: Path to directory with the annotations in COCO JSONs at the root level.
        temp_sequences_dir: Path to a flat directory of JSONs ending in '_temp.json' which are
            MegaDB sequences without the bounding box annotations.
        sequences_dir: Path to a directory to output corresponding bounding box-included sequences
            in MegaDB format.

    Returns:
        None. JSON files will be written to sequences_dir.
    """
    assert os.path.exists(annotations_dir), \
        f'annotations_dir {annotations_dir} does not exist'
    assert os.path.isdir(annotations_dir), \
        f'annotations_dir {annotations_dir} is not a directory'
    assert os.path.exists(temp_sequences_dir), \
        f'temp_sequences_dir {temp_sequences_dir} does not exist'
    assert os.path.isdir(temp_sequences_dir), \
        f'temp_sequences_dir {temp_sequences_dir} is not a directory'
    os.makedirs(sequences_dir, exist_ok=True)

    temp_megadb_files = path_utils.recursive_file_list(temp_sequences_dir)
    temp_megadb_files = [i for i in temp_megadb_files if i.endswith('.json')]
    print(f'{len(temp_megadb_files)} temporary MegaDB dataset files found.')

    annotation_files = path_utils.recursive_file_list(annotations_dir)
    annotation_files = [i for i in annotation_files if i.endswith('.json')]
    print(f'{len(annotation_files)} annotation_files found. Extracting annotations...')

    # dataset name : (seq_id, frame_num) : [bbox, bbox]
    # where bbox is a dict with str 'category' and list 'bbox'
    all_image_bbox: Dict[str, Dict[Tuple[str, int], list]]
    all_image_bbox = defaultdict(lambda: {})

    for p in tqdm(annotation_files):
        incoming_coco = IndexedJsonDb(p)
        assert bbox_categories_str == json.dumps(incoming_coco.db['categories']), \
            f'Incoming COCO JSON has a different category mapping! {p}'

        # iterate over image_id_to_image rather than image_id_to_annotations so we include
        # the confirmed empty images
        for image_id, image_entry in incoming_coco.image_id_to_image.items():
            image_file_name = image_entry['file_name']
            # The file_name field in the incoming json looks like
            # alka_squirrels.seq2020_05_07_25C.frame119221.jpg
            dataset_name, seq_id, frame_num = file_name_to_parts(image_file_name)
            bbox_field = []  # empty means this image is confirmed empty

            annotations = incoming_coco.image_id_to_annotations.get(image_id, [])
            for coco_anno in annotations:
                if coco_anno['category_id'] == 5:
                    assert len(coco_anno['bbox']) == 0, f'{coco_anno}'

                    # there seems to be a bug in the annotations where sometimes there's a
                    # non-empty label along with a label of category_id 5
                    # ignore the empty label (they seem to be actually non-empty)
                    continue

                assert coco_anno['category_id'] is not None, f'{p} {coco_anno}'

                bbox_field.append({
                    'category': bbox_cat_map[coco_anno['category_id']],
                    'bbox': ct_utils.truncate_float_array(coco_anno['bbox'], precision=4)
                })
            all_image_bbox[dataset_name][(seq_id, frame_num)] = bbox_field

    print('\nAdding bounding boxes to the MegaDB dataset files...')
    for p in temp_megadb_files:
        basename = os.path.basename(p)
        dataset_name = basename.split('_temp.')[0] if basename.endswith('_temp.json') \
            else basename.split('.json')[0]
        print(f'Adding to dataset {dataset_name}')
        dataset_image_bbox = all_image_bbox.get(dataset_name, None)
        if dataset_image_bbox is None:
            print('Skipping, no annotations found for this dataset\n')
            continue

        with open(p) as f:
            sequences = json.load(f)

        num_images_updated = 0
        for seq in tqdm(sequences):
            assert seq['dataset'] == dataset_name
            seq_id = seq['seq_id']
            for im in seq['images']:
                frame_num = im.get('frame_num', 1)
                bbox_field = dataset_image_bbox.get((seq_id, frame_num), None)
                if bbox_field is not None:  # empty list also evaluates to False
                    im['bbox'] = bbox_field
                    num_images_updated += 1
        print(f'Dataset {dataset_name} had {num_images_updated} images updated\n')

        with open(os.path.join(sequences_dir, f'{dataset_name}.json'),
                  'w', encoding='utf-8') as f:
            json.dump(sequences, f, indent=1, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'annotations_dir', type=str,
        help='Path to directory with the annotations in COCO JSONs at the root level.'
    )
    parser.add_argument(
        'temp_sequences_dir', type=str,
        help='Path to a flat directory of JSONs of MegaDB sequence entries, ending in _temp.json'
    )
    parser.add_argument(
        'sequences_dir', type=str,
        help='Path to output directory. Will be created if it does not exist.'
    )
    args = parser.parse_args()

    add_annotations_to_sequences(args.annotations_dir, args.temp_sequences_dir, args.sequences_dir)


if __name__ == '__main__':
    main()
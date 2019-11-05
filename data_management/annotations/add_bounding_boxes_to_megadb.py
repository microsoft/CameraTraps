import os
import json
from collections import defaultdict

from tqdm import tqdm

import ct_utils


bbox_categories = [
    {
      "id": "1",
      "name": "animal"
    },
    {
      "id": "2",
      "name": "person"
    },
    {
      "id": "3",
      "name": "group"
    },
    {
      "id": "4",
      "name": "vehicles"
    }
  ]

bbox_cat_map = {int(c['id']): c['name'] for c in bbox_categories}
bbox_cat_map[4] = 'vehicle'  # change to the singular form to be consistent


def extract_annotations(annotation_path, dataset_name):
    """
    Extract the bounding box annotations from the pseudo-jsons iMerit sends us for a single dataset.

    Note that the image filename in the returned mapping is lowercased.

    Args:
        annotation_path: a list or string; the list of annotation entries, a path to a directory
            containing pseudo-jsons with the annotations or a path to a single pseudo-json
        dataset_name: string used to identify this dataset when the images were sent for annotation.
            Note that this needs to be the same as what's in the annotation files, if different
            from what's in the `dataset` table

    Returns:
        image_filename_to_bboxes: a dict of image filename (lower-cased) to the bbox items ready to
        insert to MegaDB sequences' image objects.
    """
    content = []
    if type(annotation_path) == str:
        assert os.path.exists(annotation_path), 'annotation_paths provided does not exist as a dir or file'

        if os.path.isdir(annotation_path):
            # annotation_path points to a directory containing annotation pseudo-jsons
            for file_name in os.listdir(annotation_path):
                p = os.path.join(annotation_path, file_name)
                with open(p) as f:
                    c = f.readlines()
                    content.extend(c)
            print('{} files found in directory at annotation_path'.format(len(os.listdir(annotation_path))))
        else:
            # annotation_path points to a single annotation pseudo-json
            with open(annotation_path) as f:
                content = f.readlines()

    else:
        assert type(annotation_path) == list, 'annotation_paths provided is not a string (path) or list'

    print('Number of annotation entries found: {}'.format(len(content)))

    image_filename_to_bboxes = defaultdict(list)
    num_bboxes = 0
    num_bboxes_skipped = 0

    # each row in this pseudo-json is a COCO formatted entry for an image sequence
    for row in tqdm(content):
        entry = json.loads(row)

        entry_categories = entry.get('categories', [])
        assert json.dumps(bbox_categories, sort_keys=True) == json.dumps(entry_categories, sort_keys=True)

        entry_annotations = entry.get('annotations', [])

        for anno in entry_annotations:
            assert 'image_id' in anno
            assert 'bbox' in anno
            assert len(anno['bbox']) == 4
            assert 'category_id' in anno
            assert type(anno['category_id']) == int

            image_ref = anno['image_id']  # iMerit calls this field image_id

            dataset = image_ref.split('dataset')[1].split('.')[0]
            if dataset != dataset_name:
                num_bboxes_skipped += 1
                continue

            # lower-case all image filenames !
            image_filename = image_ref.split('.img')[1].lower()

            bbox_coords = anno['bbox']  # [x_rel, y_rel, w_rel, h_rel]
            bbox_coords = ct_utils.truncate_float_array(bbox_coords, precision=4)

            bbox_entry = {
                'category': bbox_cat_map[anno['category_id']],
                'bbox': bbox_coords
            }

            image_filename_to_bboxes[image_filename].append(bbox_entry)
            num_bboxes += 1

    print('{} boxes on {} images were in the annotation file(s). {} boxes skipped because they are not for the requested dataset'.format(
        num_bboxes, len(image_filename_to_bboxes), num_bboxes_skipped))

    # how many boxes of each category?
    print('\nCategory counts for the bboxes:')
    category_count = defaultdict(int)
    for filename, bboxes in image_filename_to_bboxes.items():
        for b in bboxes:
            category_count[b['category']] += 1
    for category, count in sorted(category_count.items()):
        print('{}: {}'.format(category, count))

    return image_filename_to_bboxes


def zsl_image_filename_map_func(db_img_obj):
    return db_img_obj['id'] + '.jpg'

def default_image_filename_map_func(db_img_obj):
    return db_img_obj['file']


def add_annotations_to_sequences(sequences, image_filename_to_bboxes,
                                 im_id_map_func=default_image_filename_map_func):
    """

    Args:
        sequences:
        image_filename_to_bboxes:
        im_id_map_func:

    Returns:
        the original sequences list updated with bbox annotations (not a copy)
    """

    # check that all sequences are for a single dataset; each may need adjustment to how image
    # identifiers are mapped

    datasets_present = set([s['dataset'] for s in sequences])
    assert len(datasets_present) == 1, 'the sequences provided need to come from a single dataset'

    print('Dataset to which the sequences belong to: {}. Make sure this is the intended dataset where the bboxes also come from!'.format(datasets_present.pop()))

    images_updated = []
    num_images_rewritten = 0
    num_images_no_anno_added = 0

    for seq in sequences:
        if 'images' not in seq:
            continue

        for im in seq['images']:
            im_id = im_id_map_func(im)
            if im_id in image_filename_to_bboxes:

                if 'bbox' in im:  # if we are overwriting existing bbox annotations
                    num_images_rewritten += 1

                im['bbox'] = image_filename_to_bboxes[im_id]  # reference type updates reflected in original list
                images_updated.append(im_id)
            else:
                num_images_no_anno_added += 1

    print('{} images updated; {} images had their bbox overwritten; {} images not updated'.format(
        len(images_updated), num_images_rewritten, num_images_no_anno_added
    ))
    return sequences, images_updated

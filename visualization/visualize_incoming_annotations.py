#####
#
# visualize_incoming_annotations.py
#
# Spot-check the annotations received from iMerit by visualizing annotated bounding 
# boxes on a sample of images and display them in HTML.
#
# Modified in 2021 March to use the new format (iMerit batch 12 onwards), which is a
# COCO formatted JSON with relative coordinates for the bbox.
#

import argparse
import io
import json
import os

from collections import defaultdict
from random import sample

from tqdm import tqdm
from write_html_image_list import write_html_image_list # Assumes ai4eutils is on the path

#from data_management.megadb.schema import sequences_schema_check
from data_management.megadb.megadb_utils import MegadbUtils
from data_management.annotations.add_bounding_boxes_to_megadb import file_name_to_parts
from data_management.cct_json_utils import IndexedJsonDb
from visualization import visualization_utils as vis_utils


def get_image_rel_path(dataset_seq_images, dataset_name, seq_id, frame_num):
    images = dataset_seq_images[dataset_name].get(seq_id, None)
    if images is None:
        return None
    for im in images:
        if im.get('frame_num', None) == frame_num:
            return im['file']
        
    # we used frame_num of 1 when sending out images to annotators when it is not explicitly stored (wcs esp.)
    if frame_num == 1 and len(images) == 1:
        return images[0]['file']

    return None


def visualize_incoming_annotations(args):
    print('Connecting to MegaDB to get the datasets table...')
    megadb_utils = MegadbUtils()
    datasets_table = megadb_utils.get_datasets_table()

    print('Loading the MegaDB entries...')
    with open(args.megadb_entries) as f:
        sequences = json.load(f)
    print(f'Total number of sequences: {len(sequences)}')
    dataset_seq_images = defaultdict(dict)
    for seq in sequences:
        dataset_seq_images[seq['dataset']][seq['seq_id']] = seq['images']

    print('Loading incoming annotation entries...')
    incoming = IndexedJsonDb(args.incoming_annotation)
    print(f'Number of images in this annotation file: {len(incoming.image_id_to_image)}')

    if args.num_to_visualize != -1 and args.num_to_visualize <= len(incoming.image_id_to_image):
        incoming_id_to_anno = sample(list(incoming.image_id_to_annotations.items()),
                                     args.num_to_visualize)
    else:
        incoming_id_to_anno = incoming.image_id_to_annotations.items()

    # The file_name field in the incoming json looks like
    # alka_squirrels.seq2020_05_07_25C.frame119221.jpg
    # we need to use the dataset, sequence and frame info to find the actual path in blob storage
    # using the sequences
    images_html = []
    for image_id, annotations in tqdm(incoming_id_to_anno):
        if args.trim_to_images_bboxes_labeled and annotations[0]['category_id'] == 5:
            # category_id 5 is No Object Visible
            continue

        anno_file_name = incoming.image_id_to_image[image_id]['file_name']
        dataset_name, seq_id, frame_num = file_name_to_parts(anno_file_name)

        im_rel_path = get_image_rel_path(dataset_seq_images, dataset_name, seq_id, frame_num)
        if im_rel_path is None:
            print(f'Not found in megadb entries: dataset {dataset_name},'
                  f' seq_id {seq_id}, frame_num {frame_num}')
            continue

        im_full_path = megadb_utils.get_full_path(datasets_table, dataset_name, im_rel_path)

        # download the image
        container_client = megadb_utils.get_storage_client(datasets_table, dataset_name)
        downloader = container_client.download_blob(im_full_path)
        image_file = io.BytesIO()
        blob_props = downloader.download_to_stream(image_file)
        image = vis_utils.open_image(image_file)

        boxes = [anno['bbox'] for anno in annotations]
        classes = [anno['category_id'] for anno in annotations]

        vis_utils.render_iMerit_boxes(boxes, classes, image, label_map=incoming.cat_id_to_name)

        file_name = '{}_gtbbox.jpg'.format(os.path.splitext(anno_file_name)[0].replace('/', '~'))
        image = vis_utils.resize_image(image, args.output_image_width)
        image.save(os.path.join(args.output_dir, 'rendered_images', file_name))

        images_html.append({
            'filename': '{}/{}'.format('rendered_images', file_name),
            'title': '{}, number of boxes: {}'.format(anno_file_name, len([b for b in boxes if len(b) > 0])),
            'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;text-align:left;margin-top:20;margin-bottom:5'
        })

    # Write to HTML
    images_html = sorted(images_html, key=lambda x: x['filename'])
    write_html_image_list(
        filename=os.path.join(args.output_dir, 'index.html'),
        images=images_html,
        options={
            'headerHtml': '<h1>Sample annotations from {}</h1>'.format(args.incoming_annotation)
        })

    print('Visualized {} images.'.format(len(images_html)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'incoming_annotation', type=str,
        help='Path to a json in the COCO format with relative coordinates for the bbox from annotators')
    parser.add_argument(
        'megadb_entries', type=str,
        help='Path to a json list of MegaDB entries to look up image path in blob storage')
    parser.add_argument(
        'output_dir', action='store', type=str,
        help='Output directory for html and rendered images')
    parser.add_argument(
        '--trim_to_images_bboxes_labeled', action='store_true',
        help='Only include images that have been sent for bbox labeling (but '
             'may be actually empty). Turn this on if QAing annotations.')
    parser.add_argument(
        '--num_to_visualize', action='store', type=int, default=200,
        help='Number of images to visualize. If trim_to_images_bboxes_labeled, there may be fewer than specified')
    parser.add_argument(
        '-w', '--output_image_width', type=int, default=1200,
        help='an integer indicating the desired width in pixels of the output '
             'annotated images. Use -1 to not resize.')

    args = parser.parse_args()

    assert 'COSMOS_ENDPOINT' in os.environ and 'COSMOS_KEY' in os.environ

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rendered_images'), exist_ok=True)

    visualize_incoming_annotations(args)


if __name__ == '__main__':
    main()

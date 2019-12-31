
import argparse
import json
import os
import sys
from random import shuffle
from multiprocessing.pool import ThreadPool
from functools import partial
import io

from tqdm import tqdm
from azure.storage.blob import BlockBlobService

# Assumes ai4eutils is on the path (github.com/Microsoft/ai4eutils)
from write_html_image_list import write_html_image_list

from data_management.megadb.schema import sequences_schema_check
from visualization import visualization_utils as vis_utils


def get_blob_service(datasets_table, dataset_name):
    """
    Return the azure.storage.blob BlockBlobService object corresponding to the dataset

    datasets_table is updated (no new copy) if a new BlockBlobService is created for a dataset
    """
    if dataset_name not in datasets_table:
        raise KeyError('Dataset {} is not in the datasets table.'.format(dataset_name))

    entry = datasets_table[dataset_name]

    if 'blob_service' in entry:
        return entry['blob_service']

    # need to create a new blob service for this dataset
    if 'container_sas_key' not in entry:
        raise KeyError('Dataset {} does not have the container_sas_key field in the datasets table.'.format(dataset_name))

    # the SAS token can be just for the container, not the storage account
    # - will be fine for accessing files in that container later
    blob_service = BlockBlobService(account_name=entry['storage_account'],
                                    sas_token=entry['container_sas_key'])
    datasets_table[dataset_name]['blob_service'] = blob_service  # in-place update
    return blob_service


def get_full_path(datasets_table, dataset_name, img_path):
    entry = datasets_table[dataset_name]
    if 'path_prefix' not in entry or entry['path_prefix'] == '':
        return img_path
    else:
        return os.path.join(entry['path_prefix'], img_path)


def render_image_info(rendering, args):
    blob_service = rendering['blob_service']
    image_obj = io.BytesIO()
    _ = blob_service.get_blob_to_stream(rendering['container_name'], rendering['blob_path'], image_obj)

    # resize is for displaying them more quickly
    image = vis_utils.resize_image(vis_utils.open_image(image_obj), args.output_image_width)
    vis_utils.render_megadb_bounding_boxes(rendering['bbox'], image)

    annotated_img_name = rendering['annotated_img_name']
    annotated_img_path = os.path.join(args.output_dir, 'rendered_images', annotated_img_name)
    image.save(annotated_img_path)


def visualize_sequences(datasets_table, sequences, args):
    num_images = 0

    images_html = []
    rendering_info = []

    for seq in sequences:
        if 'images' not in seq:
            continue

        # dataset and seq_id are required fields
        dataset_name = seq['dataset']
        seq_id = seq['seq_id']

        # sort the images in the sequence

        images_in_seq = sorted(seq['images'], key=lambda x: x['frame_num']) if len(seq['images']) > 1 else seq['images']

        for im in images_in_seq:
            if args.trim_to_images_bboxes_labeled and 'bbox' not in im:
                continue

            num_images += 1

            blob_path = get_full_path(datasets_table, dataset_name, im['file'])
            frame_num = im.get('frame_num', -1)
            im_class = im.get('class', None)
            if im_class is None:  # if no class label on the image, show the class label on the sequence
                im_class = seq.get('class', [])

            rendering = {}
            rendering['blob_service'] = get_blob_service(datasets_table, dataset_name)
            rendering['container_name'] = datasets_table[dataset_name]['container']
            rendering['blob_path'] = blob_path
            rendering['bbox'] = im.get('bbox', [])

            annotated_img_name = 'anno_' + blob_path.replace('/', args.pathsep_replacement).replace('\\', args.pathsep_replacement)
            rendering['annotated_img_name'] = annotated_img_name

            rendering_info.append(rendering)

            images_html.append({
                'filename': 'rendered_images/{}'.format(annotated_img_name),
                'title': 'Seq ID: {}. Frame number: {}<br/> Image file: {}<br/> number of boxes: {}, image class labels: {}'.format(seq_id, frame_num, blob_path, len(rendering['bbox']), im_class),
                'textStyle': 'font-family:verdana,arial,calibri;font-size:80%;text-align:left;margin-top:20;margin-bottom:5'
            })

        if num_images >= args.num_to_visualize:
            print('num_images visualized is {}'.format(num_images))
            break

    # pool = ThreadPool()
    render_image_info_partial = partial(render_image_info, args=args)
    # print('len of rendering_info', len(rendering_info))
    # tqdm(pool.imap_unordered(render_image_info_partial, rendering_info), total=len(rendering_info))

    for rendering in tqdm(rendering_info):
        render_image_info_partial(rendering)

    print('Making HTML...')

    html_path = os.path.join(args.output_dir, 'index.html')
    # options = write_html_image_list()
    # options['headerHtml']
    write_html_image_list(
        filename=html_path,
        images=images_html
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets_table', type=str, help='Path to a json equivalent of the datasets table in MegaDB')
    parser.add_argument('megadb_entries', type=str, help='Path to a json list of MegaDB entries')
    parser.add_argument('output_dir', action='store', type=str,
                        help='Output directory for html and rendered images')
    parser.add_argument('--trim_to_images_bboxes_labeled', action='store_true',
                        help='Only include images that have been sent for bbox labeling (but may be actually empty). Turn this on if QAing annotations.')
    parser.add_argument('--num_to_visualize', action='store', type=int, default=200,
                        help='Number of images to visualize (all comformant images in a sequence are shown, so may be a few more than specified). Sequences are shuffled. Defaults to 200. Use -1 to visualize all.')

    parser.add_argument('--pathsep_replacement', action='store', type=str, default='~',
                        help='Replace path separators in relative filenames with another character (default ~)')
    parser.add_argument('-w', '--output_image_width', type=int,
                        help=('an integer indicating the desired width in pixels of the output annotated images. '
                              'Use -1 to not resize.'),
                        default=700)

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'rendered_images'))

    with open(args.datasets_table) as f:
        datasets_table = json.load(f)

    datasets_table = {i['dataset_name']: {k: v for k, v in i.items() if not k.startswith('_')} for i in datasets_table}

    print('Loading the MegaDB entries...')
    with open(args.megadb_entries) as f:
        sequences = json.load(f)
    print('Total number of sequences: {}'.format(len(sequences)))

    # print('Checking that the MegaDB entries conform to the schema...')
    # sequences_schema_check.sequences_schema_check(sequences)

    shuffle(sequences)
    visualize_sequences(datasets_table, sequences, args)


if __name__ == '__main__':
    main()

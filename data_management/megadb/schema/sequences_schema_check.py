import argparse
import json
import sys
import os
from collections import OrderedDict
import inspect

import jsonschema

"""
This script takes one argument, path to the JSON file containing the entries to be ingested 
into MegaDB in a JSON array. It then verifies it against the schema in schema.json in this directory.
It assumes that all entries are from the same dataset and checks that the `seq_id` field is unique.
"""

def order_seq_properties(seq_item):
    ordered = OrderedDict([
        ('dataset', seq_item['dataset']),
        ('seq_id', seq_item['seq_id'])])
    if 'location' in seq_item:
        ordered['location'] = seq_item['location']
    if 'images' in seq_item:
        ordered['images'] = seq_item['images']
    if 'class' in seq_item:
        ordered['class'] = seq_item['class']
    if 'datetime' in seq_item:
        ordered['datetime'] = seq_item['datetime']
    for name, val in seq_item.items():
        if name not in ordered:
            ordered[name] = val
    return ordered

def check_frame_num(seq):
    # schema already checks that the min possible value of frame_num is 0

    if 'images' not in seq:
        return

    # if there are more than one image item, each needs a frame_num
    if len(seq['images']) > 1:
        frame_num_set = []
        for i in seq['images']:
            if 'frame_num' not in i:
                assert False, 'sequence {} has more than one image but not all images have frame_num'.format(seq['seq_id'])

            frame_num_set.append(i['frame_num'])

        assert len(set(frame_num_set)) == len(seq['images']), 'sequence {} has frame_num that are not unique'.format(seq['seq_id'])


def check_class_on_seq_or_image(seq):
    """
    Checks if the 'class' property is on either the sequence or on each image.
    Sequences or images whose 'class' label is unavailable should be denoted by '__label_unavailable'
    Args:
        seq: a sequence object

    Raises:
        AssertionError
    """
    class_on_seq = False
    class_on_all_img = False

    if 'class' in seq:
        class_on_seq = True

    if 'images' in seq:
        class_on_all_img = True
        for image in seq['images']:
            if 'class' not in image:
                class_on_all_img = False

    assert class_on_seq or class_on_all_img, 'sequence {} does not have the class property on either sequence or image level'.format(seq['seq_id'])


def sequences_schema_check(items_json):
    assert len(items_json) > 0, 'The .json file you passed in is empty'

    # checks across all sequence items
    seq_ids = set([seq['seq_id'] for seq in items_json])
    assert len(seq_ids) == len(items_json), 'Not all seq_id in this batch are unique.'

    # per sequence item checks
    for seq in items_json:
        check_class_on_seq_or_image(seq)
        check_frame_num(seq)

    print('Verified that the sequence items meet requirements not captured by the schema.')

    # load the schema
    # https://stackoverflow.com/questions/3718657/how-to-properly-determine-current-script-directory
    this_script = inspect.getframeinfo(inspect.currentframe()).filename
    dir = os.path.dirname(this_script)
    with open(os.path.join(dir, 'sequences_schema.json')) as f:
        schema = json.load(f)

    jsonschema.validate(items_json, schema)

    print('Verified that the sequence items conform to the schema.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('items_json', action='store', type=str,
                        help='.json file to ingest into MegaDB')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    with open(args.items_json) as f:
        items_json = json.load(f)

    sequences_schema_check(items_json)


if __name__ == '__main__':
    main()
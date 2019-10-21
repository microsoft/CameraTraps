import argparse
import json
import sys
import os

import jsonschema

"""
This script takes one argument, path to the JSON file containing the entries to be ingested 
into MegaDB in a JSON array. It then verifies it against the schema in schema.json in this directory.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('items_json', action='store', type=str,
                        help='.json file to ingest into MegaDB')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    cur_dir = os.path.dirname(sys.argv[0])
    with open(os.path.join(cur_dir, 'schema.json')) as f:
        schema = json.load(f)

    with open(args.items_json) as f:
        instance = json.load(f)

    jsonschema.validate(instance, schema)

    print('Items verified.')


if __name__ == '__main__':
    main()
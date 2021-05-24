"""
A script for executing a query against the `sequences` table and store the results,
which are periodically checkpointed.

The environment variables COSMOS_ENDPOINT and COSMOS_KEY need to be set.
"""

import json
import os
import time
from datetime import datetime

import humanfriendly

from data_management.megadb.megadb_utils import MegadbUtils

#%% Common queries

# This query is used when preparing tfrecords for object detector training.
# We do not want to get the whole seq obj where at least one image has bbox because
# some images in that sequence will not be bbox labeled so will be confusing.
# Include images with bbox length 0 - these are confirmed empty by bbox annotators.
query_bbox = '''
SELECT im.bbox, im.file, seq.dataset, seq.location
FROM sequences seq JOIN im IN seq.images 
WHERE ARRAY_LENGTH(im.bbox) >= 0
'''

# For public datasets to be converted to the CCT format, we get the whole seq object because
# sequence level attributes need to be included too. megadb/converters/megadb_to_cct.py handles
# the case of bbox-only JSONs with the flag exclude_species_class
query_lila = '''
SELECT seq
FROM sequences seq
WHERE (SELECT VALUE COUNT(im) FROM im IN seq.images WHERE ARRAY_LENGTH(im.bbox) >= 0) > 0
'''

query_empty = '''
SELECT im.file, seq.dataset, seq.location
FROM sequences seq JOIN im IN seq.images 
WHERE ARRAY_LENGTH(im.class) = 1 AND ARRAY_CONTAINS(im.class, "empty") 
OR ARRAY_LENGTH(seq.class) = 1 AND ARRAY_CONTAINS(seq.class, "empty") 
'''

# Getting all sequences in a dataset - for updating or deleting entries which need the id field
query_dataset_all_entries = '''
SELECT seq
FROM sequences seq
'''

#%% Parameters

query = query_dataset_all_entries

output_dir = '/Users/siyuyang/Data/CameraTraps/iMerit12_MegaDB_to_upsert/wcs'
assert os.path.isdir(output_dir), 'Please create the output directory first'

output_indent = None  # None if no indentation needed in the output JSON, or int

# Use None if querying across all partitions
# The `sequences` table has the `dataset` as the partition key, so if only querying
# entries from one dataset, set the dataset name here.
partition_key='wcs'

# e.g. {'name': '@top_n', 'value': 100} - see query_and_upsert_examples/query_for_data.ipynb
query_parameters = None

save_every = 50000
assert save_every > 0

# Use False if do not want all results stored in a single JSON.
consolidate_results = True


#%% Script

time_stamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')

db_utils = MegadbUtils()  # read the CosmosDB endpoint and key from the environment

# execute the query
start_time = time.time()

result_iterable = db_utils.query_sequences_table(query=query,
                                                 partition_key=partition_key,
                                                 parameters=query_parameters)

# loop through and save the results
results = []
item_count = 0
part_count = 0
part_paths = []

for item in result_iterable:
    # MODIFY HERE depending on the query
    item_processed = {k: v for k, v in item['seq'].items() if not k.startswith('_')}

    results.append(item_processed)
    item_count += 1

    if item_count % save_every == 0:
        part_count += 1
        print(f'Saving results part {part_count}. Example item: {results[-1]}\n')

        part_path = os.path.join(output_dir, f'{time_stamp}_{item_count}.json')
        with open(part_path, 'w') as f:
            json.dump(results, f)
        results = []  # clear the results list
        part_paths.append(part_path)

elapsed = time.time() - start_time
print(f'Getting all the results used {humanfriendly.format_timespan(elapsed)}')

if consolidate_results:
    print('Consolidating the shards of query results...')

    all_results = results  # the unsaved entries
    for part_path in part_paths:
        with open(part_path) as f:
            part_items = json.load(f)
        all_results.extend(part_items)

    print(f'Number of items from iterable is {item_count}; '
          f'number of items in consolidated results is {len(all_results)}')
    out_path = os.path.join(output_dir, f'{time_stamp}_all.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=output_indent)
    print(f'Consolidated results saved to {out_path}')
    print('Removing partitions...')
    for part_path in part_paths:
        os.remove(part_path)

print('Done!')

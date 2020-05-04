"""
A script for executing a query and store the results.

The environment variables COSMOS_ENDPOINT and COSMOS_KEY need to be set.
"""

import json
import os
import time
from datetime import datetime

import humanfriendly
from azure.cosmos.cosmos_client import CosmosClient

#%% Common queries

# This query is used when preparing tfrecords for object detector training
# We do not want to get the whole seq obj where at least one image has bbox because some images in that sequence
# will not be bbox labeled so will be confusing
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

#%% Parameters

query = query_lila

output_dir = '/home/marmot/camtrap/data/wcs_boxes'
assert os.path.isdir(output_dir), 'Please create the output directory first'

output_indent = None  # None if no indentation needed in the output, or int

partition_key='wcs'  # use None if querying across all partitions

save_every = 20000
assert save_every > 0

# use False for when the results file will be too big to store in memory or in a single JSON.
consolidate_results = True


#%% Script

time_stamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')

# initialize Cosmos DB client
url = os.environ['COSMOS_ENDPOINT']  # TODO use megadb_utils instead
key = os.environ['COSMOS_KEY']
client = CosmosClient(url, credential=key)

database = client.get_database_client('camera-trap')
container_sequences = database.get_container_client('sequences')

# execute the query
start_time = time.time()

if partition_key:
    result_iterable = container_sequences.query_items(query=query,
                                                      partition_key=partition_key)
else:
    result_iterable = container_sequences.query_items(query=query,
                                                      enable_cross_partition_query=True)

# loop through and save the results
results = []
item_count = 0
part_count = 0
part_paths = []

for item in result_iterable:
    results.append(item['seq'])  # MODIFY HERE depending on the query
    item_count += 1

    if item_count % save_every == 0:
        part_count += 1
        print('Saving results part {}. Example item:'.format(part_count))
        print(results[-1])

        part_path = os.path.join(output_dir, '{}_{}.json'.format(time_stamp, item_count))
        with open(part_path, 'w') as f:
            json.dump(results, f)
        results = []  # clear the results list
        part_paths.append(part_path)

elapsed = time.time() - start_time
print('Getting all the results used {}'.format(humanfriendly.format_timespan(elapsed)))

if consolidate_results:
    print('Consolidating the parts...')

    all_results = results  # the unsaved entries
    for part_path in part_paths:
        with open(part_path) as f:
            part_items = json.load(f)
        all_results.extend(part_items)

    print('Number of items from iterable is {}; number of items in consolidated results is {}'.format(item_count,
                                                                                                      len(all_results)))
    out_path = os.path.join(output_dir, '{}_all.json'.format(time_stamp))
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=output_indent)
    print('Consolidated results saved to {}'.format(out_path))
    print('Removing partitions...')
    for part_path in part_paths:
        os.remove(part_path)

print('Done!')

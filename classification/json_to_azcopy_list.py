"""
Given a queried_images.json file output from json_validator.py, generates
one text file <dataset>_images.txt for every dataset included.

See: https://github.com/Azure/azure-storage-azcopy/wiki/Listing-specific-files-to-transfer
"""
import json
import os

from tqdm import tqdm

from data_management.megadb import megadb_utils
import sas_blob_utils


images_dir = ''
queried_images_json_path = 'run_idfg2/queried_images.json'
output_dir = 'run_idfg2/'

with open(queried_images_json_path, 'r') as f:
    js = json.load(f)

datasets_table = megadb_utils.MegadbUtils().get_datasets_table()

output_files = {}

pbar = tqdm(js.items())
for img_path, img_info in pbar:
    save_path = os.path.join(images_dir, img_path)
    if os.path.exists(save_path):
        continue

    ds, img_file = img_path.split('/', maxsplit=1)
    if ds not in output_files:
        output_path = os.path.join(output_dir, f'{ds}_images.txt')
        output_files[ds] = open(output_path, 'w')

        dataset_info = datasets_table[ds]
        account = dataset_info['storage_account']
        container = dataset_info['container']

        if 'public' in datasets_table[ds]['access']:
            url = sas_blob_utils.build_azure_storage_uri(
                account, container)
        else:
            url = sas_blob_utils.build_azure_storage_uri(
                account, container,
                sas_token=dataset_info['container_sas_key'][1:])
        pbar.write(f'"{url}"')

    output_files[ds].write(img_file + '\n')

for f in output_files.values():
    f.close()

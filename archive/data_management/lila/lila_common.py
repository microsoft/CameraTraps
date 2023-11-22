#
# Common constants and functions related to LILA data management/retrieval.
#

#%% Imports and constants

import os
import json
import zipfile
import pandas as pd

from urllib.parse import urlparse

# LILA camera trap master metadata file
lila_metadata_url = 'http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt'

lila_taxonomy_mapping_url = 'https://lila.science/wp-content/uploads/2022/07/lila-taxonomy-mapping_release.csv'

wildlife_insights_page_size = 30000
# wildlife_insights_taxonomy_url = 'https://api.wildlifeinsights.org/api/v1/taxonomy?fields=class,order,family,genus,species,authority,taxonomyType,uniqueIdentifier,commonNameEnglish&page[size]={}'.format(wildlife_insights_page_size)
wildlife_insights_taxonomy_url = 'https://api.wildlifeinsights.org/api/v1/taxonomy/taxonomies-all?fields=class,order,family,genus,species,authority,taxonomyType,uniqueIdentifier,commonNameEnglish&page[size]={}'.format(wildlife_insights_page_size)
wildlife_insights_taxonomy_local_json_filename = 'wi_taxonomy.json'
wildlife_insights_taxonomy_local_csv_filename = \
    wildlife_insights_taxonomy_local_json_filename.replace('.json','.csv')

# from ai4eutils
from url_utils import download_url
from path_utils import unzip_file


#%% Common functions
    
def read_wildlife_insights_taxonomy_mapping(metadata_dir):
    """
    Reads the WI taxonomy mapping file, downloading the .json data (and writing to .csv) if necessary.
    
    Returns a Pandas dataframe.
    """
    
    wi_taxonomy_csv_path = os.path.join(metadata_dir,wildlife_insights_taxonomy_local_csv_filename)
    
    if os.path.exists(wi_taxonomy_csv_path):
        df = pd.read_csv(wi_taxonomy_csv_path)
    else:
        wi_taxonomy_json_path = os.path.join(metadata_dir,wildlife_insights_taxonomy_local_json_filename)
        download_url(wildlife_insights_taxonomy_url, wi_taxonomy_json_path)
        with open(wi_taxonomy_json_path,'r') as f:
            d = json.load(f)
            
        # We haven't implemented paging, make sure that's not an issue
        assert d['meta']['totalItems'] < wildlife_insights_page_size
            
        # d['data'] is a list of items that look like:
        """
         {'id': 2000003,
         'class': 'Mammalia',
         'order': 'Rodentia',
         'family': 'Abrocomidae',
         'genus': 'Abrocoma',
         'species': 'bennettii',
         'authority': 'Waterhouse, 1837',
         'commonNameEnglish': "Bennett's Chinchilla Rat",
         'taxonomyType': 'biological',
         'uniqueIdentifier': '7a6c93a5-bdf7-4182-82f9-7a67d23f7fe1'}
        """
        df = pd.DataFrame(d['data'])
        df.to_csv(wi_taxonomy_csv_path,index=False)
        
    return df

    
def read_lila_taxonomy_mapping(metadata_dir):
    """
    Reads the LILA taxonomy mapping file, downloading the .csv file if necessary.
    
    Returns a Pandas dataframe.
    """
    
    p = urlparse(lila_taxonomy_mapping_url)
    taxonomy_filename = os.path.join(metadata_dir,os.path.basename(p.path))
    download_url(lila_taxonomy_mapping_url, taxonomy_filename)
    
    df = pd.read_csv(lila_taxonomy_mapping_url)
    
    return df

    
def read_lila_metadata(metadata_dir):
    """
    Reads LILA metadata (URLs to each dataset), downloading the txt file if necessary.
    
    Returns a dict mapping dataset names (e.g. "Caltech Camera Traps") to dicts
    with keys "sas_url" (pointing to the image base) and "json_url" (pointing to the metadata
    file).
    """
    
    # Put the master metadata file in the same folder where we're putting images
    p = urlparse(lila_metadata_url)
    metadata_filename = os.path.join(metadata_dir,os.path.basename(p.path))
    download_url(lila_metadata_url, metadata_filename)
    
    # Read lines from the master metadata file
    with open(metadata_filename,'r') as f:
        metadata_lines = f.readlines()
    metadata_lines = [s.strip() for s in metadata_lines]
    
    # Parse those lines into a table
    metadata_table = {}
    
    for s in metadata_lines:
        
        if len(s) == 0 or s[0] == '#':
            continue
        
        # Each line in this file is name/sas_url/json_url/[bbox_json_url]
        tokens = s.split(',')
        assert len(tokens) == 4
        ds_name = tokens[0].strip()
        url_mapping = {'sas_url':tokens[1],'json_url':tokens[2]}
        metadata_table[ds_name] = url_mapping
        
        # Create a separate entry for bounding boxes if they exist
        if len(tokens[3].strip()) > 0:
            print('Adding bounding box dataset for {}'.format(ds_name))
            bbox_url_mapping = {'sas_url':tokens[1],'json_url':tokens[3]}
            metadata_table[tokens[0]+'_bbox'] = bbox_url_mapping
            assert 'https' in bbox_url_mapping['json_url']
    
        assert 'https' not in tokens[0]
        assert 'https' in url_mapping['sas_url']
        assert 'https' in url_mapping['json_url']
    
    return metadata_table    
    

def get_json_file_for_dataset(ds_name,metadata_dir,metadata_table=None):
    """
    Downloads if necessary - then unzips if necessary - the .json file for a specific dataset.
    Returns the .json filename on the local disk.
    """
    if metadata_table is None:
        metadata_table = read_lila_metadata(metadata_dir)
        
    json_url = metadata_table[ds_name]['json_url']
    
    p = urlparse(json_url)
    json_filename = os.path.join(metadata_dir,os.path.basename(p.path))
    download_url(json_url, json_filename)
    
    # Unzip if necessary
    if json_filename.endswith('.zip'):
        
        with zipfile.ZipFile(json_filename,'r') as z:
            files = z.namelist()
        assert len(files) == 1
        unzipped_json_filename = os.path.join(metadata_dir,files[0])
        if not os.path.isfile(unzipped_json_filename):
            unzip_file(json_filename,metadata_dir)        
        else:
            print('{} already unzipped'.format(unzipped_json_filename))
        json_filename = unzipped_json_filename
    
    return json_filename

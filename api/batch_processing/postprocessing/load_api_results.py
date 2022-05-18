#
# load_api_results.py
#
# Loads the output of the batch processing API (json) into a pandas dataframe.
#
# Also functions to group entries by seq_id.
#
# Includes the deprecated functions that worked with the old CSV API output format.
#

#%% Constants and imports

from collections import defaultdict
import json
import os
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd

headers = ['image_path', 'max_confidence', 'detections']


#%% Functions for grouping by sequence_id

def ss_file_to_file_name(f):
    # example
    # input 'file': 'SER/S1/F08/F08_R3/S1_F08_R3_PICT1150.JPG'
    # output 'id': 'S1/F08/F08_R3/S1_F08_R3_PICT1150.JPG'
    return f.split('SER/')[1].split('.JPG')[0]


def caltech_file_to_file_name(f):
    return f.split('cct_images/')[1].split('.')[0]


def api_results_groupby(api_output_path, gt_db_indexed, file_to_image_id, field='seq_id'):
    """
    Given the output file of the API, groupby (currently only seq_id).

    Args:
        api_output_path: path to the API output json file
        gt_db_indexed: an instance of IndexedJsonDb so we know the seq_id to image_id mapping
        file_to_image_id: a function that takes in the 'file' field in 'images' in the detector
            output file and converts it to the 'id' field in the gt DB.
        field: which field in the 'images' array to group by

    Returns:
    A dict where the keys are of the field requested, each points to an array
    containing entries in the 'images' section of the output file
    """

    with open(api_output_path) as f:
        detection_results = json.load(f)

    res = defaultdict(list)
    for i in detection_results['images']:
        image_id = file_to_image_id(i['file'])
        field_val = gt_db_indexed.image_id_to_image[image_id][field]
        res[field_val].append(i)
    return res


#%% Functions for loading the result as a Pandas DataFrame

def load_api_results(api_output_path: str, normalize_paths: bool = True,
                     filename_replacements: Optional[Mapping[str, str]] = None
                     ) -> Tuple[pd.DataFrame, Dict]:
    """
    Loads the json formatted results from the batch processing API to a
    Pandas DataFrame, mainly useful for various postprocessing functions.

    Args:
        api_output_path: path to the API output json file
        normalize_paths: whether to apply os.path.normpath to the 'file' field
            in each image entry in the output file
        filename_replacements: replace some path tokens to match local paths to
            the original blob structure

    Returns:
        detection_results: pd.DataFrame, contains at least the columns:
                ['file', 'max_detection_conf', 'detections','failure']            
        other_fields: a dict containing fields in the dict
    """
    print('Loading API results from {}'.format(api_output_path))

    with open(api_output_path) as f:
        detection_results = json.load(f)

    print('De-serializing API results')

    # Validate that this is really a detector output file
    for s in ['info', 'detection_categories', 'images']:
        assert s in detection_results, 'Missing field {} in detection results'.format(s)

    # Fields in the API output json other than 'images'
    other_fields = {}
    for k, v in detection_results.items():
        if k != 'images':
            other_fields[k] = v

    # Normalize paths to simplify comparisons later
    if normalize_paths:
        for image in detection_results['images']:
            image['file'] = os.path.normpath(image['file'])
            # image['file'] = image['file'].replace('\\','/')

    # Pack the json output into a Pandas DataFrame
    detection_results = pd.DataFrame(detection_results['images'])

    # Replace some path tokens to match local paths to original blob structure
    # string_to_replace = list(filename_replacements.keys())[0]
    if filename_replacements is not None:
        for string_to_replace in filename_replacements:

            replacement_string = filename_replacements[string_to_replace]

            for i_row in range(len(detection_results)):
                row = detection_results.iloc[i_row]
                fn = row['file']
                fn = fn.replace(string_to_replace, replacement_string)
                detection_results.at[i_row, 'file'] = fn

    print('Finished loading and de-serializing API results for {} images from {}'.format(
            len(detection_results),api_output_path))

    return detection_results, other_fields


def write_api_results(detection_results_table, other_fields, out_path):
    """
    Writes a Pandas DataFrame back to a json that is compatible with the API output format.
    """

    print('Writing detection results to {}'.format(out_path))

    fields = other_fields

    images = detection_results_table.to_json(orient='records',
                                             double_precision=3)
    images = json.loads(images)
    fields['images'] = images

    with open(out_path, 'w') as f:
        json.dump(fields, f, indent=1)

    print('Finished writing detection results to {}'.format(out_path))


def load_api_results_csv(filename, normalize_paths=True, filename_replacements={}, nrows=None):
    """
    DEPRECATED
    Loads .csv-formatted results from the batch processing API to a pandas table
    """

    print('Loading API results from {}'.format(filename))

    detection_results = pd.read_csv(filename,nrows=nrows)

    print('De-serializing API results from {}'.format(filename))

    # Sanity-check that this is really a detector output file
    for s in ['image_path','max_confidence','detections']:
        assert s in detection_results.columns

    # Normalize paths to simplify comparisons later
    if normalize_paths:
        detection_results['image_path'] = detection_results['image_path'].apply(os.path.normpath)

    # De-serialize detections
    detection_results['detections'] = detection_results['detections'].apply(json.loads)

    # Optionally replace some path tokens to match local paths to the original blob structure
    # string_to_replace = list(options.detector_output_filename_replacements.keys())[0]
    for string_to_replace in filename_replacements:

        replacement_string = filename_replacements[string_to_replace]

        # TODO: hit some silly issues with vectorized str() and escaped characters, vectorize
        # this later.
        #
        # detection_results['image_path'].str.replace(string_to_replace,replacement_string)
        # iRow = 0
        for iRow in range(0,len(detection_results)):
            row = detection_results.iloc[iRow]
            fn = row['image_path']
            fn = fn.replace(string_to_replace,replacement_string)
            detection_results.at[iRow,'image_path'] = fn

    print('Finished loading and de-serializing API results for {} images from {}'.format(len(detection_results),filename))

    return detection_results


def write_api_results_csv(detection_results, filename):
    """
    DEPRECATED
    Writes a pandas table to csv in a way that's compatible with the .csv API output
    format.  Currently just a wrapper around to_csv that just forces output writing
    to go through a common code path.
    """

    print('Writing detection results to {}'.format(filename))

    detection_results.to_csv(filename, index=False)

    print('Finished writing detection results to {}'.format(filename))

#
# load_api_results.py
#
# Loads the output of the batch processing API (json).
#
# Includes the deprecated functions that worked with the old CSV API output format.
#

#%% Constants and imports

import pandas as pd
import json
import os

headers = ['image_path', 'max_confidence', 'detections']


#%% Functions

def load_api_results(api_output_filename, normalize_paths=True, filename_replacements={}):
    """
    Loads the json formatted results from the batch processing API to a Pandas DataFrame, mainly useful for
    various postprocessing functions.

    Args:
        api_output_filename: path to the API output json file
        normalize_paths: whether to apply os.path.normpath to the 'file' field in each image entry in the output file
        filename_replacements: replace some path tokens to match local paths to the original blob structure

    Returns:
        detection_results: a Pandas DataFrame with columns (file, max_detection_conf, detections)
            which correspond to the old column names (image_path, max_confidence, and detections). It may also include
            a 'meta' column.
        other_fields: a dict containing fields in the dict
    """
    print('Loading API results from {}'.format(api_output_filename))

    with open(api_output_filename) as f:
        detection_results = json.load(f)

    print('De-serializing API results from {}'.format(api_output_filename))

    # Sanity-check that this is really a detector output file
    for s in ['info', 'detection_categories', 'images']:
        assert s in detection_results

    other_fields = {}  # fields in the API output json other than 'images'
    for k, v in detection_results.items():
        if k != 'images':
            other_fields[k] = v

    # Normalize paths to simplify comparisons later
    if normalize_paths:
        for image in detection_results['images']:
            image['file'] = os.path.normpath(image['file'])

    # Pack the json output into a Pandas DataFrame
    detection_results = pd.DataFrame(detection_results['images'])

    # Optionally replace some path tokens to match local paths to the original blob structure
    # string_to_replace = list(options.detector_output_filename_replacements.keys())[0]
    for string_to_replace in filename_replacements:

        replacement_string = filename_replacements[string_to_replace]

        # TODO: hit some silly issues with vectorized str() and escaped characters, vectorize
        # this later.
        #
        # detection_results['image_path'].str.replace(string_to_replace,replacement_string)
        # i_row = 0
        for i_row in range(0, len(detection_results)):
            row = detection_results.iloc[i_row]
            fn = row['file']
            fn = fn.replace(string_to_replace, replacement_string)
            detection_results.at[i_row, 'file'] = fn

    print('Finished loading and de-serializing API results for {} images from {}'.format(len(detection_results),
                                                                                         api_output_filename))

    return detection_results, other_fields


def write_api_results(detection_results_table, other_fields, out_path):
    """
    Writes a Pandas DataFrame back to a json that is compatible with the API output format.
    """
    print('Writing detection results to {}'.format(out_path))

    fields = other_fields

    images = detection_results_table.to_json(orient='records',
                                             double_precision=3)  # TODO read double_precision from a config elsewhere
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

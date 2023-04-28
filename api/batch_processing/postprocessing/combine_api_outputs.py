"""
Merges two or more .json files in batch API output format, optionally
writing the results to another .json file.
- Concatenates image lists, erroring if images are not unique.
- Errors if class lists are conflicting; errors on unrecognized fields.
- Checks compatibility in info structs, within reason.

File format:

https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format

Command-line use:

combine_api_outputs input1.json input2.json ... inputN.json output.json

Also see combine_api_shard_files() (not exposed via the command line yet) to
combine the intermediate files created by the API.

This does no checking for redundancy; if you are looking to ensemble
the results of multiple model versions, see merge_detections.py.
"""

#%% Constants and imports

import argparse
import json
from typing import Any, Dict, Iterable, Mapping, List, Optional

#%% Merge functions

def combine_api_output_files(input_files: List[str],
                             output_file: Optional[str] = None,
                             require_uniqueness: bool = True
                             ) -> Dict[str, Any]:
    """Merges list of JSON API detection files *input_files* into a single
    dictionary, optionally writing the result to *output_file*.

    Args:
        input_files: list of str, paths to JSON detection files
        output_file: optional str, path to write merged JSON
        require_uniqueness: bool, whether to require that the images in
            each input_dict be unique
    """
    input_dicts = []
    print('Loading input files')
    for fn in input_files:
        with open(fn, 'r', encoding='utf-8') as f:
            input_dicts.append(json.load(f))

    print('Merging results')
    merged_dict = combine_api_output_dictionaries(
        input_dicts, require_uniqueness=require_uniqueness)

    print('Writing output')
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(merged_dict, f, indent=1)

    return merged_dict


def combine_api_output_dictionaries(input_dicts: Iterable[Mapping[str, Any]],
                                    require_uniqueness: bool = True
                                    ) -> Dict[str, Any]:
    """Merges the list of API detection dictionaries *input_dicts*.  See header
    comment for details on merge rules.

    Args:
        input_dicts: list of dicts, each dict is the JSON of the detections
            output file from the Batch Processing API
        require_uniqueness: bool, whether to require that the images in
            each input_dict be unique

    Returns: dict, represents the merged JSON
    """
    # Map image filenames to detections, we'll convert to a list later
    images = {}
    info: Dict[str, str] = {}
    detection_categories: Dict[str, str] = {}
    classification_categories: Dict[str, str] = {}
    n_redundant_images = 0
    n_images = 0

    known_fields = ['info', 'detection_categories', 'classification_categories',
                    'images']

    for input_dict in input_dicts:

        for k in input_dict:
            if k not in known_fields:
                raise ValueError(f'Unrecognized API output field: {k}')

        # Check compatibility of detection categories
        for cat_id in input_dict['detection_categories']:
            cat_name = input_dict['detection_categories'][cat_id]
            if cat_id in detection_categories:
                assert detection_categories[cat_id] == cat_name, (
                    'Detection category mismatch')
            else:
                detection_categories[cat_id] = cat_name

        # Check compatibility of classification categories
        if 'classification_categories' in input_dict:
            for cat_id in input_dict['classification_categories']:
                cat_name = input_dict['classification_categories'][cat_id]
                if cat_id in classification_categories:
                    assert classification_categories[cat_id] == cat_name, (
                        'Classification category mismatch')
                else:
                    classification_categories[cat_id] = cat_name

        # Merge image lists, checking uniqueness
        for im in input_dict['images']:
            im_file = im['file']
            if require_uniqueness:
                assert im_file not in images, f'Duplicate image: {im_file}'
                images[im_file] = im
                n_images += 1
            else:
                if im_file in images:
                    n_redundant_images += 1
                    previous_im = images[im_file]
                    # Replace a previous failure with a success
                    if ('detections' in im) and ('detections' not in previous_im):
                        images[im_file] = im
                        print(f'Replacing previous failure for image: {im_file}')
                else:
                    images[im_file] = im
                    n_images += 1

        # Merge info dicts, don't check completion time fields
        if len(info) == 0:
            info = input_dict['info']
        else:
            info_compare = input_dict['info']
            assert info_compare['detector'] == info['detector'], (
                'Incompatible detection versions in merging')
            assert info_compare['format_version'] == info['format_version'], (
                'Incompatible API output versions in merging')
            if 'classifier' in info_compare:
                if 'classifier' in info:
                    assert info['classifier'] == info_compare['classifier']
                else:
                    info['classifier'] = info_compare['classifier']

    # ...for each dictionary

    if n_redundant_images > 0:
        print(f'Warning: found {n_redundant_images} redundant images '
              f'(out of {n_images} total) during merge')

    # Convert merged image dictionaries to a sorted list
    sorted_images = sorted(images.values(), key=lambda im: im['file'])

    merged_dict = {'info': info,
                   'detection_categories': detection_categories,
                   'classification_categories': classification_categories,
                   'images': sorted_images}
    return merged_dict


def combine_api_shard_files(input_files, output_file=None):
    """
    Merges the list of .json-formatted API shard files *input_files* into a single
    list of dictionaries, optionally writing the result to *output_file*.
    """

    input_lists = []
    print('Loading input files')
    for fn in input_files:
        input_lists.append(json.load(open(fn)))

    detections = []
    # detection_list = input_lists[0]
    for detection_list in input_lists:
        assert isinstance(detection_list, list)
        # d = detection_list[0]
        for d in detection_list:
            assert 'file' in d
            assert 'max_detection_conf' in d
            assert 'detections' in d
            detections.extend([d])

    print('Writing output')
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(detections, f, indent=1)

    return detections


#%% Driver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_paths', nargs='+',
        help='List of input .json files')
    parser.add_argument(
        'output_path',
        help='Output .json file')
    args = parser.parse_args()
    combine_api_output_files(args.input_paths, args.output_path)


if __name__ == '__main__':
    main()

####
#
# combine_api_outputs.py
#
# Merges two or more .json files in batch API output format, optionally
# writing the results to another .json file.  Concatenates image lists,
# erroring if images are not unique.  Errors if class lists are conflicting, errors 
# on unrecognized fields.  Checks compatibility in info structs, within reason.
#
# File format:
#
# https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
#
# Command-line use:
#
# combine_api_outputs input1.json input2.json ... inputN.json output.json
#
####

#%% Constants and imports

import json
import argparse


#%% Merge functions

def combine_api_output_files(input_files,output_file=None):
    """
    Merges the list of .json-formatted API output files *input_files* into a single
    dictionary, optionally writing the result to *output_file*.
    """
    
    input_dicts = []
    for fn in input_files:
        input_dicts.append(json.load(open(fn)))
        
    merged_dict = combine_api_output_dictionaries(input_dicts)
    
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(merged_dict, f, indent=1)

    return merged_dict


def combine_api_output_dictionaries(input_dicts):
    """
    Merges the list of API output dictionaries *input_dicts*.  See header comment
    for details on merge rules.
    """
    
    # Map image filenames to detections, we'll convert to a list later
    images = {}
    info = {}
    detection_categories = {}
    classification_categories = {}    
    
    for input_dict in input_dicts:
        
        known_fields = ['info','detection_categories','classification_categories','images']
        for k in input_dict:
            if k not in known_fields:
                raise ValueError('Unrecognized API output field in merging: {}'.format(k))
                
        # Check compatibility of detection categories
        for cat_id in input_dict['detection_categories']:
            cat_name = input_dict['detection_categories'][cat_id]
            if cat_id in detection_categories:
                assert detection_categories[cat_id] == cat_name, 'Detection category mismatch'
            else:
                detection_categories[cat_id] = cat_name
        
        # Check compatibility of classification categories
        if 'classification_categories' in input_dict:
            for cat_id in input_dict['classification_categories']:
                cat_name = input_dict['classification_categories'][cat_id]
                if cat_id in classification_categories:
                    assert classification_categories[cat_id] == cat_name, 'Classification category mismatch'
                else:
                    classification_categories[cat_id] = cat_name
        
        # Merge image lists, checking uniqueness
        for im in input_dict['images']:
            assert im['file'] not in images, 'Duplicate image: {}'.format(im['file'])
            images[im['file']] = im
        
        # Merge info dicts, within reason
        if len(info) == 0:
            info = input_dict['info']
        else:
            info_compare = input_dict['info']
            assert info_compare['detector'] == info['detector'], 'Incompatible detection versions in merging'
            assert info_compare['format_version'] == info['format_version'], 'Incompatible API output versions in merging'
            if 'classifier' in info_compare:
                if 'classifier' in info:
                    assert info['classifier'] == info_compare['classifier']
                else:
                    info['classifier'] = info_compare['classifier']
            # Don't check completion time fields
                    
    # ...for each dictionary

    # Convert merged image dictionaries to a sorted list
    sorted_images = sorted(images.values(), key = lambda im: im['file']) 
    
    merged_dict = {'info':info,'detection_categories':detection_categories,
                   'classification_categories':classification_categories,
                   'images':sorted_images}    
        
    return merged_dict
    

#%% Driver
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_paths', nargs='+', help='List of input .json files')
    parser.add_argument('output_path', help='Output .json file')
    args = parser.parse_args()
    combine_api_output_files(args.input_paths,args.output_path)
    
if __name__ == '__main__':
    main()
    
    

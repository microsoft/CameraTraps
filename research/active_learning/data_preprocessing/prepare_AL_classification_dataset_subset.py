'''
prepare_AL_classification_dataset_subset.py

Given a full dataset for active learning for species classification, this script creates a random subset of the dataset.
(This is convenient for testing the labeling tool, etc.)

Prerequisites:
- A directory with cropped images for a dataset
- A crops.json containing information about each cropped image in the crops directory
- A directory with full-size images for a dataset

Produces:
- A directory with cropped images for the subset dataset
- A crops.json containing information about only cropped images in the subset dataset
- A directory with full-size images corresponding to cropped images in the subset dataset

'''

import argparse, copy, json, os, pickle, random, sys, time
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', default=1000, type=int, help='Number of samples to draw from the full dataset for the subset dataset.')
    parser.add_argument('--crop_dir', type=str, help='Path to directory with the cropped images of the full dataset.')
    parser.add_argument('--crop_json', type=str, help='Path to .json with information about cropped images in the full dataset.')
    parser.add_argument('--image_dir', type=str, help='Path to directory with the full-size source images of the full dataset.')
    parser.add_argument('--old_base_dir', type=str, help='Path specifying base directory for crop and image subdirs, to be stripped from filenames in crops.json.')
    parser.add_argument('--new_base_dir', type=str, help='Path specifying base directory for crop and image subdirs, to be added to filenames in crops.json.')
    # parser.add_argument('--db_name', default='missouricameratraps', type=str, help='Name of the training (target) data Postgres DB.')
    # parser.add_argument('--db_user', default='user', type=str, help='Name of the user accessing the Postgres DB.')
    # parser.add_argument('--db_password', default='password', type=str, help='Password of the user accessing the Postgres DB.')
    # parser.add_argument('--base_model', type=str, help='Path to latest embedding model checkpoint.')
    # parser.add_argument('--output_dir', type=str, help='Output directory for subset of crops')
    parser.add_argument('--random_seed', default=1234, type=int, help='Random seed to get same samples from dataset.')
    args = parser.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    verbose = True

    if verbose:
        print('Reading crops from: \t\t\t%s'%args.crop_dir)
        print('Reading full-size images from: \t\t%s'%args.image_dir)
        print('Writing subset crops to: \t\t%s'%args.crop_dir.replace(args.old_base_dir, args.new_base_dir))
        print('Writing subset full-size images to: \t%s'%args.image_dir.replace(args.old_base_dir, args.new_base_dir))
        
        
    crop_data = json.load(open(args.crop_json, 'r'))
    subset_indices = np.random.permutation(range(len(crop_data)))[:args.num]
    subset_crops = [sorted(list(crop_data.keys()))[sidx] for sidx in subset_indices]
    
    subset_crop_data = {}
    for k in subset_crops:
        # Add json entry for this crop
        k_data = copy.copy(crop_data[k])
        k_data['file_name'] = k_data['file_name'].replace(args.old_base_dir, args.new_base_dir) # correct base directory from old (full dataset) to new (subset dataset)
        k_data['source_file_name'] = k_data['source_file_name'].replace(args.old_base_dir, args.new_base_dir)
        subset_crop_data[k] = k_data

        # Copy file for this crop to subset dataset crop dir
        sh_command = 'mkdir -p %s && cp %s $_'%(os.path.dirname(k_data['file_name']), crop_data[k]['file_name'])
        os.popen(sh_command)

        # Copy file for its full-size source image to subset dataset image dir
        sh_command = 'mkdir -p %s && cp %s $_'%(os.path.dirname(k_data['source_file_name']), crop_data[k]['source_file_name'])
        os.popen(sh_command)
    
    # Write crops.json to subset dataset crop dir
    subset_crop_json = os.path.join(os.path.dirname(k_data['file_name']), 'crops.json')
    with open(subset_crop_json, 'w') as f:
        json.dump(subset_crop_data, f)

if __name__ == '__main__':
    main()
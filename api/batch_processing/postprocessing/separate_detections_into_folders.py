#
# separate_detections_into_folders.py
#
# Given a .json file with batch processing results, separate the files in that
# set of results into folders that contain animals/people/vehicles/nothing, 
# according to per-class thresholds.
#
# Image files are copied, not moved.
#
# Preserves relative paths within each of those folders; cannot be used with .json
# files that have absolute paths in them.
#
# For example, if your .json file has these images:
#
# a/b/c/1.jpg
# a/b/d/2.jpg
# a/b/e/3.jpg
# a/b/f/4.jpg
#
# And let's say:
#
# * The results say that the first three images are empty/person/vehicle, respectively
# * The fourth image is above threshold for "animal" and "person"
# * You specify an output base folder of c:\out
#
# You will get the following files:
#
# c:\out\empty\a\b\c\1.jpg
# c:\out\people\a\b\d\2.jpg
# c:\out\vehicles\a\b\e\3.jpg
# c:\out\animal_person\a\b\f\4.jpg
#
# Hard-coded to work with MDv3 and MDv4 output files.  Not currently future-proofed
# past the classes in MegaDetector v4, not currently ready for species-level classification.  
#


#%% Constants and imports

import argparse
import json
import os
import shutil
import sys
import itertools

from multiprocessing.pool import ThreadPool
from functools import partial
        
from tqdm import tqdm

from ct_utils import args_to_object

friendly_folder_names = {'animal':'animals','person':'people','vehicle':'vehicles'}

# Occasionally we have near-zero confidence detections associated with COCO classes that
# didn't quite get squeezed out of the model in training.  As long as they're near zero
# confidence, we just ignore them.
invalid_category_epsilon = 0.00001


#%% Options class

default_threshold = 0.725
            
class SeparateDetectionsIntoFoldersOptions:

    def __init__(self,threshold=default_threshold):
        
        self.category_name_to_threshold = {
            'animal': threshold,
            'person': threshold,
            'vehicle': threshold
        }
        
        self.n_threads = 1
        
        self.allow_existing_directory = False        
        self.allow_missing_files = False
        self.overwrite = True
        
        self.results_file = None
        self.base_input_folder = None
        self.base_output_folder = None
                  
        # Dictionary mapping categories (plus combinations of categories, and 'empty') to output folders
        self.category_name_to_folder = None
        self.category_id_to_category_name = None
        self.debug_max_images = None        
        
    
#%% Support functions
    
def path_is_abs(p): return (len(p) > 1) and (p[0] == '/' or p[1] == ':')

printed_missing_file_warning = False
    
def process_detection(d,options):
    """
    Process detections for a single image
    """

    global printed_missing_file_warning
    
    relative_filename = d['file']
    
    detections = None    
    if 'detections' in d:
        detections = d['detections']
    
    if detections is None:
        
        assert d['failure'] is not None and len(d['failure']) > 0
        target_folder = options.category_name_to_folder['failure']
    
    else:
        
        category_name_to_max_confidence = {}
        category_names = options.category_id_to_category_name.values()
        for category_name in category_names:
            category_name_to_max_confidence[category_name] = 0.0
        
        # Find the maximum confidence for each category
        #
        # det = detections[0]
        for det in detections:
            
            category_id = det['category']
            
            # For zero-confidence detections, we occasionally have leftover goop
            # from COCO classes
            if category_id not in options.category_id_to_category_name:
                print('Warning: unrecognized category {} in file {}'.format(
                    category_id,relative_filename))
                # assert det['conf'] < invalid_category_epsilon
                continue
                
            category_name = options.category_id_to_category_name[category_id]
            if det['conf'] > category_name_to_max_confidence[category_name]:
                category_name_to_max_confidence[category_name] = det['conf']
        
        # Count the number of thresholds exceeded
        categories_above_threshold = []
        for category_name in category_names:
            
            threshold = default_threshold
            
            # Do we have a custom threshold for this category?
            if category_name in options.category_name_to_threshold:
                threshold = options.category_name_to_threshold[category_name]
                
            max_confidence_this_category = category_name_to_max_confidence[category_name]
            if max_confidence_this_category > threshold:
                categories_above_threshold.append(category_name)
        
        categories_above_threshold.sort()
        
        # If this is above multiple thresholds
        if len(categories_above_threshold) > 1:
            target_folder = options.category_name_to_folder['_'.join(categories_above_threshold)]
    
        elif len(categories_above_threshold) == 0:
            target_folder = options.category_name_to_folder['empty']
            
        else:
            target_folder = options.category_name_to_folder[categories_above_threshold[0]]
        
    # if this is/isn't a failure case
            
    source_path = os.path.join(options.base_input_folder,relative_filename)
    if not os.path.isfile(source_path):
        if not options.allow_missing_files:
            raise ValueError('Cannot find file {}'.format(source_path))
        else:
            if not printed_missing_file_warning:
                print('Warning: cannot find at least one file ({})'.format(source_path))    
                printed_missing_file_warning = True
            return
            
    target_path = os.path.join(target_folder,relative_filename)
    if (not options.overwrite) and (os.path.isfile(target_path)):
        return
    
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir,exist_ok=True)
    shutil.copyfile(source_path,target_path)
    
# ...def process_detection()
    
    
#%% Main function

def separate_detections_into_folders(options):

    # Create output folder if necessary
    if (os.path.isdir(options.base_output_folder)) and \
        (len(os.listdir(options.base_output_folder) ) > 0):
        if options.allow_existing_directory:
            print('Warning: target folder exists and is not empty... did you mean to delete an old version?')
        else:
            raise ValueError('Target folder exists and is not empty')
    os.makedirs(options.base_output_folder,exist_ok=True)    
    
    # Load detection results    
    print('Loading detection results')
    results = json.load(open(options.results_file))
    detections = results['images']
    
    for d in detections:
        fn = d['file']
        assert not path_is_abs(fn), 'Cannot process results with absolute image paths'
        
    print('Processing {} detections'.format(len(detections)))
    
    detection_categories = results['detection_categories']    
    options.category_id_to_category_name = detection_categories
    
    # Map class names to output folders
    options.category_name_to_folder = {}
    options.category_name_to_folder['empty'] = os.path.join(options.base_output_folder,'empty')
    options.category_name_to_folder['failure'] = os.path.join(options.base_output_folder,'processing_failure')
    
    # Create all combinations of categories
    category_names = list(detection_categories.values())
    category_names.sort()
    
    for category_name in category_names:        

        threshold = default_threshold
        
        # Do we have a custom threshold for this category?
        if category_name in options.category_name_to_threshold:
            threshold = options.category_name_to_threshold[category_name]
        print('Processing category {} at threshold {}'.format(category_name,threshold))
            
    target_category_names = []
    for c in category_names:
        target_category_names.append(c)
    
    for combination_length in range(2,len(category_names)+1):
        combined_category_names = list(itertools.combinations(category_names,combination_length))
        for combination in combined_category_names:
            
            combined_name = '_'.join(combination)
            target_category_names.append(combined_name)
    
    # Create folder mappings for each category
    for category_name in target_category_names:
        
        folder_name = category_name
        if category_name in friendly_folder_names:
            folder_name = friendly_folder_names[category_name]
        options.category_name_to_folder[category_name] = \
            os.path.join(options.base_output_folder,folder_name)
    
    # Create the actual folders
    for folder in options.category_name_to_folder.values():
        os.makedirs(folder,exist_ok=True)            
        
    if options.n_threads <= 1 or options.debug_max_images is not None:
    
        for i_detection,d in enumerate(tqdm(detections)):
            if options.debug_max_images is not None and i_detection > options.debug_max_images:
                break
            process_detection(d,options)
        
    else:
        
        print('Starting a pool with {} threads'.format(options.n_threads))
        pool = ThreadPool(options.n_threads)
        process_detection_with_options = partial(process_detection, options=options)
        results = list(tqdm(pool.imap(process_detection_with_options, detections), total=len(detections)))
        
        
#%% Interactive driver
        
if False:

    pass

    #%%
    
    default_threshold = 0.8
    options = SeparateDetectionsIntoFoldersOptions(default_threshold)
    
    options.results_file = r"G:\x\x-20200407\combined_api_outputs\x-20200407_detections.filtered_rde_0.60_0.85_5_0.05.json"
    options.base_input_folder = "z:\\"
    options.base_output_folder = r"E:\x-out"
    options.n_threads = 100
    options.allow_existing_directory = False
    
    #%%
    
    separate_detections_into_folders(options)
    
    
    #%% Find a particular file
    
    results = json.load(open(options.results_file))
    detections = results['images']    
    filenames = [d['file'] for d in detections]
    i_image = filenames.index('for_Azure\HL0913\RCNX1896.JPG')
    
    
#%% Command-line driver   

# python api\batch_processing\postprocessing\separate_detections_into_folders.py "d:\temp\rspb_mini.json" "d:\temp\demo_images\rspb_2018_2019_mini" "d:\temp\separation_test" --nthreads 2


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', type=str, help='Input .json filename')
    parser.add_argument('base_input_folder', type=str, help='Input image folder')
    parser.add_argument('base_output_folder', type=str, help='Output image folder')

    parser.add_argument('--animal_threshold', type=float, default=default_threshold,
                        help='Confidence threshold for the animal category (default={})'.format(default_threshold))
    parser.add_argument('--human_threshold', type=float, default=default_threshold,
                        help='Confidence threshold for the human category (default={})'.format(default_threshold))
    parser.add_argument('--vehicle_threshold', type=float, default=default_threshold,
                        help='Confidence threshold for vehicle category (default={})'.format(default_threshold))
    parser.add_argument('--n_threads', type=int, default=1,
                        help='Number of threads to use for parallel operation (default=1)')
    parser.add_argument('--allow_existing_directory', action='store_true', 
                        help='Proceed even if the target directory exists and is not empty')
    parser.add_argument('--no_overwrite', action='store_true', 
                        help='Skip images that already exist in the target folder, must also specify --allow_existing_directory')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()    
    
    # Convert to an options object
    options = SeparateDetectionsIntoFoldersOptions()
    
    args_to_object(args, options)

    if args.animal_threshold:
        options.category_name_to_threshold['animal'] = args.animal_threshold

    if args.human_threshold:
        options.category_name_to_threshold['person'] = args.human_threshold

    if args.vehicle_threshold:
        options.category_name_to_threshold['vehicle'] = args.vehicle_threshold
    
    options.overwrite = (not args.no_overwrite)
        
    separate_detections_into_folders(options)
    
if __name__ == '__main__':
    
    main()
    
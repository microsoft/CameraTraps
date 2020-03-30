#
# separate_detections_into_folders.py
#
# Given a .json file with batch processing results, separate the files in that
# set of results into folders that contain animals/people/vehicles/nothing, 
# according to per-class thresholds.
#
# Places images that are above threshold for multiple classes into 'multiple'
# folder.
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
# c:\out\multiple\a\b\f\4.jpg
#
# Hard-coded to work with MDv3 and MDv4 output files.  Not currently future-proofed
# past the classes in MegaDetector v4, not currently ready for species-level classification.  
#

#%% Constants and imports

import json
import os
import shutil
from multiprocessing.pool import ThreadPool

from tqdm import tqdm

output_folders = ['empty','animals','people','vehicles','multiple']


#%% Options class

class SeparateDetectionsIntoFoldersOptions:
    
    # Inputs
    animal_threshold = 0.725
    human_threshold = 0.725
    vehicle_threshold = 0.725
    n_threads = 1
    
    allow_existing_directory = False
    
    results_file = None
    base_input_folder = None
    base_output_folder = None
    
    # Populated later
    animal_category = -1
    person_category = -1
    vehicle_category = -1
    target_folders = None
    
    
#%% Function used to process each image
    
def process_detection(d,options):

    relative_filename = d['file']
    detections = d['detections']
    
    max_animal_confidence = -1
    max_person_confidence = -1
    max_vehicle_confidence = -1
    
    # det = detections[0]
    for det in detections:
        assert det['category'] == options.animal_category or \
          det['category'] == options.person_category or \
          det['category'] == options.vehicle_category
          
        if det['category'] == options.animal_category:
            max_animal_confidence = max([det['conf'],max_animal_confidence])
        elif det['category'] == options.person_category:
            max_person_confidence = max([det['conf'],max_person_confidence])
        elif det['category'] == options.vehicle_category:
            max_vehicle_confidence = max([det['conf'],max_vehicle_confidence])
        else:
            raise ValueError('Unrecognized detection category')
    
    target_folder = ''
    
    n_thresholds = 0
    if (max_person_confidence >= options.human_threshold):
        n_thresholds += 1
    if (max_animal_confidence >= options.animal_threshold):
        n_thresholds += 1
    if (max_vehicle_confidence >= options.vehicle_threshold):
        n_thresholds += 1
    
    # If this is above multiple thresholds
    if n_thresholds > 1:
        target_folder = options.target_folders['multiple']

    # Else if this is above threshold for people...
    elif (max_person_confidence >= options.human_threshold):
        target_folder = options.target_folders['people']
        
    # Else if this is above threshold for animals...
    elif (max_animal_confidence >= options.animal_threshold):
        target_folder = options.target_folders['animals']
    
    # Else if this is above threshold for vechicles...
    elif (max_vehicle_confidence >= options.vehicle_threshold):
        target_folder = options.target_folders['vehicles']
    
    # Else this is empty
    else:
        target_folder = options.target_folders['empty']
            
    source_path = os.path.join(options.base_input_folder,relative_filename)
    assert os.path.isfile(source_path), 'Cannot find file {}'.format(source_path)
    
    target_path = os.path.join(target_folder,relative_filename)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir,exist_ok=True)
    shutil.copyfile(source_path,target_path)
    
# ...def process_detection()
    
    
#%% Main function
    
def path_is_abs(p): return (len(p) > 1) and (p[0] == '/' or p[1] == ':')
    
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
    results = json.load(open(options.results_file))
    detections = results['images']
    
    for d in detections:
        fn = d['file']
        assert not path_is_abs(fn), 'Cannot process results with absolute image paths'
        
    print('Processing {} detections'.format(len(detections)))
    
    detection_categories = results['detection_categories']
    category_mappings = {value: key for key, value in detection_categories.items()}
    options.animal_category = category_mappings['animal']
    options.person_category = category_mappings['person']
    
    if 'vehicle' in category_mappings:
        options.vehicle_category = category_mappings['vehicle']
    else:
        options.vehicle_category = -1

    # Separate into folders
    options.target_folders = {}
    
    for f in output_folders:
        options.target_folders[f] = os.path.join(options.base_output_folder,f)
        os.makedirs(options.target_folders[f],exist_ok=True)            
        
    if options.n_threads <= 1:
    
        # i_image = 0; d = detections_to_process[i_image]
        for d in tqdm(detections):
            process_detection(d,options)
        
    else:
        
        pool = ThreadPool(options.n_threads)        
        results = list(tqdm(pool.imap(process_detection, detections), total=len(detections)))
        
        
#%% Interactive driver
        
if False:

    pass

    #%%
    
    options = SeparateDetectionsIntoFoldersOptions()
    options.results_file = r'd:\temp\mini.json'
    options.base_input_folder = r'd:\temp\demo_images\mini'
    options.base_output_folder = r'd:\temp\mini_out'
    
    separate_detections_into_folders(options)
        
    
#%% Command-line driver   

# python api\batch_processing\postprocessing\separate_detections_into_folders.py "d:\temp\rspb_mini.json" "d:\temp\demo_images\rspb_2018_2019_mini" "d:\temp\separation_test" --nthreads 2
    
import argparse
import inspect
import sys

# Copy all fields from a Namespace (i.e., the output from parse_args) to an object.  
#
# Skips fields starting with _.  Does not check field existence in the target object.
def argsToObject(args, obj):
    
    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v);

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', type=str, help='Input .json filename')
    parser.add_argument('base_input_folder', type=str, help='Input image folder')
    parser.add_argument('base_output_folder', type=str, help='Output image folder')
    
    options = SeparateDetectionsIntoFoldersOptions()
    parser.add_argument('--animal_threshold', type=float, default=options.animal_threshold, 
                        help='Confidence threshold for the animal category')
    parser.add_argument('--human_threshold', type=float, default=options.human_threshold, 
                        help='Confidence threshold for the human category')
    parser.add_argument('--vehicle_threshold', type=float, default=options.vehicle_threshold, 
                        help='Confidence threshold for vehicle category')
    parser.add_argument('--nthreads', type=int, default=options.n_threads, 
                        help='Number of threads to use for parallel operation')
    parser.add_argument('--allow_existing_directory', action='store_true', 
                        help='Proceed even if the target directory exists and is not empty')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()    
    
    # Convert to an options object
    argsToObject(args,options)
    
    separate_detections_into_folders(options)
    
if __name__ == '__main__':
    
    main()
    
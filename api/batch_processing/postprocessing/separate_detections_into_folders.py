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
# Preserves relative paths within each of those folders; cannot be used with .json
# files that have absolute paths in them.
#
# Hard-coded to work with MDv3 and MDv4 output files.  Not currently future-proofed
# past the classes in MegaDetector v4, not currently ready for species-level classification.  

#%% Constants and imports

import os
import json
import shutil
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

output_folders = ['empty','animals','people','vehicles','multiple']


#%% Options class

class SeparateDetectionsIntoFoldersOptions:
    
    # Inputs
    animal_threshold = 0.725
    person_threshold = 0.725
    vehicle_threshold = 0.725
    n_threads = 1
    
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
    if (max_person_confidence >= options.person_threshold):
        n_thresholds += 1
    if (max_animal_confidence >= options.animal_threshold):
        n_thresholds += 1
    if (max_vehicle_confidence >= options.vehicle_threshold):
        n_thresholds += 1
    
    # If this is above multiple thresholds
    if n_thresholds > 1:
        target_folder = options.target_folders['multiple']

    # Else if this is above threshold for people...
    elif (max_person_confidence >= options.person_threshold):
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
    assert os.path.isfile(source_path)
    
    target_path = os.path.join(target_folder,relative_filename)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir,exist_ok=True)
    shutil.copyfile(source_path,target_path)
    
# ...def process_detection()
    
    
#%% Main function
    
def path_is_abs(p): return (len(p) > 1) and (p[0] == '/' or p[1] == ':')
    
def separate_detections_into_folders(options):

    # Create output folder if necessary
    os.makedirs(options.base_output_folder,exist_ok=True)    
    
    # Load detection results    
    results = json.load(open(options.results_file))
    detections = results['images']
    
    for d in detections:
        fn = d['file']
        assert path_is_abs(fn), 'Cannot process results with absolute image paths'
        
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
    target_folders = {}
    
    for f in output_folders:
        target_folders[f] = os.path.join(options.base_output_folder,f)
        os.makedirs(target_folders[f],exist_ok=True)            
        
    if options.n_threads <= 1:
    
        # i_image = 0; d = detections_to_process[i_image]
        for d in tqdm(detections):
            process_detection(d)
        
    else:
        
        pool = ThreadPool(options.n_threads)        
        results = list(tqdm(pool.imap(process_detection, detections), total=len(detections)))
        
        
#%% Interactive driver
        
if False:

    pass

    #%%
    
    options = SeparateDetectionsIntoFoldersOptions()
    options.results_file = r'd:\temp\rspb_mini.json'
    options.base_input_folder = r'd:\temp\rspb_mini'
    options.base_output_folder = r'd:\temp\rspb_mini_out'
    
    
    
#%% Command-line driver   
    
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
    parser.add_argument('--human_threshold', type=float, default=options.person_threshold, 
                        help='Confidence threshold for the human category')
    parser.add_argument('--vehicle_threshold', type=float, default=options.vehicle_threshold, 
                        help='Confidence threshold for vehicle category')
    parser.add_argument('--nthreads', type=int, default=options.n_threads, 
                        help='Number of threads to use for parallel operation')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()    
    
    # Convert to an options object
    argsToObject(args,options)
    
    separate_detections_into_folders(options)
    
if __name__ == '__main__':
    
    main()
    
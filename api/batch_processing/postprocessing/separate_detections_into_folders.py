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
# By default, images are just copied to the target output folder.  If you specify --render_boxes,
# bounding boxes will be rendered on the output images.  Because this is no longer strictly
# a copy operation, this may result in the loss of metadata.  More accurately, this *may*
# result in the loss of some EXIF metadata; this *will* result in the loss of IPTC/XMP metadata.
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
from detection.run_detector import get_detector_metadata_from_version_string        
from detection.run_detector import get_detector_version_from_filename
from tqdm import tqdm

from ct_utils import args_to_object

import visualization.visualization_utils as viz_utils

friendly_folder_names = {'animal':'animals','person':'people','vehicle':'vehicles'}

# Occasionally we have near-zero confidence detections associated with COCO classes that
# didn't quite get squeezed out of the model in training.  As long as they're near zero
# confidence, we just ignore them.
invalid_category_epsilon = 0.00001

default_line_thickness = 8
default_box_expansion = 3


#%% Options class

class SeparateDetectionsIntoFoldersOptions:

    def __init__(self,threshold=None):
        
        self.threshold = None
        
        self.category_name_to_threshold = {
            'animal': self.threshold,
            'person': self.threshold,
            'vehicle': self.threshold
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
        
        self.render_boxes = False
        self.line_thickness = default_line_thickness
        self.box_expansion = default_box_expansion
        
    # ...__init__()
    
# ...class SeparateDetectionsIntoFoldersOptions        
        
    
#%% Support functions
    
def path_is_abs(p): return (len(p) > 1) and (p[0] == '/' or p[1] == ':')

printed_missing_file_warning = False
    
def process_detections(im,options):
    """
    Process all detections for a single image
    
    May modify *im*.
    """

    global printed_missing_file_warning
    
    relative_filename = im['file']
    
    detections = None    
    if 'detections' in im:
        detections = im['detections']
    
    categories_above_threshold = None
    
    if detections is None:
        
        assert im['failure'] is not None and len(im['failure']) > 0
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
        
        # ...for each detection on this image
        
        # Count the number of thresholds exceeded
        categories_above_threshold = []
        for category_name in category_names:
            
            threshold = options.category_name_to_threshold[category_name]
            assert threshold is not None
                
            max_confidence_this_category = category_name_to_max_confidence[category_name]
            if max_confidence_this_category > threshold:
                categories_above_threshold.append(category_name)
        
        # ...for each category
        
        categories_above_threshold.sort()
        
        # If this is above multiple thresholds
        if len(categories_above_threshold) > 1:
            target_folder = options.category_name_to_folder['_'.join(categories_above_threshold)]
    
        elif len(categories_above_threshold) == 0:
            target_folder = options.category_name_to_folder['empty']
            
        else:
            target_folder = options.category_name_to_folder[categories_above_threshold[0]]
        
    # ...if this is/isn't a failure case
            
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
    
    # Do a simple copy operation if we don't need to render any boxes
    if (not options.render_boxes) or (categories_above_threshold is None):
        
        shutil.copyfile(source_path,target_path)
        
    else:
        
        # Open the source image
        pil_image = viz_utils.load_image(source_path)
        
        # Render bounding boxes for each category separately, beacuse
        # we allow different thresholds for each category.
        
        category_name_to_id = {v: k for k, v in options.category_id_to_category_name.items()}
        assert len(category_name_to_id) == len(options.category_id_to_category_name)
        
        for category_name in categories_above_threshold:
            
            category_id = category_name_to_id[category_name]
            category_threshold = options.category_name_to_threshold[category_name]
            assert category_threshold is not None
            category_detections = [d for d in detections if d['category'] == category_id]
            
            # Until we support classification, remove classification information to 
            # maintain standard detection colors.
            #
            # TODO: remove this later
            for d in category_detections:
                if 'classifications' in d:
                    del d['classifications']
            
            viz_utils.render_detection_bounding_boxes(
                category_detections, 
                pil_image,
                label_map=options.detection_categories,                                                  
                confidence_threshold=category_threshold,
                thickness=options.line_thickness,
                expansion=options.box_expansion)
        
        # ...for each category
        
        # Read EXIF metadata
        exif = pil_image.info['exif'] if ('exif' in pil_image.info) else None
        
        # Write output with EXIF metadata and quality='keep'
        if exif is not None:
            pil_image.save(target_path, exif=exif, quality='keep')
        else:
            pil_image.save(target_path, quality='keep')
        
        # Also see:
        #
        # https://discuss.dizzycoding.com/determining-jpg-quality-in-python-pil/
        # 
        # ...for more ways to preserve jpeg quality if quality='keep' doesn't do the trick.
        
    # ...if we don't/do need to render boxes
    
# ...def process_detections()
    
    
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
    images = results['images']
    
    for im in images:
        fn = im['file']
        assert not path_is_abs(fn), 'Cannot process results with absolute image paths'
        
    print('Processing detections for {} images'.format(len(images)))
    
    default_threshold = options.threshold
    
    if default_threshold is None:
        
        if 'detector_metadata' in results['info'] and \
            'typical_detection_threshold' in results['info']['detector_metadata']:
            default_threshold = results['info']['detector_metadata']['typical_detection_threshold']
        else:
            print('Warning: detector metadata not available in results file, inferring from MD version')
            detector_filename = results['info']['detector']
            detector_version = get_detector_version_from_filename(detector_filename)
            detector_metadata = get_detector_metadata_from_version_string(detector_version)
            default_threshold = detector_metadata['typical_detection_threshold']
    
    detection_categories = results['detection_categories']    
    options.detection_categories = detection_categories
    options.category_id_to_category_name = detection_categories
    
    # Map class names to output folders
    options.category_name_to_folder = {}
    options.category_name_to_folder['empty'] = os.path.join(options.base_output_folder,'empty')
    options.category_name_to_folder['failure'] = os.path.join(options.base_output_folder,'processing_failure')
    
    # Create all combinations of categories
    category_names = list(detection_categories.values())
    category_names.sort()

    # category_name = category_names[0]
    for category_name in category_names:        

        # Do we have a custom threshold for this category?
        assert category_name in options.category_name_to_threshold
        if options.category_name_to_threshold[category_name] is None:
            options.category_name_to_threshold[category_name] = default_threshold
            
        category_threshold = options.category_name_to_threshold[category_name]
        print('Processing category {} at threshold {}'.format(category_name,category_threshold))
            
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
    
        # i_image = 1; im = images[i_image]; im
        for i_image,im in enumerate(tqdm(images)):
            if options.debug_max_images is not None and i_image > options.debug_max_images:
                break
            process_detections(im,options)
        # ...for each image
        
    else:
        
        print('Starting a pool with {} threads'.format(options.n_threads))
        pool = ThreadPool(options.n_threads)
        process_detections_with_options = partial(process_detections, options=options)
        results = list(tqdm(pool.imap(process_detections_with_options, images), total=len(images)))
        
#  ...def separate_detections_into_folders


#%% Interactive driver
        
if False:

    pass

    #%%
    
    options = SeparateDetectionsIntoFoldersOptions()
    
    options.results_file = os.path.expanduser('~/data/snapshot-safari-2022-08-16-KRU-v5a.0.0_detections.json')
    options.base_input_folder = os.path.expanduser('~/data/KRU/KRU_public')
    options.base_output_folder = os.path.expanduser('~/data/KRU-separated')
    options.n_threads = 100
    options.render_boxes = True
    options.allow_existing_directory = True
    
    #%%
    
    separate_detections_into_folders(options)
    
    #%%
    
    """
    # With boxes
    python separate_detections_into_folders.py ~/data/ena24-2022-06-15-v5a.0.0_megaclassifier.json ~/data/ENA24/images ~/data/ENA24-separated --threshold 0.17 --animal_threshold 0.2 --n_threads 10 --allow_existing_directory --render_boxes --line_thickness 10 --box_expansion 10
    
    # No boxes
    python separate_detections_into_folders.py ~/data/ena24-2022-06-15-v5a.0.0_megaclassifier.json ~/data/ENA24/images ~/data/ENA24-separated --threshold 0.17 --animal_threshold 0.2 --n_threads 10 --allow_existing_directory
    """    
    
#%% Command-line driver   

# python api\batch_processing\postprocessing\separate_detections_into_folders.py "d:\temp\rspb_mini.json" "d:\temp\demo_images\rspb_2018_2019_mini" "d:\temp\separation_test" --nthreads 2


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('results_file', type=str, help='Input .json filename')
    parser.add_argument('base_input_folder', type=str, help='Input image folder')
    parser.add_argument('base_output_folder', type=str, help='Output image folder')

    parser.add_argument('--threshold', type=float, default=None,
                        help='Default confidence threshold for all categories (defaults to selection based on model version, other options may override this for specific categories)')
    
    parser.add_argument('--animal_threshold', type=float, default=None,
                        help='Confidence threshold for the animal category')
    parser.add_argument('--human_threshold', type=float, default=None,
                        help='Confidence threshold for the human category')
    parser.add_argument('--vehicle_threshold', type=float, default=None,
                        help='Confidence threshold for vehicle category')
    
    parser.add_argument('--n_threads', type=int, default=1,
                        help='Number of threads to use for parallel operation (default=1)')
    parser.add_argument('--allow_existing_directory', action='store_true', 
                        help='Proceed even if the target directory exists and is not empty')
    parser.add_argument('--no_overwrite', action='store_true', 
                        help='Skip images that already exist in the target folder, must also specify --allow_existing_directory')
    
    parser.add_argument('--render_boxes', action='store_true',
                        help='Render bounding boxes on output images; may result in some metadata not being transferred')
    parser.add_argument('--line_thickness', type=int, default=default_line_thickness,
                        help='Line thickness (in pixels) for rendering, only meaningful if using render_boxes (defaults to {})'.format(
                            default_line_thickness))
    parser.add_argument('--box_expansion', type=int, default=default_line_thickness,
                        help='Box expansion (in pixels) for rendering, only meaningful if using render_boxes (defaults to {})'.format(
                            default_box_expansion))
        
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()    
    
    # Convert to an options object
    options = SeparateDetectionsIntoFoldersOptions()
    
    args_to_object(args, options)

    options.category_name_to_threshold['animal'] = args.animal_threshold
    options.category_name_to_threshold['person'] = args.human_threshold
    options.category_name_to_threshold['vehicle'] = args.vehicle_threshold
    
    options.overwrite = (not args.no_overwrite)
        
    separate_detections_into_folders(options)
    
if __name__ == '__main__':
    
    main()
    
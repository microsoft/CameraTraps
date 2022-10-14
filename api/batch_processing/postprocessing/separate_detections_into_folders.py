#
# separate_detections_into_folders.py
#
### Overview
#
# Given a .json file with batch processing results, separate the files in that
# set of results into folders that contain animals/people/vehicles/nothing, 
# according to per-class thresholds.
#
# Image files are copied, not moved.
#
#
### Output structure
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
# a/x/y/5.jpg
#
# And let's say:
#
# * The results say that the first three images are empty/person/vehicle, respectively
# * The fourth image is above threshold for "animal" and "person"
# * The fifth image contains an animal
#
# * You specify an output base folder of c:\out
#
# You will get the following files:
#
# c:\out\empty\a\b\c\1.jpg
# c:\out\people\a\b\d\2.jpg
# c:\out\vehicles\a\b\e\3.jpg
# c:\out\animal_person\a\b\f\4.jpg
# c:\out\animals\a\x\y\5.jpg
#
### Rendering bounding boxes
#
# By default, images are just copied to the target output folder.  If you specify --render_boxes,
# bounding boxes will be rendered on the output images.  Because this is no longer strictly
# a copy operation, this may result in the loss of metadata.  More accurately, this *may*
# result in the loss of some EXIF metadata; this *will* result in the loss of IPTC/XMP metadata.
#
# Rendering boxes also makes this script a lot slower.
#
### Classification-based separation
#
# If you have a results file with classification data, you can also specify classes to put
# in their own folders, within the "animals" folder, like this:
#
# --classification_thresholds "deer=0.75,cow=0.75"
#
# So, e.g., you might get:
#
# c:\out\animals\deer\a\x\y\5.jpg
#
# In this scenario, the folders within "animals" will be:
#
# deer, cow, multiple, unclassified
#
# "multiple" in this case only means "deer and cow"; if an image is classified as containing a 
# bird and a bear, that would end up in "unclassified", since the folder separation is based only
# on the categories you provide at the command line.
#
# No classification-based separation is done within the animal_person, animal_vehicle, or 
# animal_person_vehicle folders.
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
        self.skip_empty_images = False
        
        self.results_file = None
        self.base_input_folder = None
        self.base_output_folder = None
                  
        # Dictionary mapping categories (plus combinations of categories, and 'empty') to output folders
        self.category_name_to_folder = None
        self.category_id_to_category_name = None
        self.debug_max_images = None
        
        # Populated only when using classification results
        self.classification_category_id_to_name = None
        self.classification_categories = None
                
        self.render_boxes = False
        self.line_thickness = default_line_thickness
        self.box_expansion = default_box_expansion
        
        
        # Originally specified as a string, converted to a dict mapping name:threshold
        self.classification_thresholds = None
        
    # ...__init__()
    
# ...class SeparateDetectionsIntoFoldersOptions        
        
    
#%% Support functions
    
def path_is_abs(p): return (len(p) > 1) and (p[0] == '/' or p[1] == ':')

def is_float(v):
    """
    Determines whether v is either a float or a string representation of a float.
    """
    try:
        _ = float(v)
        return True
    except ValueError:
        return False

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
            if max_confidence_this_category >= threshold:
                categories_above_threshold.append(category_name)
        
        # ...for each category
        
        categories_above_threshold.sort()
        
        using_classification_folders = (options.classification_thresholds is not None and \
                                        len(options.classification_thresholds) > 0)
        
        # If this is above multiple thresholds
        if len(categories_above_threshold) > 1:
            
            # TODO: handle species-based separation in, e.g., the animal_person case
            target_folder = options.category_name_to_folder['_'.join(categories_above_threshold)]
    
        elif len(categories_above_threshold) == 0:
            target_folder = options.category_name_to_folder['empty']
            
        else:
            
            assert len(categories_above_threshold) == 1
            
            target_folder = options.category_name_to_folder[categories_above_threshold[0]]
            
            # Are we making species classification folders, and is this an animal?
            if ('animal' in categories_above_threshold) and (using_classification_folders):
                   
                # Do we need to put this into a specific species folder?
                
                # Find the animal-class detections that are above threshold
                category_name_to_id = {v: k for k, v in options.category_id_to_category_name.items()}
                animal_category_id = category_name_to_id['animal']
                valid_animal_detections = [d for d in detections if \
                                           (d['category'] == animal_category_id and \
                                           d['conf'] >= options.category_name_to_threshold['animal'])]
                    
                # Count the number of classification categories that are above threshold for at
                # least one detection
                classification_categories_above_threshold = set()
                
                # d = valid_animal_detections[0]
                for d in valid_animal_detections:
                    
                    if 'classifications' not in d or d['classifications'] is None:
                        continue
                    
                    # classification = d['classifications'][0]
                    for classification in d['classifications']:
                        
                        classification_category_id = classification[0]
                        classification_confidence = classification[1]
                        
                        # Do we have a threshold for this category, and if so, is 
                        # this classification above threshold?
                        assert options.classification_category_id_to_name is not None
                        classification_category_name = \
                            options.classification_category_id_to_name[classification_category_id]
                        if (classification_category_name in options.classification_thresholds) and \
                            (classification_confidence > \
                             options.classification_thresholds[classification_category_name]):
                            classification_categories_above_threshold.add(classification_category_name)
                            
                    # ...for each classification
                
                # ...for each detection
                                
                if len(classification_categories_above_threshold) == 0:
                    classification_folder_name = 'unclassified'
                    
                elif len(classification_categories_above_threshold) > 1:
                    classification_folder_name = 'multiple'
                    
                else:
                    assert len(classification_categories_above_threshold) == 1
                    classification_folder_name = list(classification_categories_above_threshold)[0]                    
                
                target_folder = os.path.join(target_folder,classification_folder_name)
                
            # ...if we have to deal with classification subfolders            
            
        # ...if we have 0/1/more categories above threshold
        
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
    
    # Skip this image if it's empty and we're not processing empty images
    if ((categories_above_threshold is None) or (len(categories_above_threshold) == 0)) and \
        options.skip_empty_images:
        return
        
    # At this point, this image is getting copied; we may or may not also need to
    # draw bounding boxes.
    
    # Do a simple copy operation if we don't need to render any boxes
    if (not options.render_boxes) or \
        (categories_above_threshold is None) or \
        (len(categories_above_threshold) == 0):
        
        shutil.copyfile(source_path,target_path)
        
    else:
        
        # Open the source image
        pil_image = viz_utils.load_image(source_path)
        
        # Render bounding boxes for each category separately, beacuse
        # we allow different thresholds for each category.
        
        category_name_to_id = {v: k for k, v in options.category_id_to_category_name.items()}
        assert len(category_name_to_id) == len(options.category_id_to_category_name)
        
        classification_label_map = None
        if using_classification_folders:
            classification_label_map = options.classification_categories
            
        for category_name in categories_above_threshold:
            
            category_id = category_name_to_id[category_name]
            category_threshold = options.category_name_to_threshold[category_name]
            assert category_threshold is not None
            category_detections = [d for d in detections if d['category'] == category_id]
            
            # When we're not using classification folders, remove classification
            # information to maintain standard detection colors.
            if not using_classification_folders:
                for d in category_detections:
                    if 'classifications' in d:
                        del d['classifications']
                
            viz_utils.render_detection_bounding_boxes(
                category_detections, 
                pil_image,
                label_map=options.detection_categories,                                                  
                classification_label_map=classification_label_map,
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
        elif ('detector' not in results['info']) or (results['info']['detector'] is None):
            print('Warning: detector version not available in results file, using MDv5 defaults')
            detector_metadata = get_detector_metadata_from_version_string('v5a.0.0')
            default_threshold = detector_metadata['typical_detection_threshold']
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
    options.category_name_to_folder['failure'] =\
        os.path.join(options.base_output_folder,'processing_failure')
    
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
    
    # Handle species classification thresholds, if specified
    if options.classification_thresholds is not None:
        
        assert 'classification_categories' in results and \
            results['classification_categories'] is not None, \
            'Classification thresholds specified, but no classification results available'
        
        classification_categories = results['classification_categories']
        classification_category_name_to_id = {v: k for k, v in classification_categories.items()}
        classification_category_id_to_name = {k: v for k, v in classification_categories.items()}
        options.classification_category_id_to_name = classification_category_id_to_name
        options.classification_categories = classification_categories
        
        if isinstance(options.classification_thresholds,str):
            
            # E.g. deer=0.75,cow=0.75
            tokens = options.classification_thresholds.split(',')
            classification_thresholds = {}
            
            # token = tokens[0]
            for token in tokens:
                subtokens = token.split('=')
                assert len(subtokens) == 2 and is_float(subtokens[1]), \
                    'Illegal classification threshold {}'.format(token)                                
                classification_thresholds[subtokens[0]] = float(subtokens[1])
                
            # ...for each token
            
            options.classification_thresholds = classification_thresholds
            
        # ...if classification thresholds are still in string format
        
        # Validate the classes in the threshold list
        for class_name in options.classification_thresholds.keys():
            assert class_name in classification_category_name_to_id, \
                'Category {} specified at the command line, but is not available in the results file'.format(
                    class_name)

    # ...if we need to deal with classification categories
    
    if options.n_threads <= 1 or options.debug_max_images is not None:
    
        # i_image = 14; im = images[i_image]; im
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
    
    options = SeparateDetectionsIntoFoldersOptions()
    
    options.results_file = os.path.expanduser('~/data/ena24-2022-06-15-v5a.0.0_megaclassifier.json')
    options.base_input_folder = os.path.expanduser('~/data/ENA24/images')
    options.base_output_folder = os.path.expanduser('~/data/ENA24-separated')
    options.n_threads = 100
    options.classification_thresholds = 'deer=0.75,cow=0.75,bird=0.75'
    options.render_boxes = True
    options.allow_existing_directory = True
    
    #%%
    
    separate_detections_into_folders(options)
    
    #%% Testing various command-line invocations
    
    """
    # With boxes, no classification
    python separate_detections_into_folders.py ~/data/ena24-2022-06-15-v5a.0.0_megaclassifier.json ~/data/ENA24/images ~/data/ENA24-separated --threshold 0.17 --animal_threshold 0.2 --n_threads 10 --allow_existing_directory --render_boxes --line_thickness 10 --box_expansion 10
    
    # No boxes, no classification (default)
    python separate_detections_into_folders.py ~/data/ena24-2022-06-15-v5a.0.0_megaclassifier.json ~/data/ENA24/images ~/data/ENA24-separated --threshold 0.17 --animal_threshold 0.2 --n_threads 10 --allow_existing_directory
    
    # With boxes, with classification
    python separate_detections_into_folders.py ~/data/ena24-2022-06-15-v5a.0.0_megaclassifier.json ~/data/ENA24/images ~/data/ENA24-separated --threshold 0.17 --animal_threshold 0.2 --n_threads 10 --allow_existing_directory --render_boxes --line_thickness 10 --box_expansion 10 --classification_thresholds "deer=0.75,cow=0.75,bird=0.75"
    
    # No boxes, with classification
    python separate_detections_into_folders.py ~/data/ena24-2022-06-15-v5a.0.0_megaclassifier.json ~/data/ENA24/images ~/data/ENA24-separated --threshold 0.17 --animal_threshold 0.2 --n_threads 10 --allow_existing_directory --classification_thresholds "deer=0.75,cow=0.75,bird=0.75"
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
    parser.add_argument('--classification_thresholds', type=str, default=None,
                        help='List of classification thresholds to use for species-based folder separation, formatted like deer=0.75,cow=0.75"')
    
    parser.add_argument('--n_threads', type=int, default=1,
                        help='Number of threads to use for parallel operation (default=1)')
    parser.add_argument('--allow_existing_directory', action='store_true', 
                        help='Proceed even if the target directory exists and is not empty')
    parser.add_argument('--no_overwrite', action='store_true', 
                        help='Skip images that already exist in the target folder, must also specify --allow_existing_directory')    
    parser.add_argument('--skip_empty_images', action='store_true',
                        help='Don\'t copy empty images to the output folder')
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

    def validate_threshold(v,name):
        # print('{} {}'.format(v,name))
        if v is not None:
            assert v >= 0.0 and v <= 1.0, \
                'Illegal {} threshold {}'.format(name,v)
    
    validate_threshold(args.threshold,'default')
    validate_threshold(args.animal_threshold,'animal')
    validate_threshold(args.vehicle_threshold,'vehicle')
    validate_threshold(args.human_threshold,'human')
    
    if args.threshold is not None:
        if args.animal_threshold is not None \
            and args.human_threshold is not None \
            and args.vehicle_threshold is not None:
                raise ValueError('Default threshold specified, but all category thresholds also specified... not exactly wrong, but it\'s likely that you meant something else.')        
                    
    options.category_name_to_threshold['animal'] = args.animal_threshold
    options.category_name_to_threshold['person'] = args.human_threshold
    options.category_name_to_threshold['vehicle'] = args.vehicle_threshold
    
    options.overwrite = (not args.no_overwrite)
        
    separate_detections_into_folders(options)
    
if __name__ == '__main__':
    
    main()
    
#
# Merge high-confidence detections from one results file into another file,
# when the target file does not detect anything on an image.
#
# Does not currently attempt to merge every detection based on whether individual 
# detections are missing; only merges detections into images that would otherwise
# be considered blank.
#
# If you want to literally merge two .json files, see combine_api_outputs.py.
#

#%% Constants and imports

import json
import os
from tqdm import tqdm


#%% Structs

class MergeDetectionsOptions:
    
    def __init__(self):
        
        self.max_detection_size = 1.01
        self.min_detection_size = 0
        self.source_confidence_thresholds = [0.8]
        
        # Don't bother merging into target images where the max detection is already
        # higher than this threshold
        self.target_confidence_threshold = 0.8
        
        # If you want to merge only certain categories, specify one
        # (but not both) of these.
        self.categories_to_include = None
        self.categories_to_exclude = None
        

#%% Main function

def merge_detections(source_files,target_file,output_file,options=None):
    
    if isinstance(source_files,str):
        source_files = [source_files]    
        
    if options is None:
        options = MergeDetectionsOptions()    
        
    assert not ((options.categories_to_exclude is not None) and \
                (options.categories_to_include is not None)), \
                'categories_to_include and categories_to_exclude are mutually exclusive'
    
    if options.categories_to_exclude is not None:
        options.categories_to_exclude = [int(c) for c in options.categories_to_exclude]
        
    if options.categories_to_include is not None:
        options.categories_to_include = [int(c) for c in options.categories_to_include]
        
    assert len(source_files) == len(options.source_confidence_thresholds)
    
    for fn in source_files:
        assert os.path.isfile(fn), 'Could not find source file {}'.format(fn)
    
    assert os.path.isfile(target_file)
    
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    
    with open(target_file,'r') as f:
        output_data = json.load(f)

    print('Loaded results for {} images'.format(len(output_data['images'])))
    
    fn_to_image = {}
    
    # im = output_data['images'][0]
    for im in output_data['images']:
        fn_to_image[im['file']] = im
    
    if 'detections_transferred_from' not in output_data['info']:
        output_data['info']['detections_transferred_from'] = []

    if 'detector' not in output_data['info']:
        output_data['info']['detector'] = 'MDv4 (assumed)'
        
    detection_categories_raw = output_data['detection_categories'].keys()
    
    # Determine whether we should be processing all categories, or just a subset
    # of categories.
    detection_categories = []

    if options.categories_to_exclude is not None:    
        for c in detection_categories_raw:
            if int(c) not in options.categories_to_exclude:
                detection_categories.append(c)
            else:
                print('Excluding category {}'.format(c))
    elif options.categories_to_include is not None:
        for c in detection_categories_raw:
            if int(c) in options.categories_to_include:
                print('Including category {}'.format(c))
                detection_categories.append(c)
    else:
        detection_categories = detection_categories_raw
    
    # i_source_file = 0; source_file = source_files[i_source_file]
    for i_source_file,source_file in enumerate(source_files):
        
        print('Processing detections from file {}'.format(source_file))
        
        with open(source_file,'r') as f:
            source_data = json.load(f)
        
        if 'detector' in source_data['info']:
            source_detector_name = source_data['info']['detector']
        else:
            source_detector_name = os.path.basename(source_file)
        
        output_data['info']['detections_transferred_from'].append(os.path.basename(source_file))
        output_data['info']['detector'] = output_data['info']['detector'] + ' + ' + source_detector_name
        
        assert source_data['detection_categories'] == output_data['detection_categories']
        
        source_confidence_threshold = options.source_confidence_thresholds[i_source_file]
        
        # source_im = source_data['images'][0]
        for source_im in tqdm(source_data['images']):
            
            image_filename = source_im['file']            
            
            assert image_filename in fn_to_image, 'Image {} not in target image set'.format(image_filename)
            target_im = fn_to_image[image_filename]
            
            if 'detections' not in source_im or source_im['detections'] is None:
                continue
            
            if 'detections' not in target_im or target_im['detections'] is None:
                continue
                    
            source_detections_this_image = source_im['detections']
            target_detections_this_image = target_im['detections']
              
            detections_to_transfer = []
            
            # detection_category = list(detection_categories)[0]
            for detection_category in detection_categories:
                
                target_detections_this_category = \
                    [det for det in target_detections_this_image if det['category'] == \
                     detection_category]
                
                max_target_confidence_this_category = 0.0
                
                if len(target_detections_this_category) > 0:
                    max_target_confidence_this_category = max([det['conf'] for \
                      det in target_detections_this_category])
                
                # This is already a detection, no need to proceed looking for detections to 
                # transfer
                if max_target_confidence_this_category >= options.target_confidence_threshold:
                    continue
                
                source_detections_this_category_raw = [det for det in \
                  source_detections_this_image if det['category'] == detection_category]
                
                # Boxes are x/y/w/h
                # source_sizes = [det['bbox'][2]*det['bbox'][3] for det in source_detections_this_category_raw]
                
                # Only look at boxes below the size threshold
                source_detections_this_category_filtered = [
                    det for det in source_detections_this_category_raw if \
                        (det['bbox'][2]*det['bbox'][3] <= options.max_detection_size) and \
                        (det['bbox'][2]*det['bbox'][3] >= options.min_detection_size) \
                        ]
                                
                for det in source_detections_this_category_filtered:
                    if det['conf'] >= source_confidence_threshold:
                        det['transferred_from'] = source_detector_name
                        detections_to_transfer.append(det)
                    
            # ...for each detection category
            
            if len(detections_to_transfer) > 0:
                # print('Adding {} detections to image {}'.format(len(detections_to_transfer),image_filename))
                detections = fn_to_image[image_filename]['detections']                
                detections.extend(detections_to_transfer)

                # Update the max_detection_conf field (if present)
                if 'max_detection_conf' in fn_to_image[image_filename]:
                    fn_to_image[image_filename]['max_detection_conf'] = \
                        max([d['conf'] for d in detections])
                
        # ...for each image
        
    # ...for each source file        
    
    with open(output_file,'w') as f:
        json.dump(output_data,f,indent=2)
    
    print('Saved merged results to {}'.format(output_file))


#%% Test driver

if False:
    
    #%%
    
    options = MergeDetectionsOptions()
    options.max_detection_size = 0.1
    options.target_confidence_threshold = 0.3
    options.categories_to_include = [1]
    source_files = ['/home/user/postprocessing/iwildcam/iwildcam-mdv4-2022-05-01/combined_api_outputs/iwildcam-mdv4-2022-05-01_detections.json']
    options.source_confidence_thresholds = [0.8]
    target_file = '/home/user/postprocessing/iwildcam/iwildcam-mdv5-camcocoinat-2022-05-02/combined_api_outputs/iwildcam-mdv5-camcocoinat-2022-05-02_detections.json'
    output_file = '/home/user/postprocessing/iwildcam/merged-detections/mdv4_mdv5-camcocoinat-2022-05-02.json'
    merge_detections(source_files, target_file, output_file, options)
    
    options = MergeDetectionsOptions()
    options.max_detection_size = 0.1
    options.target_confidence_threshold = 0.3
    options.categories_to_include = [1]
    source_files = [
        '/home/user/postprocessing/iwildcam/iwildcam-mdv4-2022-05-01/combined_api_outputs/iwildcam-mdv4-2022-05-01_detections.json',
        '/home/user/postprocessing/iwildcam/iwildcam-mdv5-camonly-2022-05-02/combined_api_outputs/iwildcam-mdv5-camonly-2022-05-02_detections.json',
        ]
    options.source_confidence_thresholds = [0.8,0.5]
    target_file = '/home/user/postprocessing/iwildcam/iwildcam-mdv5-camcocoinat-2022-05-02/combined_api_outputs/iwildcam-mdv5-camcocoinat-2022-05-02_detections.json'
    output_file = '/home/user/postprocessing/iwildcam/merged-detections/mdv4_mdv5-camonly_mdv5-camcocoinat-2022-05-02.json'
    merge_detections(source_files, target_file, output_file, options)
    
    
#%% Command-line driver (TODO)

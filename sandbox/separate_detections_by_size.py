#
# separate_detections_by_size
#
# Not-super-well-maintained script to break a list of API output files up
# based on bounding box size.
#

#%% Imports and constants

import json
import os

# Folder with one or more .json files in it that we want to split up
output_base = r'g:\my_data\combined_api_outputs'

# Enumerate .json files
input_files = os.listdir(output_base)
input_files = [os.path.join(output_base,fn) for fn in input_files]
input_files = [fn for fn in input_files if fn.endswith('.json')]

# Define size thresholds and confidence thresholds
size_thresholds = [0.03,0.05,0.065,0.075,0.1]
confidence_threshold = 0.7

# Not used directly in this script, but useful if we want to generate previews
smallbox_files = []
bigbox_files = []
empty_files = []


#%% Split by size

# For each size threshold...
for i_size,size_threshold in enumerate(size_thresholds):
    
    # For each file...
    # fn = input_files[0]
    for fn in input_files:
            
        # Just double-checking; we already filtered this out above
        if not fn.endswith('.json'):        
            continue
        
        # Don't reprocess .json files we generated with this script
        if 'small' in fn or 'big' in fn or 'empty' in fn:
            continue
        
        print('Processing file {} at size threshold {}'.format(fn,size_threshold))
        
        smallbox_filename = fn.replace('.json','.smallbox_{0:.2f}.json'.format(size_threshold))
        smallbox_files.append(smallbox_filename)
        
        bigbox_filename = fn.replace('.json','.bigbox_{0:.2f}.json'.format(size_threshold))
        bigbox_files.append(bigbox_filename)
        
        empty_filename = fn.replace('.json','.empty_{0:.2f}.json'.format(size_threshold))
        empty_files.append(empty_filename)
        
        smallbox_images = []
        bigbox_images = []
        empty_images = []
        failed_images = []
        
        # Load the input file
        with open(fn) as f:
            data = json.load(f)
        
        images = data['images']
        print('Loaded {} images'.format(len(images)))
        
        # For each image...
        for im in images:
    
            # 1.1 is the same as infinity here; no box can be bigger than a whole image
            smallest_detection_size_above_threshold = 1.1        
            n_above_threshold = 0
            
            if 'detections' not in im:
                print('Warning: no detections for image {}'.format(im['file']))
                failed_images.append(im)
                continue
            
            # What's the smallest detection above threshold?
            for d in im['detections']:
                
                if d['conf'] < confidence_threshold:
                    continue
                n_above_threshold += 1
                
                # [x_min, y_min, width_of_box, height_of_box]
                #
                # size = w * h
                box_size = d['bbox'][2] * d['bbox'][3]
                
                if box_size < smallest_detection_size_above_threshold:
                    smallest_detection_size_above_threshold = box_size
                    
            # ...for each detection
            
            # Which list do we put this image on?
            if n_above_threshold == 0:
                empty_images.append(im)
            elif smallest_detection_size_above_threshold < size_threshold:
                smallbox_images.append(im)
            else:
                bigbox_images.append(im)
    
        # ...for each image in this file
    
        # Make sure the number of images adds up
        assert len(bigbox_images) + len(smallbox_images) + len(empty_images) + len(failed_images) == len(images)
        
        # Write out all files
        data['images'] = smallbox_images
        with open(smallbox_filename, 'w') as f:
            json.dump(data, f, indent=1)
        
        data['images'] = bigbox_images
        with open(bigbox_filename, 'w') as f:
            json.dump(data, f, indent=1)
            
        data['images'] = empty_images
        with open(empty_filename, 'w') as f:
            json.dump(data, f, indent=1)
    
    # ...for each size threshold
            
# ...for each file

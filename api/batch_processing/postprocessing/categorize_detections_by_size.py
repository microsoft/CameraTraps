"""

categorize_detections_by_size.py

Given an API output .json file, creates a separate category for bounding boxes
above a size threshold.

"""

import json

class SizeCategorizationOptions:

    threshold=0.95
    
def categorize_detections_by_size(input_file,output_file,options=None):
    
    if options is None:
        options = SizeCategorizationOptions()
    
    with open(input_file) as f:
        data = json.load(f)
    
    detection_categories = data['detection_categories']
    category_keys = list(detection_categories.keys())
    category_keys = [int(k) for k in category_keys]
    max_key = max(category_keys)
    large_detection_category_id = str(max_key+1)
    detection_categories[large_detection_category_id] = 'large_detection'
    
    print('Creating large-box category as {}'.format(large_detection_category_id))
    
    images = data['images']
    
    print('Loaded {} images'.format(len(images)))
        
    # For each image...
    #
    # im = images[0]
    
    n_large_detections = 0
    
    for im in images:
        
        if im['detections'] is None:
            assert im['failure'] is not None and len(im['failure']) > 0
            continue
            
        # d = im['detections'][0]
        for d in im['detections']:
            
            if (d is None) or ('bbox' not in d) or (d['bbox'] is None):
                continue
            
            # https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#detector-outputs
            w = d['bbox'][2]
            h = d['bbox'][3]
            detection_size = w*h
            if detection_size >= options.threshold:
                d['category'] = large_detection_category_id
                n_large_detections += 1
        
        # ...for each detection
        
    # ...for each image
    
    print('Found {} large detections'.format(n_large_detections))
    
    with open(output_file,'w') as f:
        json.dump(data,f,indent=1)
        
    return data
    
# ...def categorize_detections_by_size()
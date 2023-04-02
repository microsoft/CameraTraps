#
# cct_to_md.py
#
# "Converts" a COCO Camera Traps file to a MD results file.  Currently ignores
# non-bounding-box annotations, and gives all annotations a confidence of 1.0.
#
# The only reason to do this is if you are going to add information to an existing
# CCT-formatted dataset, and want to do that in Timelapse.
#
# Currently assumes that width and height are present in the input data, does not
# read them from images.
#

#%% Constants and imports

import os
import json

from collections import defaultdict
from tqdm import tqdm


#%% Functions

def cct_to_md(input_filename,output_filename=None):
    
    ## Validate input
    
    assert os.path.isfile(input_filename)
    
    if (output_filename is None):
        
        tokens = os.path.splitext(input_filename)
        assert len(tokens) == 2
        output_filename = tokens[0] + '_md-format' + tokens[1]
    
        
    ## Read input
    
    with open(input_filename,'r') as f:
        d = json.load(f)
        
    for s in ['annotations','images','categories']:
        assert s in d.keys(), 'Cannot find category {} in input file, is this a CCT file?'.format(s)
        
    
    ## Prepare metadata
    
    image_id_to_annotations = defaultdict(list)
    
    # ann = d['annotations'][0]
    for ann in tqdm(d['annotations']):
        image_id_to_annotations[ann['image_id']].append(ann)
    
    category_id_to_name = {}
    for cat in d['categories']:
        category_id_to_name[str(cat['id'])] = cat['name']
        
    results = {}
    
    info = {}
    info['format_version'] = 1.2
    info['detector'] = 'cct_to_md'
    results['info'] = info
    results['detection_categories'] = category_id_to_name
        
    
    ## Process images
    
    images_out = []
    
    # im = d['images'][0]
    for im in tqdm(d['images']):
        
        im_out = {}
        im_out['file'] = im['file_name']
        im_out['location'] = im['location']
        im_out['id'] = im['id']
        
        image_h = im['height']
        image_w = im['width']
        
        detections = []
        
        annotations_this_image = image_id_to_annotations[im['id']]
        
        max_detection_conf = 0
        
        for ann in annotations_this_image:
            
               if 'bbox' in ann:
                   
                   det = {}
                   det['category'] = str(ann['category_id'])
                   det['conf'] = 1.0
                   max_detection_conf = 1.0
                   
                   # MegaDetector: [x,y,width,height] (normalized, origin upper-left)
                   # CCT: [x,y,width,height] (absolute, origin upper-left)
                   bbox_in = ann['bbox']
                   bbox_out = [bbox_in[0]/image_w,bbox_in[1]/image_h,
                               bbox_in[2]/image_w,bbox_in[3]/image_h]
                   det['bbox'] = bbox_out
                   detections.append(det)
                   
              # ...if there's a bounding box
              
        # ...for each annotation
        
        im_out['detections'] = detections
        
        # This field is no longer included in MD output files by default
        # im_out['max_detection_conf'] = max_detection_conf
    
        images_out.append(im_out)
        
    # ...for each image
    
    
    ## Write output
    
    results['images'] = images_out
    
    with open(output_filename,'w') as f:
        json.dump(results, f, indent=1)
        
    return output_filename

# ...cct_to_md()    
    

#%% Command-line driver

# TODO


#%% Interactive driver

if False:
    
    pass

    #%%

    input_filename = r"G:\temp\noaa_estuary_fish.json"
    output_filename = None
    output_filename = cct_to_md(input_filename,output_filename)
    
    #%%
    
    from visualization import visualize_detector_output
    
    visualize_detector_output.visualize_detector_output(
                              detector_output_path=output_filename,
                              out_dir=r'g:\temp\fish_output',
                              images_dir=r'G:\temp\noaa_estuary_fish-images\JPEGImages',
                              output_image_width=-1,
                              sample=100,
                              render_detections_only=True)


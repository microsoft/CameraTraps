#
# coco_to_yolo.py
#
# Converts a COCO-formatted dataset to a YOLO-formatted dataset.  "COCO" here
# does *not* mean COCO Camera Traps, it means standard COCO, though it would be
# only a slight modification to this script to support COCO Camera Traps.
#
# I am writing this for a very specific use case, so there are some quirky limitations
# that will be easy to fix if/when we need to:
#
# * Currently ignores segmentation masks, and errors if an annotation has a 
#   segmentation polygon but no bbox
#
# * The data set I'm writing this for has width/height information in the COCO
#   .json file, so this script does not open up the images to read/verify width
#   and height.  This is technically required, but sometimes people are rebellious.
#
# Basically this is a work in progress, YMMV.
#

#%% Imports and constants

import json
import os
import shutil

from collections import defaultdict
from tqdm import tqdm


#%% Support functions

def coco_to_yolo(input_image_folder,output_folder,input_file):
    
    # Validate input
    
    assert os.path.isdir(input_image_folder)
    assert os.path.isfile(input_file)
    os.makedirs(output_folder,exist_ok=True)
    
    
    # Read input data
    
    with open(input_file,'r') as f:
        data = json.load(f)
        
        
    # Parse annotations
  
    image_id_to_annotations = defaultdict(list)
    
    # i_ann = 0; ann = data['annotations'][0]
    for i_ann,ann in enumerate(data['annotations']):
        
        # Make sure no annotations have *only* segmentation data 
        if ( \
            ('segmentation' in ann.keys()) and \
            (ann['segmentation'] is not None) and \
            (len(ann['segmentation']) > 0) ) \
            and \
            (('bbox' not in ann.keys()) or (ann['bbox'] is None) or (len(ann['bbox'])==0)):
                raise ValueError('Oops: segmentation data present without bbox information, this script isn\'t ready for this dataset')
        
        image_id_to_annotations[ann['image_id']].append(ann)
        
    print('Parsed annotations for {} images'.format(len(image_id_to_annotations)))
    
    
    # Re-map class IDs to make sure they run from 0...n-classes-1
    
    next_category_id = 0
    coco_id_to_yolo_id = {}
    for category in data['categories']:
        assert category['id'] not in coco_id_to_yolo_id
        coco_id_to_yolo_id[category['id']] = next_category_id
        next_category_id += 1
        
    
    # Process images (everything but I/O)
    
    # List of dictionaries with keys 'source_image','dest_image','bboxes','dest_txt'
    images_to_copy = []
    
    missing_images = []
    
    image_names = set()
    
    typical_image_extensions = set(['.jpg','.jpeg','.png','.gif','.tif','.bmp'])
    
    # i_image = 0; im = data['images'][i_image]
    for i_image,im in tqdm(enumerate(data['images']),total=len(data['images'])):
        
        output_info = {}
        source_image = os.path.join(input_image_folder,im['file_name'])        
        output_info['source_image'] = source_image
        
        tokens = os.path.splitext(im['file_name'])
        if tokens[1].lower() not in typical_image_extensions:
            print('Warning: unusual image file name {}'.format(im['file_name']))
                  
        image_name = tokens[0].replace('\\','/').replace('/','_') + '_' + str(i_image).zfill(6)
        assert image_name not in image_names, 'Image name collision for {}'.format(image_name)
        image_names.add(image_name)
        
        dest_image = os.path.join(output_folder,image_name + tokens[1])
        output_info['dest_image'] = dest_image
        dest_txt = os.path.join(output_folder,image_name + '.txt')
        output_info['dest_txt'] = dest_txt
        output_info['bboxes'] = []
        
        # assert os.path.isfile(source_image), 'Could not find image {}'.format(source_image)
        if not os.path.isfile(source_image):
            print('Warning: could not find image {}'.format(source_image))
            missing_images.append(im['file_name'])
            continue
        
        image_id = im['id']
        
        image_bboxes = []
        
        if image_id in image_id_to_annotations:
                        
            for ann in image_id_to_annotations[image_id]:
                if 'bbox' not in ann or ann['bbox'] is None or len(ann['bbox']) == 0:
                    # This is not entirely clear from the COCO spec, but it seems to be consensus
                    # that if you want to specify an image with no objects, you don't include any
                    # annotations for that image.
                    raise ValueError('If an annotation exists, it should have content')
                coco_bbox = ann['bbox']
                yolo_category_id = coco_id_to_yolo_id[ann['category_id']]
                
                # COCO: [x_min, y_min, width, height] in absolute coordinates
                # YOLO: [class, x_center, y_center, width, height] in normalized coordinates
                
                # Convert from COCO coordinates to YOLO coordinates
                img_w = im['width']
                img_h = im['height']
                
                x_min_absolute = coco_bbox[0]
                y_min_absolute = coco_bbox[1]
                box_w_absolute = coco_bbox[2]
                box_h_absolute = coco_bbox[3]
                
                x_center_absolute = (x_min_absolute + (x_min_absolute + box_w_absolute)) / 2
                y_center_absolute = (y_min_absolute + (y_min_absolute + box_h_absolute)) / 2
                
                x_center_relative = x_center_absolute / img_w
                y_center_relative = y_center_absolute / img_h
                
                box_w_relative = box_w_absolute / img_w
                box_h_relative = box_h_absolute / img_h
                
                yolo_box = [yolo_category_id,
                            x_center_relative, y_center_relative, 
                            box_w_relative, box_h_relative]
                
                image_bboxes.append(yolo_box)
                
            # ...for each annotation 
            
        # ...for each image
        
        output_info['bboxes'] = image_bboxes
        
        images_to_copy.append(output_info)        
    
    # ...for each image
        
    print('{} missing images (of {})'.format(len(missing_images),len(data['images'])))
    
    
    # Write output
    
    # output_info = images_to_copy[0]
    for output_info in images_to_copy:
        
        source_image = output_info['source_image']
        dest_image = output_info['dest_image']
        os.makedirs(os.path.dirname(dest_image),exist_ok=True)
        dest_txt = output_info['dest_txt']
        assert os.path.dirname(dest_image) == os.path.dirname(dest_txt)
        shutil.copyfile(source_image,dest_image)
        bboxes = output_info['bboxes']
        
        with open(dest_txt,'w') as f:
            
            # bbox = bboxes[0]
            for bbox in bboxes:
                assert len(bbox) == 5
                s = '{} {} {} {} {}'.format(bbox[0],bbox[1],bbox[2],bbox[3],bbox[4])
                f.write(s + '\n')
                
    # ...for each image                

# ...def coco_to_yolo()


#%% Interactive driver

if False:
    
    pass

    #%%
    
    input_image_folder = os.path.expanduser('~/tmp/test/images')
    output_folder = os.path.expanduser('~/tmp/test/yolo_train')
    input_file = os.path.expanduser('~/tmp/test/annotations_train.json')

    coco_to_yolo(input_image_folder,output_folder,input_file)    


#%% Command-line driver

# TODO

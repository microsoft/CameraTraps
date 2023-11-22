#
# coco_to_yolo.py
#
# Converts a COCO-formatted dataset to a YOLO-formatted dataset. 
#
# If the input and output folders are the same, writes .txt files to the input folder,
# and neither moves nor modifies images.
#
# Currently ignores segmentation masks, and errors if an annotation has a 
# segmentation polygon but no bbox
# 
# Has only been tested on a handful of COCO Camera Traps data sets; if you
# use it for more general COCO conversion, YMMV.
#

#%% Imports and constants

import json
import os
import shutil

from collections import defaultdict
from tqdm import tqdm


#%% Support functions

def coco_to_yolo(input_image_folder,output_folder,input_file,
                 source_format='coco',overwrite_images=False,
                 create_image_and_label_folders=False,
                 class_file_name='classes.txt',
                 allow_empty_annotations=False,
                 clip_boxes=False,
                 image_id_to_output_image_json_file=None):
    
    if output_folder is None:
        output_folder = input_image_folder
    
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
                raise ValueError('Oops: segmentation data present without bbox information, ' + \
                                 'this script isn\'t ready for this dataset')
        
        image_id_to_annotations[ann['image_id']].append(ann)
        
    print('Parsed annotations for {} images'.format(len(image_id_to_annotations)))
        
    # Re-map class IDs to make sure they run from 0...n-classes-1
    #
    # TODO: this allows unused categories in the output data set, which I *think* is OK,
    # but I'm only 81% sure.
    next_category_id = 0
    coco_id_to_yolo_id = {}
    yolo_id_to_name = {}
    for category in data['categories']:
        assert category['id'] not in coco_id_to_yolo_id
        coco_id_to_yolo_id[category['id']] = next_category_id
        yolo_id_to_name[next_category_id] = category['name']
        next_category_id += 1
        
    
    # Process images (everything but I/O)
    
    # List of dictionaries with keys 'source_image','dest_image','bboxes','dest_txt'
    images_to_copy = []
    
    missing_images = []
    
    image_names = set()
    
    typical_image_extensions = set(['.jpg','.jpeg','.png','.gif','.tif','.bmp'])
    
    printing_empty_annotation_warning = False
    
    image_id_to_output_image_name = {}
    
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
        
        assert im['id'] not in image_id_to_output_image_name
        image_id_to_output_image_name[im['id']] = image_name
        
        dest_image_relative = image_name + tokens[1]
        output_info['dest_image_relative'] = dest_image_relative
        dest_txt_relative = image_name + '.txt'
        output_info['dest_txt_relative'] = dest_txt_relative
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
                
                # If this annotation has no bounding boxes...
                if 'bbox' not in ann or ann['bbox'] is None or len(ann['bbox']) == 0:
                    
                    if source_format == 'coco':
                    
                        if not allow_empty_annotations:
                            # This is not entirely clear from the COCO spec, but it seems to be consensus
                            # that if you want to specify an image with no objects, you don't include any
                            # annotations for that image.
                            raise ValueError('If an annotation exists, it should have content')
                        else:
                            continue
                    
                    else:
                        
                        # We allow empty bbox lists in COCO camera traps; this is typically a negative
                        # example in a dataset that has bounding boxes, and 0 is typically the empty 
                        # category.
                        if ann['category_id'] != 0:
                            if not printing_empty_annotation_warning:
                                printing_empty_annotation_warning = True
                                print('Warning: empty annotation found with category {}'.format(
                                    ann['category_id']))
                        continue
                    
                # ...if this is an empty annotation
                
                coco_bbox = ann['bbox']
                yolo_category_id = coco_id_to_yolo_id[ann['category_id']]
                
                # COCO: [x_min, y_min, width, height] in absolute coordinates
                # YOLO: [class, x_center, y_center, width, height] in normalized coordinates
                
                # Convert from COCO coordinates to YOLO coordinates
                img_w = im['width']
                img_h = im['height']
                                
                if source_format == 'coco':
                    
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
                    
                else:
                    
                    raise ValueError('Unrecognized source format {}'.format(source_format))
                
                if clip_boxes:
                    
                    clipped_box = False
                    
                    box_right = x_center_relative + (box_w_relative / 2.0)                    
                    if box_right > 1.0:
                        clipped_box = True
                        overhang = box_right - 1.0
                        box_w_relative -= overhang
                        x_center_relative -= (overhang / 2.0)

                    box_bottom = y_center_relative + (box_h_relative / 2.0)                                        
                    if box_bottom > 1.0:
                        clipped_box = True
                        overhang = box_bottom - 1.0
                        box_h_relative -= overhang
                        y_center_relative -= (overhang / 2.0)
                    
                    box_left = x_center_relative - (box_w_relative / 2.0)
                    if box_left < 0.0:
                        clipped_box = True
                        overhang = abs(box_left)
                        box_w_relative -= overhang
                        x_center_relative += (overhang / 2.0)
                        
                    box_top = y_center_relative - (box_h_relative / 2.0)
                    if box_top < 0.0:
                        clipped_box = True
                        overhang = abs(box_top)
                        box_h_relative -= overhang
                        y_center_relative += (overhang / 2.0)
                        
                    if clipped_box:
                        print('Warning: clipped box for image {}'.format(image_id))
                
                yolo_box = [yolo_category_id,
                            x_center_relative, y_center_relative, 
                            box_w_relative, box_h_relative]
                
                image_bboxes.append(yolo_box)
                
            # ...for each annotation 
            
        # ...if this image has annotations
        
        output_info['bboxes'] = image_bboxes
        
        images_to_copy.append(output_info)        
    
    # ...for each image
        
    print('{} missing images (of {})'.format(len(missing_images),len(data['images'])))
    
    
    # Write output
    
    print('Generating class list')
    
    class_list_filename = os.path.join(output_folder,class_file_name)
    with open(class_list_filename, 'w') as f:
        print('Writing class list to {}'.format(class_list_filename))
        for i_class in range(0,len(yolo_id_to_name)):
            # Category IDs should range from 0..N-1
            assert i_class in yolo_id_to_name
            f.write(yolo_id_to_name[i_class] + '\n')
    
    if image_id_to_output_image_json_file is not None:
        print('Writing image ID mapping to {}'.format(image_id_to_output_image_json_file))
        with open(image_id_to_output_image_json_file,'w') as f:
            json.dump(image_id_to_output_image_name,f,indent=1)
            
    print('Copying images and creating annotation files')
    
    if create_image_and_label_folders:
        dest_image_folder = os.path.join(output_folder,'images')
        dest_txt_folder = os.path.join(output_folder,'labels')
    else:
        dest_image_folder = output_folder
        dest_txt_folder = output_folder
        
    # TODO: parallelize this loop
    #
    # output_info = images_to_copy[0]
    for output_info in tqdm(images_to_copy):

        source_image = output_info['source_image']
        dest_image_relative = output_info['dest_image_relative']
        dest_txt_relative = output_info['dest_txt_relative']
        
        dest_image = os.path.join(dest_image_folder,dest_image_relative)
        os.makedirs(os.path.dirname(dest_image),exist_ok=True)
        
        dest_txt = os.path.join(dest_txt_folder,dest_txt_relative)
        os.makedirs(os.path.dirname(dest_txt),exist_ok=True)
        
        if not create_image_and_label_folders:
            assert os.path.dirname(dest_image) == os.path.dirname(dest_txt)
        
        if (not os.path.isfile(dest_image)) or (overwrite_images):
            shutil.copyfile(source_image,dest_image)
        
        bboxes = output_info['bboxes']        
        
        # Only write an annotation file if there are bounding boxes.  Images with 
        # no .txt files are treated as hard negatives, at least by YOLOv5:
        #
        # https://github.com/ultralytics/yolov5/issues/3218
        #
        # I think this is also true for images with empty annotation files, but 
        # I'm using the convention suggested on that issue, i.e. hard negatives 
        # are expressed as images without .txt files.
        if len(bboxes) > 0:
            
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

    #%% CCT data
    
    input_image_folder = os.path.expanduser('~/data/noaa-fish/JPEGImages')
    output_folder = os.path.expanduser('~/data/noaa-fish/AllImagesWithAnnotations')
    input_file = os.path.expanduser('~/data/noaa-fish/noaa_estuary_fish.json')

    # If preview_export is True, I'm exporting to preview these with BoundingBoxEditor:
    #
    # https://github.com/mfl28/BoundingBoxEditor
    #
    # This export will be compatible, other than the fact that you need to move
    # "object.data" into the "labels" folder.
    #
    # Otherwise I'm exporting for training, in the YOLOv5 flat format.
    preview_export = False
    
    if preview_export:
        
        coco_to_yolo(input_image_folder,output_folder,input_file,
                     source_format='coco',
                     overwrite_images=False,
                     create_image_and_label_folders=True,
                     class_file_name='object.data',
                     allow_empty_annotations=True,
                     clip_boxes=True)
        
    else:
        
        coco_to_yolo(input_image_folder,output_folder,input_file,
                     source_format='coco',
                     overwrite_images=False,
                     create_image_and_label_folders=False,
                     class_file_name='classes.txt',
                     allow_empty_annotations=True,
                     clip_boxes=True,
                     image_id_to_output_image_json_file=\
                         os.path.join(output_folder,'image_id_to_output_image_name.json'))
    
    

#%% Command-line driver

# TODO

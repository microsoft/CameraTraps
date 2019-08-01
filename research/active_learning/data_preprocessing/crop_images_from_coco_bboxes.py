'''
Produces a directory of crops from a COCO-annotated .json full of 
bboxes.
'''
import numpy as np
import argparse, ast, csv, json, pickle, os, sys, time, tqdm, uuid
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True, type=str, help='Path to a directory containing full-sized images in the dataset.')
    parser.add_argument('--coco_json', required=True, type=str, help='Path to COCO JSON file for the dataset.')
    parser.add_argument('--crop_dir', required=True, type=str, help='Path to output directory for crops.')
    parser.add_argument('--padding_factor', type=float, default=1.3, help='We will crop a tight square box around the animal enlarged by this factor. ' + \
                   'Default is 1.3 * 1.3 = 1.69, which accounts for the cropping at test time and for' + \
                    ' a reasonable amount of context')
    args = parser.parse_args()
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
    
    IMAGE_DIR = args.image_dir
    COCO_JSON = args.coco_json
    CROP_DIR = args.crop_dir
    PADDING_FACTOR = args.padding_factor
    
    crops_json = {}

    coco_json_data = json.load(open(COCO_JSON, 'r'))
    images = {im['id']:im for im in coco_json_data['images']}
    bboxes_available = any([('bbox' in a.keys()) for a in coco_json_data['annotations']])
    assert bboxes_available, 'COCO JSON does not contain bounding boxes, need to run a detector first.'
    
    crop_counter = 0
    timer = time.time()
    for ann in coco_json_data['annotations']:
        if 'bbox' not in ann.keys():
            continue
        image_id = ann['image_id']
        image_fn = images[image_id]['file_name'].replace('\\', '/')
        img = np.array(Image.open(os.path.join(IMAGE_DIR, image_fn)))
        if img.dtype != np.uint8:
            print('Failed to load image '+ image_fn)
            continue
        crop_counter += 1

        image_height = images[image_id]['height']
        image_width = images[image_id]['width']
        image_grayscale = bool(np.all(abs(np.mean(img[:,:,0]) - np.mean(img[:,:,1])) < 1) & (abs(np.mean(img[:,:,1]) - np.mean(img[:,:,2])) < 1))
        image_seq_id = images[image_id]['seq_id']
        image_seq_num_frames = images[image_id]['seq_num_frames']
        image_frame_num = images[image_id]['frame_num']

        detection_box_pix = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]
        detection_box_size = np.vstack([detection_box_pix[2] - detection_box_pix[0], detection_box_pix[3] - detection_box_pix[1]]).T
        offsets = (PADDING_FACTOR*np.max(detection_box_size, keepdims=True) - detection_box_size)/2
        crop_box_pix = detection_box_pix + np.hstack([-offsets,offsets])
        crop_box_pix = np.maximum(0,crop_box_pix).astype(int)
        crop_box_pix = crop_box_pix[0]
        detection_padded_cropped_img = img[crop_box_pix[1]:crop_box_pix[3], crop_box_pix[0]:crop_box_pix[2]]
        
        crop_id = str(uuid.uuid4())
        crop_fn = os.path.join(CROP_DIR, crop_id + '.JPG')
        crop_width = int(detection_padded_cropped_img.shape[1])
        crop_height = int(detection_padded_cropped_img.shape[0])
        crop_rel_size = (crop_width*crop_height)/(image_width*image_height)
        detection_conf = 1 # for annotated crops, assign confidence of 1

        Image.fromarray(detection_padded_cropped_img).save(crop_fn)
        crops_json[crop_id] = {'id': crop_id, 'file_name': crop_fn,
                'width': crop_width, 'height':crop_height,
                'grayscale': image_grayscale, 'relative_size': crop_rel_size,
                'source_file_name': image_fn, 'seq_id': image_seq_id, 'seq_num_frames': image_seq_num_frames, 'frame_num': image_frame_num,
                'bbox_confidence': detection_conf, 'bbox_X1': int(crop_box_pix[1]), 'bbox_Y1': int(crop_box_pix[0]),
                'bbox_X2': int(crop_box_pix[3]), 'bbox_Y2': int(crop_box_pix[2])}
        
        if crop_counter%100 == 0:
            print('Produced crops for %d out of %d detections in %0.2f seconds.'%(crop_counter, len(coco_json_data['annotations']), time.time() - timer))
            assert 2==3, 'brek brek'
            
        with open(os.path.join(CROP_DIR, 'crops.json'), 'w') as outfile:
            json.dump(crops_json, outfile)
        

if __name__ == '__main__':
    main()
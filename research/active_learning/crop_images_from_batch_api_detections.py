'''
crop_images_from_batch_api_detections.py

Uses detections obtained using CameraTraps/detection/run_tf_detector_batch.py to produce cropped images for classification. The batch detector saves
detections in a .csv file that is used as input for making crops, which are saved to a specified crop directory. Data about the crops is also stored
in a .json file (e.g. information about the source image, information about the bounding box used to generate the crop, etc.).
'''

import numpy as np
import argparse, ast, csv, json, pickle, os, sys, time, tqdm, uuid
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_output', type=str, required=True,
                        help='Path to a .csv file with the output of run_tf_detector_batch.py')
    parser.add_argument('--crop_dir', type=str, required=True,
                    help='Output directory to save cropped image files.')
    parser.add_argument('--padding_factor', type=float, default=1.3*1.3,
                    help='We will crop a tight square box around the animal enlarged by this factor. ' + \
                   'Default is 1.3 * 1.3 = 1.69, which accounts for the cropping at test time and for' + \
                    ' a reasonable amount of context')
    

    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()
    PADDING_FACTOR = args.padding_factor
    
    # store info about the crops produced in a JSON file
    crops_json = {}

    with open(args.detector_output, 'r') as f:
        counter = 0
        timer = time.time()
        reader = csv.reader(f)
        headers = next(reader, None)
        data = list(reader)
        num_images = sum(1 for row in data)
        for row in data:
            counter += 1
            imgfile = row[0]
            maxconf = float(row[1])
            detections = ast.literal_eval(row[2])

            # get some information about the source image
            img = np.array(Image.open(imgfile))
            if img.dtype != np.uint8:
                print('Failed to load image '+ imgfile)
                continue
            imgwidth = img.shape[1]
            imgheight = img.shape[0]
            imggrayscale = bool(np.all(abs(np.mean(img[:,:,0]) - np.mean(img[:,:,1])) < 1) & (abs(np.mean(img[:,:,1]) - np.mean(img[:,:,2])) < 1))

            # get info about sequence the source image belongs to from path and directory
            imgframenum = int(os.path.basename(imgfile).split('.JPG')[0].split('_')[-1])
            imgseqid = os.path.split(os.path.dirname(imgfile))[-1]
            imgseqnumframes = len([name for name in os.listdir(os.path.dirname(imgfile)) if os.path.isfile(os.path.join(os.path.dirname(imgfile), name))])
            
            for box_id in range(len(detections)):
                if detections[box_id][5] != 1: # something besides an animal was detected (vehicle, person)
                    continue
                detection_box_pix = detections[box_id][0:4]*np.tile([imgheight, imgwidth], (1,2))
                detection_box_pix = detection_box_pix[0]
                detection_box_size = np.vstack([detection_box_pix[2] - detection_box_pix[0], detection_box_pix[3] - detection_box_pix[1]]).T
                
                offsets = (PADDING_FACTOR*np.max(detection_box_size, keepdims=True) - detection_box_size)/2
                crop_box_pix = detection_box_pix + np.hstack([-offsets,offsets])
                crop_box_pix = np.maximum(0,crop_box_pix).astype(int)
                crop_box_pix = crop_box_pix[0]
                
                detection_padded_cropped_img = img[crop_box_pix[0]:crop_box_pix[2], crop_box_pix[1]:crop_box_pix[3]]
                crop_data = []
                crop_id = str(uuid.uuid4())
                crop_fn = args.crop_dir + crop_id + '.JPG'
                crop_rel_size = (crop_box_pix[2] - crop_box_pix[0])*(crop_box_pix[3] - crop_box_pix[1])/(imgwidth*imgheight)
                
                try:
                    Image.fromarray(detection_padded_cropped_img).save(crop_fn)
                    crops_json[crop_id] = {'id': crop_id, 'file_name': crop_fn,
                     'width': detection_padded_cropped_img.shape[1], 'height':detection_padded_cropped_img.shape[0],
                     'grayscale': imggrayscale, 'relative_size': crop_rel_size,
                     'source_file_name': imgfile, 'seq_id': imgseqid, 'seq_num_frames': imgseqnumframes, 'frame_num': imgframenum,
                     'bbox_confidence': detections[box_id][4], 'bbox_X1': detections[box_id][0], 'bbox_Y1': detections[box_id][1],
                     'bbox_X2': detections[box_id][2], 'bbox_Y2': detections[box_id][3]}
                except ValueError:
                    continue
                except FileNotFoundError:
                    continue
            
            if counter%100 == 0:
                print('Processed crops for %d out of %d images in %0.2f seconds'%(counter, num_images, time.time() - timer))

            
    with open(args.crop_dir+'crops.json', 'w') as outfile:
        json.dump(crops_json, outfile)
            


if __name__ == '__main__':
    main()
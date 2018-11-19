#
# remove_corrupted_images_from_database.py
#
# Given a coco-camera-traps .json file, checks all images for TF-friendliness and generates
# a new .json file that only contains the non-corrupted images.
#

import json
import tensorflow as tf
import argparse

def remove_corrupted_images_from_database(data, image_file_root):

    keep_im = {im['id']:True for im in data['images']}

    with tf.Session() as sess:
        count = 0
        for im in data['images']:
            if count % 1000 == 0:
                print('Processed ' + str(count) + ' images')
            count += 1
            image_file = image_file_root + im['file_name']
        
            image_data = tf.gfile.FastGFile(image_file,'r').read()
            try:
                #print('decoding '+im['file_name'])
                image = tf.image.decode_jpeg(image_data)
                #print('running file')
                sess.run(image)
            except:
                #print('image corrupted')
                keep_im[im['id']] = False

    data['images'] = [im for im in data['images'] if keep_im[im['id']]]
    data['annotations'] = [ann for ann in data['annotations'] if keep_im[ann['image_id']]]

    return data


def parse_args():
    parser = argparse.ArgumentParser(description = 'Convert a multiclass .json to a oneclass .json')

    parser.add_argument('--input_file', dest='input_file',
                         help='Path to .json database that includes corrupted jpegs',
                         type=str, required=True)
    parser.add_argument('--image_file_root', dest='image_file_root',
                         help='Path to image files',
                         type=str, required=True)
    parser.add_argument('--output_file', dest='output_file',
                         help='Path to store uncorrupted .json database',
                         type=str, required=True)

    args = parser.parse_args()
    return args
 

def main():
    args = parse_args()
    print('Reading input file')
    with open(args.input_file,'r') as f:
        data = json.load(f)
    print('Removing corrupted images from database')
    uncorrupted_data = remove_corrupted_images_from_database(data, args.image_file_root)

    json.dump(uncorrupted_data, open(args.output_file,'w'))


if __name__ == '__main__':
    main()




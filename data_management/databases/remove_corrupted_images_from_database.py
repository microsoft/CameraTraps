#
# remove_corrupted_images_from_database.py
#
# Given a coco-camera-traps .json file, checks all images for TF-friendliness and generates
# a new .json file that only contains the non-corrupted images.
#

#%% Imports and constants

import argparse
# import multiprocessing
import gc
import json
import os
import time
from multiprocessing.pool import ThreadPool

import humanfriendly
import numpy as np
import tensorflow as tf

N_THREADS = 16 # 1 # multiprocessing.cpu_count()
DEBUG_MAX_IMAGES = -1

# I leave this at an annoying low number, since by definition weird stuff will
# be happening in the TF kernel, and it's useful to keep having content in the console.
IMAGE_PRINT_FREQUENCY = 10


#%% Function definitions

def check_images(images, image_file_root):    
    ''' 
    Checks all the images in [images] for corruption using TF.
    
    [images] is a list of image dictionaries, as they would appear in COCO
    files.
    
    Returns a dictionary mapping image IDs to booleans. 
    '''    
    
    # I sometimes pass in a list of images, sometimes a dict with a single
    # element mapping a job ID to the list of images
    if isinstance(images,dict):
        assert(len(images) == 1)
        jobID = list(images.keys())[0]
        images = images[jobID]
    else:
        jobID = 0
        
    keep_im = {im['id']:True for im in images}
            
    count = 0
    nImages = len(images)
    
    # We're about to start a lot of TF sessions, and we don't want gobs 
    # of debugging information printing out for every session.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    config=tf.ConfigProto(log_device_placement=False)
    
    # At some point we were creating a single session and looping over images
    # within that session, but the only way I found to reliably not run out
    # of GPU memory was to create a session per image and gc.collect() after
    # each session.
    for iImage,im in enumerate(images):

        with tf.Session(config=config) as sess:

            if ((DEBUG_MAX_IMAGES > 0) and (iImage >= DEBUG_MAX_IMAGES)):
                print('Breaking after {} images'.format(DEBUG_MAX_IMAGES))
                break
            
            if (count % IMAGE_PRINT_FREQUENCY == 0):
                print('Job {}: processed {} of {} images'.format(jobID,count,nImages))
                
            count += 1
            image_file = os.path.join(image_file_root,im['file_name'])
            assert(os.path.isfile(image_file))
            
            try:
                image_data = tf.gfile.FastGFile(image_file,'rb').read()
                image = tf.image.decode_jpeg(image_data)
                sess.run(image)
            except:
                keep_im[im['id']] = False

        gc.collect()
        
    return keep_im


def remove_corrupted_images_from_database(data, image_file_root):
    '''
    Given the COCO database [data], checks all images for corruption using
    TF, and returns a subset of [data] containing only non-corrupted images.
    '''
    
    # Map Image IDs to boolean (should I keep this image?)
    images = data['images']
    
    if (N_THREADS == 1):
        
        keep_im = check_images(images,image_file_root)
        
    else:
        
        start = time.time()
        imageChunks = np.array_split(images,N_THREADS)
        # Convert to lists, append job numbers to the image lists
        for iChunk in range(0,len(imageChunks)):
            imageChunks[iChunk] = list(imageChunks[iChunk])
            imageChunks[iChunk] = {iChunk:imageChunks[iChunk]}
        pool = ThreadPool(N_THREADS)
        # results = pool.imap_unordered(lambda x: fetch_url(x,nImages), indexedUrlList)
        results = pool.map(lambda x: check_images(x,image_file_root), imageChunks)
        processingTime = time.time() - start
        
        # Merge results
        keep_im = {}
        for d in results:
            keep_im.update(d)
        bValid = keep_im.values()
            
        print("Checked image corruption in {}, found {} invalid images (of {})".format(
                humanfriendly.format_timespan(processingTime),
                len(bValid)-sum(bValid),len(bValid)))
        
    data['images'] = [im for im in data['images'] if keep_im[im['id']]]
    data['annotations'] = [ann for ann in data['annotations'] if keep_im[ann['image_id']]]
    
    return data


#%% Interactive driver

if False:    
    
    #%%
    
    # base_dir = r'D:\temp\snapshot_serengeti_tfrecord_generation'
    base_dir = r'/data/ss_corruption_check'
    input_file = os.path.join(base_dir,'imerit_batch7_renamed.json')
    output_file = os.path.join(base_dir,'imerit_batch7_renamed_uncorrupted.json')
    image_file_root = os.path.join(base_dir,'imerit_batch7_images_renamed')
    assert(os.path.isfile(input_file))
    assert(os.path.isdir(image_file_root))

    # Load annotations
    with open(input_file,'r') as f:
            data = json.load(f)    
            
    # Check for corruption
    data_uncorrupted = remove_corrupted_images_from_database(data,image_file_root)
    
    # Write out only the uncorrupted data
    json.dump(data_uncorrupted, open(output_file,'w'))
    
    
#%% Command-line driver

def parse_args():
    
    parser = argparse.ArgumentParser(description = 'Remove images from a .json file that can''t be opened in TF')

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




#
# Given a json-formatted list of image filenames, retrieve the width and height of every image.
#

#%% Constants and imports

import argparse
import json
import os
from PIL import Image
import sys

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from functools import partial
from tqdm import tqdm

image_base = ''
default_n_threads = 1
use_threads = False


#%% Processing functions

def process_image(image_path,image_prefix=None):
    
    if image_prefix is not None:
        full_path = os.path.join(image_prefix,image_path)
    else:
        full_path = image_path
    
    # Is this image on disk?
    if not os.path.isfile(full_path):
        print('Could not find image {}'.format(full_path))
        return (image_path,-1,-1)

    try:        
        pil_im = Image.open(full_path)
        w = pil_im.width            
        h = pil_im.height
        return (image_path,w,h)
    except Exception as e:    
        print('Error reading image {}: {}'.format(full_path,str(e)))
        return (image_path,-1,-1)
    
def process_images(filenames,image_prefix=None,n_threads=default_n_threads):
    
    if n_threads <= 1:
        
        all_results = []
        for i_file,fn in tqdm(enumerate(filenames),total=len(filenames)):
            all_results.append(process_image(fn,image_prefix=image_prefix))
    
    else:
        
        print('Creating a pool with {} threads'.format(n_threads))
        if use_threads:
            pool = ThreadPool(n_threads)        
        else:
            pool = Pool(n_threads)
        # all_results = list(tqdm(pool.imap(process_image, filenames), total=len(filenames)))
        all_results = list(tqdm(pool.imap(partial(process_image,image_prefix=image_prefix), filenames), total=len(filenames)))
                
    return all_results


def process_list_file(input_file,output_file,image_prefix=None,n_threads=default_n_threads):
    
    assert os.path.isdir(os.path.dirname(output_file))
    assert os.path.isfile(input_file)
    
    with open(input_file,'r') as f:        
        filenames = json.load(f)
    filenames = [s.strip() for s in filenames]
    
    all_results = process_images(filenames,image_prefix=image_prefix,n_threads=n_threads)
    
    with open(output_file,'w') as f:
        json.dump(all_results,f,indent=2)
    
    
#%% Interactive driver

if False:

    pass    

    #%%
    
    # List images in a test folder
    base_dir = r'c:\temp\test_images'
    image_list_file = os.path.join(base_dir,'images.json')
    relative_image_list_file = os.path.join(base_dir,'images_relative.json')
    image_size_file = os.path.join(base_dir,'image_sizes.json')
    import path_utils
    image_names = path_utils.find_images(base_dir,recursive=True)
    
    with open(image_list_file,'w') as f:
        json.dump(image_names,f,indent=2)
        
    relative_image_names = []
    for s in image_names:
        relative_image_names.append(os.path.relpath(s,base_dir))
    
    with open(relative_image_list_file,'w') as f:
        json.dump(relative_image_names,f,indent=2)
    
    
    #%%
    
    # process_list_file(image_list_file,image_size_file,image_prefix=base_dir)
    process_list_file(relative_image_list_file,image_size_file,image_prefix=base_dir,n_threads=4)
    
    
#%% Command-line driver
    
def main():
    
    # python sanity_check_json_db.py "e:\wildlife_data\wellington_data\wellington_camera_traps.json" --baseDir "e:\wildlife_data\wellington_data\images" --bFindUnusedImages --bCheckImageSizes
    # python sanity_check_json_db.py "D:/wildlife_data/mcgill_test/mcgill_test.json" --baseDir "D:/wildlife_data/mcgill_test" --bFindUnusedImages --bCheckImageSizes
    
    # Here the '-u' prevents buffering, which makes tee happier
    #
    # python -u sanity_check_json_db.py '/datadrive1/nacti_metadata.json' --baseDir '/datadrive1/nactiUnzip/' --bFindUnusedImages --bCheckImageSizes | tee ~/nactiTest.out
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',type=str)
    parser.add_argument('output_file',type=str)
    parser.add_argument('--image_prefix', type=str, default=None)
    parser.add_argument('--n_threads', type=int, default=default_n_threads)
                        
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
        
    args = parser.parse_args()
    
    process_list_file(args.input_file,args.output_file,args.image_prefix,args.n_threads)
    

if __name__ == '__main__':
    
    main()

#
# tile_images.py
#
# Split a folder of images into tiles.  Preserves relative folder structure in a
# new output folder, with a/b/c/d.jpg becoming, e.g.:
# 
# a/b/c/d_row_0_col_0.jpg    
# a/b/c/d_row_0_col_1.jpg
#

#%% Imports and constants

import os

from PIL import Image
from tqdm import tqdm

# from ai4eutils
import path_utils


#%% Main function

def split_image_folder(input_folder,output_folder,n_rows=2,n_cols=2,overwrite=False):
    
    image_full_paths = path_utils.find_images(input_folder,recursive=True)
    image_relative_paths = [os.path.relpath(fn,input_folder) for fn in image_full_paths]
    os.makedirs(output_folder,exist_ok=True)
    
    # TODO: parallelization
    #
    # i_fn = 2; relative_fn = image_relative_paths[i_fn]
    for i_fn,relative_fn in tqdm(enumerate(image_relative_paths),total=len(image_relative_paths)):
        
        input_fn = os.path.join(input_folder,relative_fn)

        # Can we skip this image because we've already generated all the tiles?
        if overwrite:
            skip_image = False
        else:
            
            skip_image = True
            
            for i_col in range(0, n_cols):            
                
                for i_row in range(0, n_rows):
                    
                    # TODO: super-sloppy that I'm pasting this code from below
                    tokens = os.path.splitext(relative_fn)
                    base_fn = tokens[0]; ext = tokens[1]
                    output_relative_fn = base_fn + '_row_{}_col_{}'.format(i_row,i_col) + ext
                    output_path = os.path.join(output_folder,output_relative_fn)
                    
                    if not os.path.isfile(output_path):
                        skip_image = False
                        break
                    
                if not skip_image:
                    break
                
            if skip_image:
                print('Skipping {}, all images generated'.format(relative_fn))
                continue
                
        # From:
        #
        # https://github.com/whiplashoo/split-image/blob/main/src/split_image/split.py
        try:            
            im = Image.open(input_fn)
        except Exception as e:
            print('Error opening {}:\n{}'.format(input_fn,str(e)))
            continue
        
        im_width, im_height = im.size
                
        tile_width = int(im_width / n_cols)
        tile_height = int(im_height / n_rows)
        
        # i_col = 0; i_row = 1
        for i_col in range(0, n_cols): 
            
            for i_row in range(0, n_rows):
                
                try:
                    
                    # left/top/right/bottom
                    l = i_col*tile_width
                    r = ((i_col+1)*tile_width)-1
                    t = i_row*tile_height
                    b = ((i_row+1)*tile_height)-1
                    box = (l,t,r,b)
                    
                    outp = im.crop(box)
                    tokens = os.path.splitext(relative_fn)
                    base_fn = tokens[0]; ext = tokens[1]
                    output_relative_fn = base_fn + '_row_{}_col_{}'.format(i_row,i_col) + ext
                    output_path = os.path.join(output_folder,output_relative_fn)
                    os.makedirs(os.path.dirname(output_path),exist_ok=True)
                    outp.save(output_path,quality=95,subsampling=0)
                
                except Exception:
                    
                    print('Warning: tiling error at {}/{}/{}'.format(i_fn,i_col,i_row))
                    
            # ...for each row
        
        # ...for each column

    # ...for each image
    
    
#%% Interactive driver

if False:

    pass

    #%%
    
    input_folder = '/datadrive/home/sftp/organization/data'
    output_folder = '/datadrive/home/sftp/organization/data_tiled'
    n_rows = 2
    n_cols = 2
        
    #%%
    split_image_folder(input_folder,output_folder,n_rows,n_cols)
#%% Imports and constants

import os
import pandas as pd
from tqdm import tqdm

# from github.com/microsoft/ai4eutils
import url_utils

# from github.com/ecologize/CameraTraps
from visualization import visualization_utils

# A list of files in the lilablobssc container for this data set
container_file_list = r'C:\temp\seals\seal_files.txt'

# The raw detection files provided by NOAA
detections_fn = r'C:\temp\seals\surv_test_kamera_detections_20210212.csv'

# A version of the above with filename columns added
detections_fn_full_paths = detections_fn.replace('.csv','_full_paths.csv')

base_url = 'https://lilablobssc.blob.core.windows.net/noaa-kotz'


#%% Read input .csv

df = pd.read_csv(detections_fn)
df['rgb_image_path'] = ''
df['ir_image_path'] = ''
print('Read {} rows from {}'.format(len(df),detections_fn))

camera_view_to_path = {}
camera_view_to_path['C'] = 'CENT'
camera_view_to_path['L'] = 'LEFT'

valid_flights = set(['fl04','fl05','fl06','fl07'])


#%% Read list of files

with open(container_file_list,'r') as f:
    all_files = f.readlines()
all_files = [s.strip() for s in all_files]
all_files = set(all_files)


#%% Convert paths to full paths

missing_ir_files = []

# i_row = 0; row = df.iloc[i_row]
for i_row,row in tqdm(df.iterrows(),total=len(df)):
    
    assert row['flight'] in valid_flights
    assert row['camera_view'] in camera_view_to_path
    
    assert isinstance(row['rgb_image_name'],str)
    rgb_image_path = 'Images/{}/{}/{}'.format(row['flight'],camera_view_to_path[row['camera_view']],
                                     row['rgb_image_name'])
    assert rgb_image_path in all_files
    df.loc[i_row,'rgb_image_path'] = rgb_image_path
    
    if not isinstance(row['ir_image_name'],str):
        continue
    
    ir_image_path = 'Images/{}/{}/{}'.format(row['flight'],camera_view_to_path[row['camera_view']],
                                     row['ir_image_name'])
    # assert ir_image_path in all_files
    if ir_image_path not in all_files:
        missing_ir_files.append(ir_image_path)
    df.loc[i_row,'ir_image_path'] = ir_image_path
            
# ...for each row    

missing_ir_files = list(set(missing_ir_files))
missing_ir_files.sort()
print('{} missing IR files (of {})'.format(len(missing_ir_files),len(df)))

for s in missing_ir_files:
    print(s)


#%% Write results

df.to_csv(detections_fn_full_paths,index=False)


#%% Load output file, just to be sure

df = pd.read_csv(detections_fn_full_paths)


#%% Render annotations on an image

import random; i_image = random.randint(0,len(df))
# i_image = 2004 
row = df.iloc[i_image]
rgb_image_path = row['rgb_image_path']
rgb_image_url = base_url + '/' + rgb_image_path
ir_image_path = row['ir_image_path']
ir_image_url = base_url + '/' + ir_image_path


#%% Download the image

rgb_image_fn = url_utils.download_url(rgb_image_url,progress_updater=True)
ir_image_fn = url_utils.download_url(ir_image_url,progress_updater=True)


#%% Find all the rows (detections) associated with this image

# as l,r,t,b
rgb_boxes = []
ir_boxes = []

for i_row,row in df.iterrows():
    
    if row['rgb_image_path'] == rgb_image_path:
        box_l = row['rgb_left']
        box_r = row['rgb_right']
        box_t = row['rgb_top']
        box_b = row['rgb_bottom']
        rgb_boxes.append([box_l,box_r,box_t,box_b])

    if row['ir_image_path'] == ir_image_path:
        box_l = row['ir_left']
        box_r = row['ir_right']
        box_t = row['ir_top']
        box_b = row['ir_bottom']
        ir_boxes.append([box_l,box_r,box_t,box_b])
        
print('Found {} RGB, {} IR annotations for this image'.format(len(rgb_boxes),
                                                              len(ir_boxes))) 


#%% Render the detections on the image(s)

img_rgb = visualization_utils.load_image(rgb_image_fn)
img_ir = visualization_utils.load_image(ir_image_fn)

for b in rgb_boxes:
    
    # In pixel coordinates
    box_left = b[0]; box_right = b[1]; box_top = b[2]; box_bottom = b[3]    
    assert box_top > box_bottom; assert box_right > box_left    
    ymin = box_bottom; ymax = box_top; xmin = box_left; xmax = box_right

    visualization_utils.draw_bounding_box_on_image(img_rgb,ymin,xmin,ymax,xmax,
                                                   use_normalized_coordinates=False,
                                                   thickness=3)

for b in ir_boxes:
    
    # In pixel coordinates
    box_left = b[0]; box_right = b[1]; box_top = b[2]; box_bottom = b[3]    
    assert box_top > box_bottom; assert box_right > box_left    
    ymin = box_bottom; ymax = box_top; xmin = box_left; xmax = box_right

    visualization_utils.draw_bounding_box_on_image(img_ir,ymin,xmin,ymax,xmax,
                                                   use_normalized_coordinates=False,
                                                   thickness=3)

visualization_utils.show_images_in_a_row([img_rgb,img_ir])


#%% Save images

img_rgb.save(r'c:\temp\seals_rgb.png')
img_ir.save(r'c:\temp\seals_ir.png')


#%% Clean up

import shutil
tmp_dir = os.path.dirname(rgb_image_fn)
assert 'ai4eutils' in tmp_dir
shutil.rmtree(tmp_dir)

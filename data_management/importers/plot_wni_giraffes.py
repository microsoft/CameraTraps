#
# plot_wni_giraffes.py
#
# Plot keypoints on a random sample of images from the wni-giraffes data set.
#

#%% Constants and imports

import os
import json
import random

from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm

input_file = r"G:\data_staging\wni-out\wni_giraffes_train.json"
image_base = r"G:\data_staging\wni-out\images"
output_base = r"G:\data_staging\wni-out\test-plots"
os.makedirs(output_base,exist_ok=True)

tool_colors = ['red','green','blue','magenta']
use_fancy_ellipses = True
draw_individual_samples = False

median_radius = 20
median_linewidth = 8

sample_radius = 10

n_images_to_plot = 100


#%% Load and select data

with open(input_file,'r') as f:
    d = json.load(f)
annotations = d['annotations']
print(d['info'])

short_tool_names = list(d['info']['tool_names'].keys())
annotations_to_plot = random.sample(annotations,n_images_to_plot)


#%% Support functions

# https://stackoverflow.com/questions/32504246/draw-ellipse-in-python-pil-with-line-thickness
def draw_fancy_ellipse(image, x, y, radius, width=1, outline='white', antialias=4):

    bounds = (x-radius,y-radius,x+radius,y+radius)
    
    # Use a single channel image (mode='L') as mask.
    # The size of the mask can be increased relative to the imput image
    # to get smoother looking results. 
    mask = Image.new(
        size=[int(dim * antialias) for dim in image.size],
        mode='L', color='black')
    draw = ImageDraw.Draw(mask)

    # draw outer shape in white (color) and inner shape in black (transparent)
    for offset, fill in (width/-2.0, 'white'), (width/2.0, 'black'):
        left, top = [(value + offset) * antialias for value in bounds[:2]]
        right, bottom = [(value - offset) * antialias for value in bounds[2:]]
        draw.ellipse([left, top, right, bottom], fill=fill)

    # downsample the mask using PIL.Image.LANCZOS 
    # (a high-quality downsampling filter).
    mask = mask.resize(image.size, Image.LANCZOS)
    
    # paste outline color to input image through the mask
    image.paste(outline, mask=mask)

    
def draw_ellipse(image, x, y, radius, linewidth, color_index, use_imagedraw=False):
    
    if use_imagedraw:
        draw_fancy_ellipse(image, x, y, radius=radius, width=linewidth, outline=tool_colors[color_index])
    else:
        draw = ImageDraw.Draw(image)
        bounds = (x-radius,y-radius,x+radius,y+radius)
        draw.ellipse(bounds, fill=tool_colors[color_index])
            
        
#%% Plot some images

# ann = annotations_to_plot[0]
for ann in tqdm(annotations_to_plot):
    
    input_path = os.path.join(image_base,ann['filename'])    
    output_path = os.path.join(output_base,ann['filename'].replace('/','_'))

    im = None
    im = Image.open(input_path)
    
    # i_tool = 0; tool_name = short_tool_names[i_tool]
    for i_tool,tool_name in enumerate(short_tool_names):
        
        tool_keypoints = ann['keypoints'][tool_name]
        
        # Don't plot tools that don't have a consensus annotation
        if tool_keypoints['median_x'] is None:
            continue
        
        median_x = tool_keypoints['median_x']
        median_y = tool_keypoints['median_y']
        
        draw_ellipse(im, median_x, median_y, median_radius, median_linewidth, color_index=i_tool, 
                     use_imagedraw=use_fancy_ellipses)
    
        if draw_individual_samples:
            for i_sample in range(0,len(tool_keypoints['x'])):
                x = tool_keypoints['x'][i_sample]
                y = tool_keypoints['y'][i_sample]
                draw_ellipse(im, x, y, sample_radius, None, color_index=i_tool, 
                         use_imagedraw=False)
            
    # ...for each tool
        
    im.save(output_path)

# ...for each annotation

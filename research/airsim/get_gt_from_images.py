from PIL import Image
import json
import os
import re
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

plot_bboxes = False
image_dir = r"C:\Users\t-sabeer\Documents\AirSimImages\\"
environment_file = "environment_lookup.json"
animal_file = 'animal_lookup.json'
json_output = 'airsim_images_batch_1.json'

def read_seg_rgb_file(filename):
    rgb_lookup = {}
    with open(filename) as f:
        for line in f:
            line = ''.join(x for x in line if x not in '[,]')
            words = line.split()
            rgb_lookup[int(words[0])] = [int(x) for x in words[1:]]
    return rgb_lookup


rgb_lookup = read_seg_rgb_file("seg_rgbs.txt")
red_val_to_seg_id = {rgb_lookup[idx][0]:idx for idx in rgb_lookup}

with open(environment_file) as f:
    environment_lookup = json.load(f)

with open(animal_file) as f:
    animal_lookup = json.load(f)

dirs = os.listdir(image_dir)

images = {}
annotations = [] #coco format: x,y,w,h, x,y is top left, coordinated not normalized
ann_count = 0
categories = []
cat_ids = []

count = 0

for file in dirs:
    count +=1 
    if count % 1000 == 0:
        print('Processed ' + str(count) + ' images')
    im_name = file.split(".")[0]
    im_id = im_name[:-2]
    if im_id.split('_')[0] not in environment_lookup:
        continue
    if im_id not in images:
        #print(im_id)
        images[im_id] = {}
        images[im_id]['id'] = im_id
        images[im_id]['seq_id'] = im_name[:-10]
        #print(images[im_id]['seq_id'])
        images[im_id]['seq_num_frames'] = 3
        images[im_id]['frame_num'] = int(im_id[-1])
        images[im_id]['environment'] = environment_lookup[im_id.split('_')[0]]

    if im_name[-1] == str(0):
        images[im_id]['file_name'] = file
        img = np.asarray(Image.open(image_dir + file))
        s = img.shape; h = s[0]; w = s[1]  #TODO: check this isn't backwards
        images[im_id]['width'] = w
        images[im_id]['height'] = h
        if plot_bboxes:
            dpi = 50
            figsize = w / float(dpi), h / float(dpi)
            fig = plt.figure(figsize=figsize)
            ax = plt.axes([0,0,1,1])
            ax.imshow(img)
            ax.set_axis_off()

    if im_name[-1] == str(1):
        img = np.asarray(Image.open(image_dir + file))
        #get unique colors in segmentation
        red = np.unique(img[:,:,0])
        #get box for each color
        for idx in red:
            if red_val_to_seg_id[idx] == 0:
                #this is the background class
                continue
            
            #get a box around this color
            mask = img[:,:,0] == idx
            i, j = np.where(mask)
            # x1,y1 is top left corner, x2,y2 is bottom right corner
            x1 = min(j) - 1
            x2= max(j) + 1
            y1 = min(i) - 1
            y2 = max(i) + 1
            
            if plot_bboxes:
                rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, edgecolor='r', facecolor='none')
                plt.plot(x1,y1,'c*')
                plt.plot(x1,y2,'y*')
                ax.add_patch(rect)
            
            ann = {}
            ann['bbox'] = [int(x1),int(y1),int(x2-x1),int(y2-y1)]
            ann['image_id'] = im_id
            ann['category_id'] = images[im_id]['environment']['AnimalClass']
            ann['id'] = ann_count
            ann_count += 1
            annotations.append(ann)

            if ann['category_id'] not in cat_ids: #need to add class to the category list
                cat_ids.append(ann['category_id'])
                cat = {}
                cat['id'] = ann['category_id']
                cat['name'] = animal_lookup[str(ann['category_id'])]
                categories.append(cat)


        plt.show()

#create coco-style json
ims = list(images.keys())

data = {}
data['images'] = ims
data['annotations'] = annotations
data['categories'] = categories

if not plot_bboxes:
    with open(json_output,'w') as f:
        json.dump(data, f)






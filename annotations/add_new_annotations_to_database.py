# coding: utf-8 -*-

#%% Imports and environment

import os
import json
import glob
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import matplotlib.ticker as ticker
import uuid
import pickle

batch_folder = '/ai4efs/annotations/'
bboxFile = batch_folder+'incoming_annotations/microsoft_batch5_12sep2018.json'
imageBase = batch_folder+'modified_annotations/images/imerit_batch_5/'    
outputFile = batch_folder+'modified_annotations/imerit_batch_5_with_humans.json'
annotationFile = '/ai4efs/databases/snapshotserengeti/SnapshotSerengeti.json'

get_w_h_from_image = False

#os.makedirs(outputBase, exist_ok=True)


#%%  Read all source images and build up a hash table from image name to full path

# This spans training and validation directories, so it's not the same as
# just joining the image name to a base path
print("Loading json database")
with open(annotationFile, 'r') as f:
    data = json.load(f)

if not get_w_h_from_image:
    w_h_db = json.load(open(batch_folder+'modified_annotations/imerit_batch_5.json'))
    im_id_to_w_h = {im['id']:{'width':im['width'],'height':im['height']} for im in w_h_db['images']}

images = data['images']
annotations = data['annotations']
categories = data['categories']
im_id_to_im = {im['id']:im for im in images}
cat_id_to_cat = {cat['id']: cat for cat in categories}
cat_name_to_cat_id = {cat['name']:cat['id'] for cat in categories}
empty_id = cat_name_to_cat_id['empty']
im_id_to_cats = {ann['image_id']:[] for ann in annotations}
for ann in annotations:
    im_id_to_cats[ann['image_id']].append(ann['category_id'])

print(images[0])
print(annotations[0])

print("Reading annotations")

annData = []
annCats = []
with open(bboxFile) as annotationFileHandle:
    for line in annotationFileHandle:
        annData.append(json.loads(line))
     
    # annData has keys:
    #
    # annotations, categories, images
    #        
    # Each of these are lists of dictionaries

print("Creating annotations database")
new_images = []
new_annotations = []
switch_to_empty = []
switch_to_full = 0
image_count = 0
bbox_count = 0
new_id_to_old_id = {}
new_fn_to_old_id = {}
images_not_in_db = []
cats = []
human_images = []
trunc_images = []
ann_images = []

print(annData[0]['categories'])

for sequence in annData:
    sequenceImages = sequence['images']
    if len(sequenceImages) > 10:
        print("Annotation file {} has {} images".format(annotationFile, len(sequenceImages)))
        continue
 
    if (len(sequenceImages) == 0):
        print("Annotation file {} has no images".format(annotationFile))
        continue
    
    sequenceAnnotations = sequence['annotations']
    
            
    #%% Save images and annotations
    
    for im in sequenceImages:
        image_count += 1
        imID = im['id']
        imFileName = im['file_name']
        m = re.findall(r'img(.*\.jpg)$', imFileName, re.M|re.I)
        old_fn = imFileName.split('.')[2][3:]
        old_fn = old_fn.split('_')
        old_id = ''
        old_id_fn = ''
        for chunk in old_fn:
            old_id_fn += chunk + '_'
        for idx in range(len(old_fn[:-1])):
            chunk = old_fn[idx]
            if idx != 2:
                old_id += chunk + '/'
            else:
                old_id += old_fn[idx-1]+'_'+chunk+'/'
        old_id+=old_id_fn[:-1]
        #print(old_id)
        #assert(len(m) == 1)
        new_id_to_old_id[imID] = old_id
        new_fn_to_old_id[imFileName] = old_id
        if old_id in im_id_to_im:
            new_im = im_id_to_im[old_id]
            if 'height' not in new_im:
                if get_w_h_from_image:
                    im_w, im_h = Image.open(imageBase+old_id+'.JPG').size
                else:
                    im_w = im_id_to_w_h[new_im['id']]['width']
                    im_h = im_id_to_w_h[new_im['id']]['height']
                new_im['height'] = im_h
                new_im['width'] = im_w

            new_images.append(new_im)
            im_id_to_im[old_id] = new_im
        else:
            images_not_in_db.append(old_id)
            #print(im, old_id)

    # ...for each image
    for ann in sequenceAnnotations:
        #print(ann)
        if ann['human_visible'] != 0:
             human_images.append(new_fn_to_old_id[ann['image_id']])
       
        if ann['truncation'] != 0:
             #print(ann['truncation'])
             #print(new_fn_to_old_id[ann['image_id']])
             trunc_images.append(new_fn_to_old_id[ann['image_id']])

        if cat_id_to_cat[ann['category_id']]['name'] not in annCats:
            annCats.append(cat_id_to_cat[ann['category_id']]['name'])
        
        new_ann = {}
        new_ann['id'] = str(uuid.uuid1())
        new_ann['image_id'] = new_fn_to_old_id[ann['image_id']]
        ann_images.append(new_ann['image_id'])
        if new_ann['image_id'] in images_not_in_db:
            continue

        ann_im = im_id_to_im[new_ann['image_id']]
        #if cat_id_to_cat[ann['category_id']]['name'] == 'human':
        #    new_ann['category_id'] = ann['category_id']
        new_ann['category_id'] = im_id_to_cats[new_ann['image_id']][0] #need to deal with images with multiple cats later
           
        #new_annotations.append(ann)
        if 'bbox' not in ann and new_ann['category_id'] != empty_id:  
            new_ann['category_id'] = empty_id
            #print("switch to empty")
            switch_to_empty.append(new_ann['image_id'])
        if 'bbox' in ann and new_ann['category_id'] == empty_id:
            #print("switch to full")
            new_ann['category_id'] = 1000
            switch_to_full += 1
        if 'bbox' in ann:
            new_ann['bbox'] = ann['bbox']
            #unnormalize the bbox
            new_ann['bbox'][0] = new_ann['bbox'][0]*ann_im['width']
            new_ann['bbox'][1] = new_ann['bbox'][1]*ann_im['height']
            new_ann['bbox'][2] = new_ann['bbox'][2]*ann_im['width']
            new_ann['bbox'][3] = new_ann['bbox'][3]*ann_im['height']
            bbox_count += 1
        
        if int(ann['category_id']) == 3: #catch group annotations
            new_ann['iscrowd'] = True
        if int(ann['category_id']) == 2: #make sure humans are annotated as humans
            new_ann['category_id'] = cat_name_to_cat_id['human']
        new_annotations.append(new_ann)

    # ... for each annotation 

    if image_count % 100 == 0:
        print('Processed '+str(image_count)+' images')
for im in new_images:
    if im['id'] not in ann_images and im_id_to_cats[im['id']][0] != empty_id:
        switch_to_empty.append(im['id'])

print('Human images: ',len(human_images))
print('Truncated animals: ', len(trunc_images))
categories.append({'id':1000, 'name': 'needs label'})

# ...for each file
print(annCats)
print('Num images: ', image_count)
print('Num bboxs: ', bbox_count)
print('Images not in the database: ', len(images_not_in_db))
print(len(new_images), ' images')
print(len(new_annotations), ' annotations')
print(len(switch_to_empty), ' images switched to empty')
pickle.dump(switch_to_empty, open(batch_folder+'ims_switched_to_empty.p','wb'))
print(switch_to_full, ' images switched to full')
print([(cat['id'], cat['name']) for cat in categories])
print(len([cat['id'] for cat in categories]), len(list(set([cat['id'] for cat in categories]))))
new_data = {}
new_data['images'] = new_images
new_data['categories'] = categories
new_data['annotations'] = new_annotations
new_data['info'] = data['info']
new_data['licenses'] = {}
json.dump(new_data, open(outputFile,'w'))
    


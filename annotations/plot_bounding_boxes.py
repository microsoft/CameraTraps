#
# plot_bounding_boxes.py
#
# Takes a .json database containing bounding boxes and renders those boxes on the 
# source images.
#
# This assumes annotations in coco-camera-traps format, with absolute bbox
# coordinates.
#

#%% Imports and environment

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import matplotlib.ticker as ticker

# Old configurations
BASE_DIR = r'd:\temp\snapshot_serengeti_tfrecord_generation'
annotationFile = os.path.join(BASE_DIR,'imerit_batch7_renamed.json')
outputBase = os.path.join(BASE_DIR,'imerit_batch7_bboxes')
imageBase = os.path.join(BASE_DIR,'imerit_batch7_images_renamed')

if False:
    imageBase = '/datadrive/iwildcam/'    
    outputBase = '/datadrive/iwildcam/imerit/tmp/'
    annotationFile = '/datadrive/iwildcam/annotations/CaltechCameraTrapsFullAnnotations.json'

os.makedirs(outputBase, exist_ok=True)

LINE_WIDTH_HEIGHT_FRACTION = 0.003
FONT_SIZE_HEIGHT_FRACTION = 0.015

        
#%%  Read all source images and build up convenience mappings 

print("Loading json database")
with open(annotationFile, 'r') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']
categories = data['categories']

# Category ID to category name
categoryIDToCategoryName = {cat['id']:cat['name'] for cat in categories}

# Image ID to image info
imageIDToImage = {im['id']:im for im in images}

# Image ID to image path
imageIDToPath = {}

for im in images:
    imageID = im['id']
    imageIDToPath[imageID] = os.path.join(imageBase,im['file_name'])

# Image ID to all annotations referring to this image
imageIDToAnnotations = {}

for ann in annotations:
    
    imageID = ann['image_id']
    assert imageID in imageIDToPath
    assert imageID in imageIDToImage
    
    if imageID in imageIDToAnnotations:
        imageIDToAnnotations[imageID].append(ann)
    else:
        imageIDToAnnotations[imageID] = [ann]

print("Loaded database and built mappings for {} images".format(len(images)))
    

#%% Iterate over images, draw bounding boxes, write to file

# For each image
# image = images[0]
for image in images:
    
    imageID = image['id']
    
    imageAnnotations = imageIDToAnnotations.get(imageID,[])
    
    # Build up a list of bounding boxes to draw on this image
    boundingBoxes = []
        
    imageFileName = imageIDToPath[imageID]
    assert(os.path.isfile(imageFileName))
    
    outputID = imageID.replace('/','_')
    outputFileName = os.path.join(outputBase,outputID +'.jpg')
            
    # Load the image
    sourceImage = Image.open(imageFileName).convert("RGBA")
        
    s = sourceImage.size
    imageWidth = s[0]
    imageHeight = s[1]
    imageNP = np.array(sourceImage, dtype=np.uint8)
    
    dpi = 100
    figsize = imageWidth / float(dpi), imageHeight / float(dpi)
    
    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1])
    
    # Display the image
    ax.imshow(imageNP)
    ax.set_axis_off()    
        
    # ann = imageAnnotations[0]
    
    # For each annotation associated with this image, render bounding box and label
    for ann in imageAnnotations:
        
        boundingBox = ann['bbox']
        
        (x,y,w,h) = boundingBox
            
        # In the Rectangle() function, the first argument ("location") is the bottom-left 
        # of the rectangle.
        #
        # Origin is the upper-left of the image.
        iLeft = x
        iBottom = y
        
        lineWidth = imageHeight * LINE_WIDTH_HEIGHT_FRACTION
        fontSize = FONT_SIZE_HEIGHT_FRACTION * imageHeight
        
        rect = patches.Rectangle((iLeft,iBottom),w,h,linewidth=lineWidth,edgecolor='r',
                                 facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)        
        
        # Add a class label
        categoryID = ann['category_id']
        categoryName = categoryIDToCategoryName[categoryID]
        iTop = iBottom + h
        
        plt.text(x,iTop,categoryName,color='red',fontsize=fontSize,bbox=dict(facecolor='black',alpha=0.5))
        
    # This is magic goop that removes whitespace around image plots (sort of)    
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.axis('tight')
    plt.axis('off')                

    # Write the output image    
    plt.savefig(outputFileName, bbox_inches='tight', pad_inches=0.0)
    plt.close('all')

    

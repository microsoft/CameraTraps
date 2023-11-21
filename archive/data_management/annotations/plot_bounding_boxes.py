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
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import random
import matplotlib.ticker as ticker

BASE_DIR = r'cameraTrapStuff'
annotationFile = os.path.join(BASE_DIR,'annotations.json')
outputBase = os.path.join(BASE_DIR,'output')
imageBase = os.path.join(BASE_DIR,'imageBase')

os.makedirs(outputBase, exist_ok=True)
assert(os.path.isfile(annotationFile))

LINE_WIDTH_HEIGHT_FRACTION = 0.003
FONT_SIZE_HEIGHT_FRACTION = 0.015

# How many images should we process?  Set to -1 to process all images.
MAX_IMAGES = 1000 # -1

# Should we randomize the image order?
SHUFFLE_IMAGES = True
        

#%%  Read database and build up convenience mappings 

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

if (SHUFFLE_IMAGES):
    print('Shuffling image list')
    random.shuffle(images)
    
if (MAX_IMAGES > 0):
    print('Trimming image list to {}'.format(MAX_IMAGES))
    images = images[:MAX_IMAGES]
    
# For each image
# image = images[0]
nImages = len(images)
for iImage in tqdm(range(nImages)):
    
    image = images[iImage]
    
    imageID = image['id']
    
    imageAnnotations = imageIDToAnnotations.get(imageID,[])
    
    # Build up a list of bounding boxes to draw on this image
    boundingBoxes = []
        
    imageFileName = imageIDToPath[imageID]
    assert(os.path.isfile(imageFileName))
    
    outputID = imageID.replace('/','_')
    outputFileName = os.path.join(outputBase,outputID +'.jpg')
            
    # Load the image
    try:
        sourceImage = Image.open(imageFileName).convert("RGBA")
    except:
        print('Warning: image read error on {}'.format(imageFileName))
        continue
    
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

# ...for each image
    
print('Finished rendering boxes')    

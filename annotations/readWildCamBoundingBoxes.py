# -*- coding: utf-8 -*-

#%% Imports and environment

import os
import json
import glob
import re

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import matplotlib.ticker as ticker

bboxBase = r'e:\imageLabeling\iWildCam\responses'
imageBase = r'e:\wildlife_data_sets\iWildCam2018'    
outputBase = r'e:\imageLabeling\tmp'

os.makedirs(outputBase, exist_ok=True)


#%%  Read all source images and build up a hash table from image name to full path

# This spans training and validation directories, so it's not the same as
# just joining the image name to a base path

print("Enumerating images...")

allImages = glob.glob(os.path.join(imageBase,'**/*.jpg'), recursive=True)

imageToPathMappings = {}

for imageFilePath in allImages:
    
    # Pull out the image name
    pathName, fileName = os.path.split(imageFilePath)
    
    imageToPathMappings[fileName] = imageFilePath
    
print("Built path name mappings for {} images".format(len(allImages)))
    

#%% Get all .json files in the annotations directory

allAnnotationFiles = glob.glob(os.path.join(bboxBase,'**/*.json'), recursive=True)
nAnnotationFiles = len(allAnnotationFiles)
annotationFile = allAnnotationFiles[0]

print("Read {} annotation files from {}".format(len(allAnnotationFiles),bboxBase))


#%% Iterate over annotations, draw bounding boxes, write to file

# annotationFile = allAnnotationFiles[0]
for iAnnotationFile,annotationFile in enumerate(allAnnotationFiles):
    
    print("Processing annotation file {} of {}: {}",iAnnotationFile,nAnnotationFiles,annotationFile)
    
    #%%  Read annotations from this file
    
    # os.startfile(annotationFile)

    with open(annotationFile) as annotationFileHandle:
        annData = json.load(annotationFileHandle)
     
    # annData has keys:
    #
    # annotations, categories, image
    #        
    # Each of these are lists of dictionaries
    
    sequenceImages = annData['images']
    
    if (len(sequenceImages) == 0):
        print("Annotation file {} has no images".format(annotationFile))
        continue
    
    sequenceAnnotations = annData['annotations']
    
            
    #%% Render all annotations on each image in the sequence
    
    for img in sequenceImages:
        
        #%% Pull out image metadata
        
        imgID = img['id']
        imgFileName = img['file_name']
        
        # Build up a list of bounding boxes to draw on this image
        boundingBoxes = []
        
        # Pull out just the image name from the filename
        #
        # File names look like:
        #
        # seq6efffac2-5567-11e8-b3fe-dca9047ef277.frame1.img59a94e52-23d2-11e8-a6a3-ec086b02610b.jpg
        m = re.findall(r'img(.*\.jpg)$', imgFileName, re.M|re.I)
        assert(len(m) == 1)
        queryFileName = m[0]
        
        physicalFileName = ''
        
        # Map this image back to the original directory
        try:
            physicalFileName = imageToPathMappings[queryFileName]
        except:
            print("File name {} not found".format(queryFileName))
    
        outputFileName = os.path.join(outputBase,imgFileName)
            
        sourceImage = Image.open(physicalFileName).convert("RGBA")
        
        s = sourceImage.size
        imageWidth = s[0]
        imageHeight = s[1]
        
        
        #%% Loop over annotations, find annotations that match this image
        
        for annotation in sequenceAnnotations:
            
            #%%
            
            annotationImgID = annotation['image_id']
            
            if (imgID != annotationImgID):
                continue
            
            annotationRelative = annotation['bbox']
            
            # x,y,w,h
            #
            # x,y is the bottom-left of the rectangle
            #
            # x,y origin is the upper-left
            xRel = annotationRelative[0]
            yRel = annotationRelative[1]
            wRel = annotationRelative[2]
            hRel = annotationRelative[3]
            
            x = xRel * imageWidth
            y = yRel * imageHeight
            w = wRel * imageWidth
            h = hRel * imageHeight
            
            boundingBoxes.append( (x,y,w,h) )

        # ...for each annotation
        

        #%% Render with PIL (scrap)
        
        if False:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(sourceImage)
            draw.rectangle(((x, y), (x+w, y+h)), outline="red")
            sourceImage.show()        
            sourceImage.save('tmp.jpg', "JPEG")


        #%% Render with Matplotlib
        
        im = np.array(sourceImage, dtype=np.uint8)
        
        # Create figure and axes
        fig = plt.figure()
        ax = plt.axes([0,0,1,1])
        
        # Display the image
        ax.imshow(im)
        ax.set_axis_off()
        
        for boundingBox in boundingBoxes:
            
            (x,y,w,h) = boundingBox
            
            # Location is the bottom-left of the rect
            #
            # Origin is the upper-left
            iLeft = x
            iBottom = y
            rect = patches.Rectangle((iLeft,iBottom),w,h,linewidth=2,edgecolor='r',facecolor='none')
            
            # Add the patch to the Axes
            ax.add_patch(rect)        
        
        # This is magic goop that removes whitespace around image plots (sort of)
        
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.axis('tight')
        plt.axis('off')                

        plt.savefig(outputFileName, bbox_inches='tight', pad_inches=0.0)
        # os.startfile(outputFileName)
        
        
        #%% Showing figures on-screen during debugging
        
        # plt.show()        
        
        # Various (mostly unsuccessful) approaches to getting the plot window to show up
        # in the foreground, which is a backend-specific operation...
        #
        # fig.canvas.manager.window.activateWindow()
        # fig.canvas.manager.window.raise_()
                
        # fm = plt.get_current_fig_manager() 
        # fm.window.attributes('-topmost', 1)
        # fm.window.attributes('-topmost', 0)
        #
        # # This is the one that I found to be most robust... at like 80% robust.
        # plt.get_current_fig_manager().window.raise_()
                
        #%%       
        
        plt.close('all')
        
    # ...for each image

# ...for each file
    
        
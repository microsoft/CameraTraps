#
# ocr_sandbox.py
#
# sandbox for experimenting with using OCR to pull metadata from camera trap images
#

#%% Constants and imports

import os
import pytesseract
import warnings
import glob
import cv2
import numpy as np
import ntpath
import write_html_image_list        
from PIL import Image

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings("ignore", "Metadata warning", UserWarning)

baseDir = r'd:\temp\camera_trap_images_for_metadata_extraction'


#%% Support functions

from PIL.ExifTags import TAGS
def get_exif(img):
    
    ret = {}
    if isinstance(img,str):
        img = Image.open(img)
    info = img._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value
    return ret


#%% Load some images, pull EXIF data

images = []
exifInfo = []

imageFileNames = glob.glob(os.path.join(baseDir,'*.jpg'))

for fn in imageFileNames:
    
    img = Image.open(fn)
    exifInfo.append(get_exif(img))
    images.append(img)
        

#%% Crop images

# This will be a list of two-element lists (image top, image bottom)
imageRegions = []
    
imageCropFraction = [0.025 , 0.045]

# image = images[0]
for image in images:
    
    exif_data = image._getexif()
    h = image.height
    w = image.width
    
    cropHeightTop = round(imageCropFraction[0] * h)
    cropHeightBottom = round(imageCropFraction[1] * h)
    
    # l,t,r,b
    #
    # 0,0 is upper-left
    topCrop = image.crop([0,0,w,cropHeightTop])
    bottomCrop = image.crop([0,h-cropHeightBottom,w,h])
    imageRegions.append([topCrop,bottomCrop])
        


#%% Go to OCR-town

imageText = []
processedRegions = []

# region = imageRegions[2][0]
for iImage,regionSet in enumerate(imageRegions):
    
    regionText = []
    processedRegionsThisImage = []
    
    for iRegion,region in enumerate(regionSet):
        
        # text = pytesseract.image_to_string(region)

        # pil --> cv2        
        image = np.array(region) 
        image = image[:, :, ::-1].copy() 
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.medianBlur(image, 3)
        # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # image = cv2.blur(image, (3,3))
        # image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])
        imagePil = Image.fromarray(image)
        processedRegionsThisImage.append(imagePil)
        text = pytesseract.image_to_string(imagePil, lang='eng')
        # text = pytesseract.image_to_string(imagePil, lang='eng', config='--psm 7') # psm 6
        text = text.replace('\n', ' ').replace('\r', '')

        regionText.append(text)
        
        print('Image {} ({}), region {}:\n{}\n'.format(iImage,imageFileNames[iImage],
              iRegion,text))

    # ...for each cropped region
    
    imageText.append(regionText)
    processedRegions.append(processedRegionsThisImage)
    
# ...for each image
    
    
#%% Write results
 
def resizeImage(img):
    
    targetWidth = 800
    wpercent = (targetWidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    imgResized = img.resize((targetWidth, hsize), Image.ANTIALIAS)
    return imgResized
 
    
outputDir = r'd:\temp\ocrTest'
os.makedirs(outputDir,exist_ok=True)

outputSummaryFile = os.path.join(outputDir,'summary.html')

htmlImageList = []
htmlTitleList = []

for iImage,regionSet in enumerate(imageRegions):
        
    image = resizeImage(images[iImage])
    fn = os.path.join(outputDir,'img_{}_base.png'.format(iImage))
    image.save(fn)
    
    htmlImageList.append(fn)
    htmlTitleList.append('Image: {}'.format(ntpath.basename(fn)))
        
    for iRegion,region in enumerate(regionSet):
            
        text = imageText[iImage][iRegion].strip()
        if (len(text) == 0):
            continue
        
        if False:
            image = resizeImage(region)
            fn = os.path.join(outputDir,'img_{}_r{}_raw.png'.format(iImage,iRegion))
            image.save(fn)
            htmlImageList.append(fn)
            htmlTitleList.append('Result: [' + text + ']')
        
        image = resizeImage(processedRegions[iImage][iRegion])
        fn = os.path.join(outputDir,'img_{}_r{}_processed.png'.format(iImage,iRegion))
        image.save(fn)
        htmlImageList.append(fn)
        htmlTitleList.append('Result: [' + text + ']')
        
htmlOptions = {}
htmlOptions['makeRelative'] = True
write_html_image_list.write_html_image_list(outputSummaryFile,
                                            htmlImageList,
                                            htmlTitleList,
                                            htmlOptions)
os.startfile(outputSummaryFile)
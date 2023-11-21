#
# ocr_sandbox.py
#
# sandbox for experimenting with using OCR to pull metadata from camera trap images
#
# The general approach is:
#
# * Crop a fixed percentage from the top and bottom of an image, slightly larger
#   than the largest examples we've seen of how much space is used for metadata.
#
# * Refine that crop by blurring a little, then looking for huge peaks in the 
#   color histogram suggesting a solid background, then finding rows that are
#   mostly that color.
#
# * Crop to the refined crop, then run pytesseract to extract text
#
# * Use regular expressions to find time and date, in the future can add, e.g.,
#   temperature (which is often present *only* in the images, unlike time/date which
#   are also usually in EXIF but often wrong or lost in processing)
#
# The metadata extraction (EXIF, IPTC) here is just sample code that seemed to 
# belong in this file.
#
# Contact: Dan Morris (cameratraps@lila.science)
#
   
#%% Constants and imports

import os
import warnings
import glob
import cv2
import numpy as np
import ntpath
import re
import logging

from PIL import Image

# pip install pytesseract
#
# Also intall tesseract from: https://github.com/UB-Mannheim/tesseract/wiki, and add
# the installation dir to your path (on Windows, typically C:\Program Files (x86)\Tesseract-OCR)
import pytesseract

# pip install IPTCInfo3
from iptcinfo3 import IPTCInfo

# from the ai4eutils repo: https://github.com/Microsoft/ai4eutils
#
# Only used for writing out a summary, not important for the core metadata extraction
import write_html_image_list        

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings("ignore", "Metadata warning", UserWarning)

baseDir = r'd:\temp\camera_trap_images_for_metadata_extraction'

# Using a semi-arbitrary metric of how much it feels like we found the 
# text-containing region, discard regions that appear to be extraction failures
pCropSuccessThreshold = 0.5

# Pad each crop with a few pixels to make tesseract happy
borderWidth = 10        

# Discard text from the top 
minTextLength = 4

# When we're looking for pixels that match the background color, allow some 
# tolerance around the dominant color
backgroundTolerance = 2
    
# We need to see a consistent color in at least this fraction of pixels in our rough 
# crop to believe that we actually found a candidate metadata region.
minBackgroundFraction = 0.3

# What fraction of the [top,bottom] of the image should we use for our rough crop?
imageCropFraction = [0.045 , 0.045]

# A row is considered a probable metadata row if it contains at least this fraction
# of the background color.  This is used only to find the top and bottom of the crop area, 
# so it's not that *every* row needs to hit this criteria, only the rows that are generally
# above and below the text.
minBackgroundFractionForBackgroundRow = 0.5


#%% Support functions

from PIL.ExifTags import TAGS
def get_exif(img):
    
    ret = {}
    try:
        if isinstance(img,str):
            img = Image.open(img)
        info = img._getexif()
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            ret[decoded] = value
    except:
        pass
    
    return ret


#%% Load some images, pull EXIF and IPTC data for fun

images = []
exifInfo = []
iptcInfo = []

logger = logging.getLogger('iptcinfo')
logger.disabled = True

imageFileNames = glob.glob(os.path.join(baseDir,'*.jpg'))

for fn in imageFileNames:
    
    img = Image.open(fn)
    exifInfo.append(get_exif(img))
    images.append(img)
    iptc = []
    try:
        iptc = IPTCInfo(fn)                
    except:
        pass
    iptcInfo.append(iptc)        


#%% Rough crop 

# This will be an nImages x 1 list of 2 x 1 lists (image top, image bottom)
imageRegions = []
    
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


#%% Close-crop around the text, return a revised image and success metric
    
def crop_to_solid_region(image):
    """   
    croppedImage,pSuccess,paddedImage = crop_to_solid_region(image):

    The success metric is totally arbitrary right now, but is a placeholder.
    """
           
    analysisImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    analysisImage = analysisImage.astype('uint8')     
    analysisImage = cv2.medianBlur(analysisImage,3) 
    pixelValues = analysisImage.flatten()
    counts = np.bincount(pixelValues)
    backgroundValue = int(np.argmax(counts))
    
    # Did we find a sensible mode that looks like a background value?
    backgroundValueCount = int(np.max(counts))
    pBackGroundValue = backgroundValueCount / np.sum(counts)
    
    # This looks very scientific, right?  Definitely a probability?
    if (pBackGroundValue < minBackgroundFraction):
        # print('Failed min background fraction test: {} of {}'.format(pBackGroundValue,minBackgroundFraction))
        pSuccess = 0.0
    else:
        pSuccess = 1.0
        
    analysisImage = cv2.inRange(analysisImage,
                                backgroundValue-backgroundTolerance,
                                backgroundValue+backgroundTolerance)
    
    # Notes to self, things I tried that didn't really go anywhere...
    # analysisImage = cv2.blur(analysisImage, (3,3))
    # analysisImage = cv2.medianBlur(analysisImage,5) 
    # analysisImage = cv2.Canny(analysisImage,100,100)
    # imagePil = Image.fromarray(analysisImage); imagePil
    
    # Use row heuristics to refine the crop
    #
    # This egregious block of code makes me miss my fluency in Matlab.
    if True:
        
        h = analysisImage.shape[0]
        w = analysisImage.shape[1]
        
        minX = 0
        minY = -1
        maxX = w
        maxY = -1
        
        for y in range(h):
            rowCount = 0
            for x in range(w):
                if analysisImage[y][x] > 0:
                    rowCount += 1
            rowFraction = rowCount / w
            if rowFraction > minBackgroundFractionForBackgroundRow:
                if minY == -1:
                    minY = y
                maxY = y
        
        x = minX
        y = minY
        w = maxX-minX
        h = maxY-minY
        
        x = minX
        y = minY
        w = maxX-minX
        h = maxY-minY
    
    # print('Cropping to {},{},{},{}'.format(x,y,w,h))
    
    # Crop the image
    croppedImage = image[y:y+h,x:x+w]
      
    # For some reason, tesseract doesn't like characters really close to the edge
    paddedCrop = cv2.copyMakeBorder(croppedImage,borderWidth,borderWidth,borderWidth,borderWidth,
                                   cv2.BORDER_CONSTANT,value=[backgroundValue,backgroundValue,
                                                              backgroundValue])
        
        
    # imagePil = Image.fromarray(croppedImage); imagePil
    
    return croppedImage,pSuccess,paddedCrop
    
    
#%% Go to OCR-town

# An nImages x 2 list of strings, extracted from the top and bottom of each image
imageText = []

# An nImages x 2 list of cropped images
processedRegions = []
    
# iImage = 0; iRegion = 1; regionSet = imageRegions[iImage]; region = regionSet[iRegion]

for iImage,regionSet in enumerate(imageRegions):
    
    regionText = []
    processedRegionsThisImage = []
        
    for iRegion,region in enumerate(regionSet):

        regionText.append('')
        processedRegionsThisImage.append(None)
        
        # text = pytesseract.image_to_string(region)

        # pil --> cv2        
        image = np.array(region) 
        image = image[:, :, ::-1].copy()         
        
        # image = cv2.medianBlur(image, 3)        
        # image = cv2.erode(image, None, iterations=2)
        # image = cv2.dilate(image, None, iterations=4)
        # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # image = cv2.blur(image, (3,3))
        # image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])
        
        crop,pSuccess,paddedCrop = crop_to_solid_region(image)
        
        if pSuccess < pCropSuccessThreshold:
            continue
        
        imagePil = Image.fromarray(paddedCrop)
        processedRegionsThisImage[-1] = imagePil
        # text = pytesseract.image_to_string(imagePil, lang='eng')
        # https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
        
        # psm 6: "assume a single uniform block of text"
        #
        text = pytesseract.image_to_string(imagePil, lang='eng', config='--psm 6') 
        
        text = text.replace('\n', ' ').replace('\r', '').strip()

        regionText[-1] = text
        
        if len(text) > minTextLength:
            print('Image {} ({}), region {}:\n{}\n'.format(iImage,imageFileNames[iImage],
                  iRegion,text))

    # ...for each cropped region
    
    imageText.append(regionText)
    processedRegions.append(processedRegionsThisImage)
    
# ...for each image
    

#%% Extract dates and times    
    
imageExtractedDatestamps = []

import dateutil

for iImage,regionTextList in enumerate(imageText):
    
    allTextThisImage = ''.join(regionTextList)        
    imageExtractedDatestamps.append('')
    
    print('Source: ' + allTextThisImage)
    
    datePat = r'(\d+[/-]\d+[/-]\d+)'    
    s = allTextThisImage
    match = re.findall(datePat,s)     
    dateString = ''
    if len(match) > 1:
        print('Oops: multiple date matches for {}'.format(allTextThisImage))
    elif len(match) == 1:
        dateString = match[0]
    
    timePat = r'(\d+:\d+(:\d+)?\s*(am|pm)?)'    
    # s = '1:22 pm'
    # s = '1:23:44 pm'
    s = allTextThisImage
    match = re.findall(timePat,s,re.IGNORECASE)     
    
    if len(match) > 0:
        timeString = match[0][0]
    else:
        timeString = ''
        
    if len(dateString) == 0 or len(timeString) == 0:
        continue
    
    dtString = dateString + ' ' + timeString
    
    print('Found datetime: {}'.format(dtString))    
    
    dt = dateutil.parser.parse(dtString)

    print('Converted datetime: {}'.format(str(dt)))
    
    imageExtractedDatestamps[-1] = dt
    
    print('')
    
    
#%% Write results to a handy html file
      
def resizeImage(img):
    
    targetWidth = 600
    wpercent = (targetWidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    imgResized = img.resize((targetWidth, hsize), Image.ANTIALIAS)
    return imgResized
 
    
outputDir = r'd:\temp\ocrTest'
os.makedirs(outputDir,exist_ok=True)

outputSummaryFile = os.path.join(outputDir,'summary.html')

htmlImageList = []
htmlTitleList = []

options = write_html_image_list.write_html_image_list()

for iImage,regionSet in enumerate(imageRegions):
        
    # Add image name and resized image
    image = resizeImage(images[iImage])
    fn = os.path.join(outputDir,'img_{}_base.png'.format(iImage))
    image.save(fn)
    
    title = 'Image: {}'.format(ntpath.basename(imageFileNames[iImage]))
    htmlImageList.append({'filename':fn,'title':title})
            
    bPrintedDate = False
    
    # Add results and individual region images
    for iRegion,region in enumerate(regionSet):
            
        text = imageText[iImage][iRegion].strip()
        if (len(text) == 0):
            continue
                
        image = resizeImage(processedRegions[iImage][iRegion])
        fn = os.path.join(outputDir,'img_{}_r{}_processed.png'.format(iImage,iRegion))
        image.save(fn)
        
        imageStyle = options['defaultImageStyle'] + 'margin-left:50px;'
        textStyle = options['defaultTextStyle'] + 'margin-left:50px;'
        imageFilename = fn
        extractedText = str(imageExtractedDatestamps[iImage])
        title = 'Raw text: ' + text
        if (not bPrintedDate) and (len(extractedText) != 0):
            title += '<br/>Extracted datetime: ' + extractedText
            bPrintedDate = True

        # textStyle = "font-family:calibri,verdana,arial;font-weight:bold;font-size:150%;text-align:left;margin-left:50px;"
        htmlImageList.append({'filename':fn,'imageStyle':imageStyle,'title':title,'textStyle':textStyle})
                        
htmlOptions = {}
htmlOptions['makeRelative'] = True
write_html_image_list.write_html_image_list(outputSummaryFile,
                                            htmlImageList,
                                            htmlOptions)
os.startfile(outputSummaryFile)


#%% Scrap

# Alternative approaches to finding the text/background  region
# Using findCountours()
if False:

    # imagePil = Image.fromarray(analysisImage); imagePil
    
    # analysisImage = cv2.erode(analysisImage, None, iterations=3)
    # analysisImage = cv2.dilate(analysisImage, None, iterations=3)

    # analysisImage = cv2.threshold(analysisImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    im2, contours, hierarchy = cv2.findContours(analysisImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx
   
# Using connectedComponents()
if False:
    
    # analysisImage = image
    nb_components, output, stats, centroids = \
        cv2.connectedComponentsWithStats(analysisImage, connectivity = 4)
    # print('Found {} components'.format(nb_components))
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    # We just want the *background* image
    max_label = 0
    
    maskImage = np.zeros(output.shape)
    maskImage[output == max_label] = 255
    
    thresh = 127
    binaryImage = cv2.threshold(maskImage, thresh, 255, cv2.THRESH_BINARY)[1]
    
    minX = -1
    minY = -1
    maxX = -1
    maxY = -1
    h = binaryImage.shape[0]
    w = binaryImage.shape[1]
    for y in range(h):
        for x in range(w):
            if binaryImage[y][x] > thresh:
                if minX == -1:
                    minX = x
                if minY == -1:
                    minY = y
                if x > maxX:
                    maxX = x
                if y > maxY:
                    maxY = y
    

#
# filenames_to_json.py
#
# Take a directory of images in which species labels are encoded by folder
# names, and produces a COCO-style .json file 
#
# 

#%% Constants and imports

import json
import io
import os
import uuid
import csv
import warnings
import datetime
from PIL import Image

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings("ignore", "Metadata warning", UserWarning)


# Filenames will be stored in the output .json relative to this base dir
baseDir = r'd:\wildlife_data\tigerblobs'
outputJsonFilename = os.path.join(baseDir,'tigerblobs.json')
outputCsvFilename = os.path.join(baseDir,'tigerblobs.csv')
outputEncoding = 'utf-8'

info = {}
info['year'] = 2019
info['version'] = '1.0'
info['description'] = ''
info['contributor'] = ''
info['date_created'] = str(datetime.date.today())

maxFiles = -1


#%% Enumerate files, read image sizes

# Each element will be a list of relative path/full path/width/height
fileInfo = []
nonImages = []
nFiles = 0

print('Enumerating files from {} to {}'.format(baseDir,outputCsvFilename))

with io.open(outputCsvFilename, "w", encoding=outputEncoding) as outputFileHandle:
    
    for root, subdirs, files in os.walk(baseDir):
    
        for fname in files:
      
            nFiles = nFiles + 1
            if maxFiles >= 0 and nFiles > maxFiles:            
                print('Warning: early break at {} files'.format(maxFiles))
                break
            
            fullPath = os.path.join(root,fname)            
            relativePath = os.path.relpath(fullPath,baseDir)
             
            if maxFiles >= 0:
                print(relativePath)
        
            h = -1
            w = -1

            # Read the image
            try:
            
                im = Image.open(fullPath)
                h = im.height
                w = im.width
                
            except:
                # Corrupt or not an image
                nonImages.append(fullPath)
                continue
            
            # Store file info
            imageInfo = [relativePath, fullPath, w, h]
            fileInfo.append(imageInfo)
            
            # Write to output file
            outputFileHandle.write('"' + relativePath + '"' + ',' + 
                                   '"' + fullPath + '"' + ',' + 
                                   str(w) + ',' + str(h) + '\n')
                                   
        # ...if we didn't hit the max file limit, keep going
        else:
            continue
        
        break

    # ...for each file

# ...csv file output
    
print("Finished writing {} file names to {}".format(nFiles,outputCsvFilename))


#%% Read from .csv if we're starting mid-script

if False:
    
    #%%
    
    with open(outputCsvFilename,'r') as f:
        reader = csv.reader(f)
        csvInfo = list(list(item) for item in csv.reader(f, delimiter=','))
    
    for iRow in range(len(csvInfo)):
        csvInfo[iRow][2] = int(csvInfo[iRow][2])
        csvInfo[iRow][3] = int(csvInfo[iRow][3])
    
    fileInfo = csvInfo
    
    
#%% Enumerate classes, add a class name to each row in the image table

# Maps classes to counts
classList = {}

for iRow,row in enumerate(fileInfo):
    
    fullPath = row[0]
    className = os.path.split(os.path.dirname(fullPath))[1]
    className = className.lower().strip()
    if className in classList:
        classList[className] += 1
    else:
        classList[className] = 1
    row.append(className)


#%% Assemble dictionaries

images = []
annotations = []
categories = []

categoryNameToId = {}
idToCategory = {}

nextId = 0
classNames = list(classList.keys())
classNames.insert(0,'empty')

for categoryName in classNames:
    
    catId = nextId
    nextId += 1
    categoryNameToId[categoryName] = catId
    newCat = {}
    newCat['id'] = categoryNameToId[categoryName]
    newCat['name'] = categoryName
    newCat['count'] = 0
    categories.append(newCat) 
    idToCategory[catId] = newCat
    
    
# ...for each category
    
    
# Each element is a list of relative path/full path/width/height/className
    
for iRow,row in enumerate(fileInfo):
    
    relativePath = row[0]
    w = row[2]
    h = row[3]    
    className = row[4]  
    
    assert className in categoryNameToId
    categoryId = categoryNameToId[className]
    
    im = {}
    im['id'] = str(uuid.uuid1())
    im['file_name'] = relativePath
    im['height'] = h
    im['width'] = w
    images.append(im)
    
    ann = {}
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']
    ann['category_id'] = categoryId
    annotations.append(ann)
    
    cat = idToCategory[categoryId]
    cat['count'] += 1
    
# ...for each image


#%% Write output .json

data = {}
data['info'] = info
data['images'] = images
data['annotations'] = annotations
data['categories'] = categories

json.dump(data, open(outputJsonFilename,'w'))    

print('Finished writing json to {}'.format(outputJsonFilename))

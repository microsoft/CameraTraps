#
# filenames_to_json.py
#
# Take a directory of images in which species labels are encoded by folder
# names, and produces a COCO-style .json file 
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

# from the ai4eutils repo
from path_utils import find_images

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings("ignore", "Metadata warning", UserWarning)

# Filenames will be stored in the output .json relative to this base dir
baseDir = r'D:\wildlife_data\bellevue_camera_traps\bellevue_camera_traps.19.06.02.1320'
outputJsonFilename = os.path.join(baseDir,'bellevue_camera_traps.19.06.02.1320.json')
outputCsvFilename = os.path.join(baseDir,'bellevue_camera_traps.19.06.02.1320.csv')

# rawClassListFilename = os.path.join(baseDir,'bellevue_camera_traps.19.06.02.1320_classes.csv')
# classMappingsFilename = os.path.join(baseDir,'bellevue_camera_traps.19.06.02.1320_class_mapping.csv')
outputEncoding = 'utf-8'

classMappings = {'transitional':'unlabeled','moving':'unlabeled','setup':'unlabeled','blurry':'unlabeled','transitional':'unlabeled','junk':'unlabeled','unknown':'unlabeled'}

bLoadFileListIfAvailable = True

info = {}
info['year'] = 2019
info['version'] = '1.0'
info['description'] = 'Bellevue Camera Traps'
info['contributor'] = 'Dan Morris'
info['date_created'] = str(datetime.date.today())

maxFiles = -1
bReadImageSizes = False
bUseExternalRemappingTable = False


#%% Enumerate files, read image sizes

# Each element will be a list of relative path/full path/width/height
fileInfo = []
nonImages = []
nFiles = 0

if bLoadFileListIfAvailable and os.path.isfile(outputCsvFilename):
    
    print('Loading file list from {}'.format(outputCsvFilename))
    
    with open(outputCsvFilename,'r') as f:
        reader = csv.reader(f)
        csvInfo = list(list(item) for item in csv.reader(f, delimiter=','))
    
    for iRow in range(len(csvInfo)):
        csvInfo[iRow][2] = int(csvInfo[iRow][2])
        csvInfo[iRow][3] = int(csvInfo[iRow][3])
    
    fileInfo = csvInfo
    
    print('Finished reading list of {} files'.format(len(fileInfo)))
    
else:
        
    print('Enumerating files from {} to {}'.format(baseDir,outputCsvFilename))
    
    image_files = find_images(baseDir,bRecursive=True)
    print('Enumerated {} images'.format(len(image_files)))
    
    with io.open(outputCsvFilename, "w", encoding=outputEncoding) as outputFileHandle:    
    
        for fname in image_files:
      
            nFiles = nFiles + 1
            if maxFiles >= 0 and nFiles > maxFiles:            
                print('Warning: early break at {} files'.format(maxFiles))
                break
            
            fullPath = fname
            relativePath = os.path.relpath(fullPath,baseDir)
             
            if maxFiles >= 0:
                print(relativePath)
        
            h = -1
            w = -1

            if bReadImageSizes:
            
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
                                   
        # ...for each image file
                
    # ...csv file output
        
    print("Finished writing {} file names to {}".format(nFiles,outputCsvFilename))

# ...if the file list is/isn't available
    
    
#%% Enumerate classes

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

classNames = list(classList.keys())

# We like 'empty' to be class 0
if 'empty' in classNames:
    classNames.remove('empty')    
classNames.insert(0,'empty')

print('Finished enumerating {} classes'.format(len(classList)))


#%% Assemble dictionaries

images = []
annotations = []
categories = []

categoryNameToId = {}
idToCategory = {}
imageIdToImage = {}

nextId = 0
    
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
    imageIdToImage[im['id']] = im
    
    ann = {}
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']
    ann['category_id'] = categoryId
    annotations.append(ann)
    
    cat = idToCategory[categoryId]
    cat['count'] += 1
    
# ...for each image

oldNameToOldId = categoryNameToId
originalCategories = categories

print('Finished assembling dictionaries')


#%% External class mapping

if bUseExternalRemappingTable:
    
    assert classMappings is None
    
    
    #%% Write raw class table
    
    # cat = categories[0]
    if os.path.isfile(rawClassListFilename):
        
        print('Not over-writing raw class table')
    
    else:
        
        with io.open(rawClassListFilename, "w", encoding=outputEncoding) as classListFileHandle:
            for cat in categories:
                catId = cat['id']
                categoryName = cat['name']
                categoryCount = cat['count']
                classListFileHandle.write(str(catId) + ',"' + categoryName + '",' + str(categoryCount) + '\n')
        
        print('Finished writing raw class table')
    
    
    #%% Read the mapped class table
    
    classMappings = {}
    
    if os.path.isfile(classMappingsFilename):
    
        print('Loading file list from {}'.format(classMappingsFilename))
            
        with open(classMappingsFilename,'r') as f:
            reader = csv.reader(f)
            mappingInfo = list(list(item) for item in csv.reader(f, delimiter=','))
        
        for mapping in mappingInfo:
            assert len(mapping) == 4
            
            # id, source, count, target
            sourceClass = mapping[1]
            targetClass = mapping[3]
            assert sourceClass not in classMappings
            classMappings[sourceClass] = targetClass
        
        print('Finished reading list of {} class mappings'.format(len(mappingInfo)))
        
else:

#%% Make classMappings contain *all* classes, not just remapped classes
    
    # cat = categories[0]
    for cat in categories:
        if cat['name'] not in classMappings:
            classMappings[cat['name']] = cat['name']

      
#%% Create new class list
    
categories = []
categoryNameToId = {}
oldIdToNewId = {}

# Start at 1, explicitly assign 0 to "empty"
nextCategoryId = 1
for sourceClass in classMappings:
    targetClass = classMappings[sourceClass]
    
    if targetClass not in categoryNameToId:
        
        if targetClass == 'empty':
            categoryId = 0
        else:
            categoryId = nextCategoryId
            nextCategoryId = nextCategoryId + 1
            
        categoryNameToId[targetClass] = categoryId
        newCat = {}
        newCat['id'] = categoryId
        newCat['name'] = targetClass
        newCat['count'] = 0
        
        if targetClass == 'empty':
            categories.insert(0,newCat)
        else:
            categories.append(newCat)

    else:
        
        categoryId = categoryNameToId[targetClass]
    
    # One-off issue with character encoding
    if sourceClass == 'humanÃ¯â‚¬Â¨':
        sourceClass = 'humanï€¨'
        
    assert sourceClass in oldNameToOldId
    oldId = oldNameToOldId[sourceClass]
    oldIdToNewId[oldId] = categoryId

categoryIdToCat = {}
for cat in categories:
    categoryIdToCat[cat['id']] = cat
    
print('Mapped {} original classes to {} new classes'.format(len(originalCategories),len(categories)))


#%% Re-map annotations
            
# ann = annotations[0]            
for ann in annotations:

    ann['category_id'] = oldIdToNewId[ann['category_id']]
    
    
#%% Write output .json

data = {}
data['info'] = info
data['images'] = images
data['annotations'] = annotations
data['categories'] = categories

json.dump(data, open(outputJsonFilename,'w'), indent=4)

print('Finished writing json to {}'.format(outputJsonFilename))


#%% Utilities

if False:
    
    #%% 
    # Find images with a particular tag
    className = 'hum'
    matches = []
    assert className in categoryNameToId
    catId = categoryNameToId[className]
    for ann in annotations:
        if ann['category_id'] == catId:
            imageId = ann['image_id']
            im = imageIdToImage[imageId]
            matches.append(im['file_name'])
    print('Found {} matches'.format(len(matches)))
    
    os.startfile(os.path.join(baseDir,matches[0]))
    
    
    #%% Randomly sample annotations
    
    import random
    nAnnotations = len(annotations)
    iAnn = random.randint(0,nAnnotations)
    ann = annotations[iAnn]
    catId = ann['category_id']
    imageId = ann['image_id']
    im = imageIdToImage[imageId]
    fn = os.path.join(baseDir,im['file_name'])
    cat = categoryIdToCat[catId]
    className = cat['name']
    print('This should be a {}'.format(className))
    os.startfile(fn)

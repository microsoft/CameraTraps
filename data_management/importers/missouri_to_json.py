#
# missouri_to_json.py
#
# Create .json files from the original source files for the Missouri Camera Traps
# data set.
# 

#%% Constants and imports

import json
import io
import os
import uuid
import csv
import warnings
import ntpath
import datetime
from PIL import Image

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings("ignore", "Metadata warning", UserWarning)

# Filenames will be stored in the output .json relative to this base dir
baseDir = r'e:\wildlife_data\missouri_camera_traps'
outputJsonFilenameSet1 = os.path.join(baseDir,'missouri_camera_traps_set1.json')
outputJsonFilenameSet2 = os.path.join(baseDir,'missouri_camera_traps_set2.json')
outputEncoding = 'utf-8'
fileListJsonFilename = os.path.join(baseDir,'images.json')

set1BaseDir = os.path.join(baseDir,'Set1')
set2BaseDir = os.path.join(baseDir,'Set2')

metadataFilenameSet1 = os.path.join(set1BaseDir,'labels.txt')
metadataFilenameSet2 = os.path.join(set1BaseDir,'labels.txt')

assert(os.path.isdir(baseDir))
assert(os.path.isfile(metadataFilenameSet1))
assert(os.path.isfile(metadataFilenameSet2))

info = {}
info['year'] = 2019
info['version'] = '1.0'
info['description'] = 'Missouri Camera Traps (set 1)'
info['contributor'] = ''
info['date_created'] = str(datetime.date.today())

info = {}
info['year'] = 2019
info['version'] = '1.0'
info['description'] = 'Missouri Camera Traps (set 2)'
info['contributor'] = ''
info['date_created'] = str(datetime.date.today())

maxFiles = -1


#%% Enumerate files, read image sizes

# Each element will be a list of relative path/full path/width/height
fileInfo = []
nonImages = []
nFiles = 0

relPathToIm = {}
imageIdToImage = {}

set1ImageIDs = []
set2ImageIDs = []
   
sequenceIDtoCount = {}

print('Enumerating files from {} to {}'.format(baseDir,fileListJsonFilename))

for root, subdirs, files in os.walk(baseDir):
            
    bn = ntpath.basename(root)
    
    if ('Set1' in root and 'SEQ' in root) or ('Set2' in root and bn != 'Set2'):
        sequenceID = bn
        assert sequenceID not in sequenceIDtoCount
        sequenceIDtoCount[sequenceID] = 0
    else:
        assert len(files) <= 2
    
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
            
            # Not an image...
            continue
        
        # Store file info
        im = {}
        im['id'] = str(uuid.uuid1())
        im['file_name'] = relativePath
        im['height'] = h
        im['width'] = w
        
        im['seq_id'] = sequenceID
        im['seq_num_frames'] = -1
        
        count = sequenceIDtoCount[sequenceID]
        im['frame_num'] = count
        sequenceIDtoCount[sequenceID] = sequenceIDtoCount[sequenceID] + 1
        
        imageIdToImage[im['id']] = im
        relPathToIm[relativePath] = im
        
        if 'Set1' in relativePath:
            set1ImageIDs.append(im['id'])
        elif 'Set2' in relativePath:
            set2ImageIDs.append(im['id'])
        else:
            raise Exception('Oops, can''t assign this image to a set')
            
    # ...if we didn't hit the max file limit, keep going
    else:
        
        continue
    
    break

# ...for each file


#%% Add sequence lengths
    
for imageID in imageIdToImage:
    im = imageIdToImage[imageID]
    sequenceID = im['seq_id']
    count = sequenceIDtoCount[sequenceID]
    assert(im['seq_num_frames'] == -1)
    im['seq_num_frames'] = count
    

#%% Load the set 1 metadata file, split into tokens

with open(metadataFilenameSet1) as f:
    metadataSet1Lines = f.readlines()

correctedFiles = []
missingFiles = []

metadataSet1Lines = [x.strip() for x in metadataSet1Lines] 

# List of lists, length varies according to number of bounding boxes
#
# Preserves original ordering
metadataSet1 = []

relPathToMetadata = {}

# iLine = 0; line = metadataSet1Lines[0]
for iLine,line in enumerate(metadataSet1Lines):
    tokens = line.split()
    nTokens = len(tokens)
    
    # Lines should be filename, number of bounding boxes, boxes (four values per box)
    assert ((nTokens - 2) % 4) == 0
    relPath = tokens[0].replace('/',os.sep).replace('\\',os.sep)
    relPath = os.path.join('Set1',relPath)
    absPath = os.path.join(baseDir,relPath)
    
    originalAbsPath = absPath
    originalRelPath = relPath
        
    if not os.path.isfile(absPath):
        
        absPath = absPath.replace('IMG','IMG_')     
        relPath = relPath.replace('IMG','IMG_')        
        correctedFiles.append([relPath,originalRelPath,absPath,originalAbsPath])
        
    if not os.path.isfile(absPath):
        
        missingFiles.append([originalRelPath,originalAbsPath])
        
    else:
        
        metadataSet1.append(tokens)
        relPathToMetadata[relPath] = tokens
        
        # Make sure we have image info for this image
        assert relPath in relPathToIm

print('Corrected {} paths, missing {} images of {}'.format(len(correctedFiles),
      len(missingFiles),len(metadataSet1Lines)))


#%% Print missing files

# 'IMG' --> 'IMG_'
# Red_Brocket_Deer --> Red_Deer
# European-Hare --> European_Hare
# Wood-Mouse --> Wood_Mouse
# Coiban-Agouti --> Coiban_Agouti
for iFile,fInfo in enumerate(missingFiles):
    print(fInfo[0])
    

#%% Create records for each image and annotation, accumulating classes as we go
    
images = []
annotations = []
categories = []

categoryNameToId = {}
idToCategory = {}
imageIdToImage = {}
relativeFilenameToImageId = {}
sequenceIdToCount = {}

nextCategoryId = 0

# For each image...

    # Confirm we haven't seen this image before
    
    # Pull out the species name
    
    # If we haven't seen this species, create a new record for the class
    if False:
        catId = nextCategoryId
        nextCategoryId += 1
        categoryNameToId[categoryName] = catId
        newCat = {}
        newCat['id'] = categoryNameToId[categoryName]
        newCat['name'] = categoryName
        newCat['count'] = 0
        categories.append(newCat) 
        idToCategory[catId] = newCat
    
    # Pull out the sequence ID
    
    # If we haven't seen this sequence before, create a count record for it
    
    # Else update the count
    
    # Give it an ID
    im = {}
    im['id'] = str(uuid.uuid1())
    im['file_name'] = relativePath
    im['height'] = h
    im['width'] = w
    im['seq_id'] = seqID
    im['seq_num_frames'] = -1
    im['frame_num'] = iFrame
    images.append(im)
    imageIdToImage[im['id']] = im
    
    # If this image is empty, create an empty annotation
    
    # Create one annotation per bounding box
    ann = {}
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']
    ann['category_id'] = categoryId
    annotations.append(ann)

# ...for each image
    
#%% Assemble dictionaries


for categoryName in classNames:
    
    
            
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
    
print('Mapped {} original classes to {} new classes'.format(len(mappingInfo),len(categories)))


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

json.dump(data, open(outputJsonFilename,'w'))    

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

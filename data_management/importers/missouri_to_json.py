#
# missouri_to_json.py
#
# Create .json files from the original source files for the Missouri Camera Traps
# data set.  Metadata was provided here in two formats:
#
# 1) In one subset of the data, folder names indicated species names.  In Set 1,
#    there are no empty sequences.  Set 1 has a metadata file to indicate image-level
#    bounding boxes.
#
# 2) A subset of the data (overlapping with (1)) was annotated with bounding
#    boxes, specified in a whitespace-delimited text file.  In set 2, there are
#    some sequences omitted from the metadata file, which implied emptiness.
# 
# In the end, set 2 labels were not reliable enough to publish, so LILA includes only set 1.
#

#%% Constants and imports

import json
import os
import uuid
import time
import humanfriendly
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
metadataFilenameSet2 = os.path.join(set2BaseDir,'labels.txt')

assert(os.path.isdir(baseDir))
assert(os.path.isfile(metadataFilenameSet1))
assert(os.path.isfile(metadataFilenameSet2))

info = {}
info['year'] = 2019
info['version'] = '1.1'
info['description'] = 'Missouri Camera Traps (set 1)'
info['contributor'] = ''
info['date_created'] = str(datetime.date.today())
infoSet1 = info

info = {}
info['year'] = 2019
info['version'] = '1.1'
info['description'] = 'Missouri Camera Traps (set 2)'
info['contributor'] = ''
info['date_created'] = str(datetime.date.today())
infoSet2 = info

maxFiles = -1
emptyCategoryId = 0
emptyCategoryName = 'empty'


#%% Enumerate files, read image sizes (both sets)

# Takes a few minutes, since we're reading image sizes.

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

startTime = time.time()

for root, subdirs, files in os.walk(baseDir):
            
    if root == baseDir:
        continue
    
    bn = ntpath.basename(root)
    
    # Only process leaf nodes corresponding to sequences, which look like:
    #
    # E:\wildlife_data\missouri_camera_traps\Set1\1.02-Agouti\SEQ75583
    # E:\wildlife_data\missouri_camera_traps\Set2\p1d101
    #
    if ('Set1' in root and 'SEQ' in bn) or ('Set2' in root and bn.startswith('p')):
        sequenceID = bn
        assert sequenceID not in sequenceIDtoCount
        sequenceIDtoCount[sequenceID] = 0
    else:
        print('Skipping folder {}:{}'.format(root,bn))
        continue
        # assert len(files) <= 2
    
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
        
        frame_number = sequenceIDtoCount[sequenceID]
        im['frame_num'] = frame_number
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

elapsed = time.time() - startTime
print('Finished file enumeration in {}'.format(
      humanfriendly.format_timespan(elapsed)))


#%% Add sequence lengths (both sets)
    
for imageID in imageIdToImage:
    
    im = imageIdToImage[imageID]
    sequenceID = im['seq_id']
    seq_num_frames = sequenceIDtoCount[sequenceID]
    assert(im['seq_num_frames'] == -1)
    im['seq_num_frames'] = seq_num_frames
    

#%% Load the set 1 metadata file

with open(metadataFilenameSet1) as f:
    metadataSet1Lines = f.readlines()

correctedFiles = []
missingFilesSet1 = []

metadataSet1Lines = [x.strip() for x in metadataSet1Lines] 

# List of lists, length varies according to number of bounding boxes
#
# Preserves original ordering
metadataSet1 = []

relPathToMetadataSet1 = {}

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
        
        missingFilesSet1.append([originalRelPath,originalAbsPath])
        
    else:
        
        metadataSet1.append(tokens)
        relPathToMetadataSet1[relPath] = tokens
        
        # Make sure we have image info for this image
        assert relPath in relPathToIm

print('Corrected {} paths, missing {} images of {}'.format(len(correctedFiles),
      len(missingFilesSet1),len(metadataSet1Lines)))


#%% Print missing files from Set 1 metadata

# Manual changes I made to the metadata file:
#
# 'IMG' --> 'IMG_'
# Red_Brocket_Deer --> Red_Deer
# European-Hare --> European_Hare
# Wood-Mouse --> Wood_Mouse
# Coiban-Agouti --> Coiban_Agouti

print('Missing files in Set 1:\n')
for iFile,fInfo in enumerate(missingFilesSet1):
    print(fInfo[0])
    

#%% Load the set 2 metadata file

# This metadata file contains most (but not all) images, and a class label (person/animal/empty)
# for each, plus bounding boxes.

with open(metadataFilenameSet2) as f:
    metadataSet2Lines = f.readlines()

metadataSet2Lines = [x.strip() for x in metadataSet2Lines] 

set2ClassMappings = {0:'human',1:'animal'}

# List of lists, length varies according to number of bounding boxes
#
# Preserves original ordering
metadataSet2 = []

relPathToMetadataSet2 = {}

# Create class IDs for each *sequence*, which we'll use to attach classes to 
# images for which we don't have metadata
#
# This only contains mappings for sequences that appear in the metadata.
set2SequenceToClass = {}

missingFilesSet2 = []

# iLine = 0; line = metadataSet2Lines[0]
for iLine,line in enumerate(metadataSet2Lines):
    
    tokens = line.split()
    nTokens = len(tokens)
    
    # Lines should be filename, number of bounding boxes, labeled boxes (five values per box)
    #
    # Empty images look like filename\t0\t0
    assert (nTokens == 3 and tokens[1] == '0' and tokens[2] == '0') or (((nTokens - 2) % 5) == 0)
    relPath = tokens[0].replace('/',os.sep).replace('\\',os.sep)
    relPath = os.path.join('Set2',relPath)
    absPath = os.path.join(baseDir,relPath)

    # E.g. 'Set2\\p1d101\\p1d101s100i10.JPG'
    fnTokens = os.path.normpath(relPath).split(os.sep)
    sequenceID = fnTokens[-2]
    
    # Make sure we don't have mixed classes within an image
    nBoxes = int(tokens[1])
    imageClass = None
    if nBoxes == 0:
        imageClass = 'empty'
    else:
        for iBox in range(0,nBoxes):
            boxClass = int(tokens[2+5*(iBox)])
            boxClass = set2ClassMappings[boxClass]
            if imageClass is None:
                imageClass = boxClass
            elif boxClass != imageClass:
                imageClass = 'mixed'                
    
    assert imageClass is not None
    
    # Figure out what class this *sequence* is, so we know how to handle unlabeled
    # images from this sequence
    if sequenceID in set2SequenceToClass:
        
        sequenceClass = set2SequenceToClass[sequenceID]
        
        # Can't un-do a mixed sequence
        if sequenceClass == 'mixed':
            pass
        
        # Previously-empty sequences get the image class label
        elif sequenceClass == 'empty':
            if imageClass != 'empty':
                sequenceClass = imageClass
        
        # If the sequence has a non-empty class, possibly change it
        else:
            if imageClass == 'empty':
                pass
            elif imageClass != sequenceClass:
                sequenceClass = imageClass
        
        set2SequenceToClass[sequenceID] = sequenceClass
        
    else:
        
        set2SequenceToClass[sequenceID] = imageClass
    
    if not os.path.isfile(absPath):
        missingFilesSet2.append(absPath)
        continue
    
    metadataSet2.append(tokens)
    relPathToMetadataSet2[relPath] = tokens
        
    # Make sure we have image info for this image
    assert relPath in relPathToIm

# ...for each line in the set 2 metadata file
    
print('Missing {} of {} files in set 2'.format(len(missingFilesSet2),len(metadataSet2Lines)))

if False:
    for iFile,filename in enumerate(missingFilesSet2):
        print(filename)


#%% What Set 2 images do I not have metadata for?
    
# These are *mostly* empty images

if True:

    set2UnlabeledImageIndices = []
    set2ImageIDsUnlabeled = []
    
    # iImage = 0; imageID = set2ImageIDs[iImage]
    for iImage,imageID in enumerate(set2ImageIDs):
        
        im = imageIdToImage[imageID]
        if not im['file_name'] in relPathToMetadataSet2:
            set2UnlabeledImageIndices.append(iImage)
            set2ImageIDsUnlabeled.append(imageID)
            
    print('{} of {} files in set 2 lack labels'.format(len(set2UnlabeledImageIndices),len(set2ImageIDs)))
    
    set2ImageIDsLabeled = [x for i,x in enumerate(set2ImageIDs) if i not in set2UnlabeledImageIndices]    
    
    assert len(set2ImageIDsLabeled) + len(set2ImageIDsUnlabeled) == len(set2ImageIDs)


#%% Create categories and annotations for set 1

imagesSet1 = []
categoriesSet1 = []
annotationsSet1 = []

categoryNameToId = {}
idToCategory = {}

# Though we have no empty sequences, we do have empty images in this set
emptyCat = {}
emptyCat['id'] = emptyCategoryId
emptyCat['name'] = emptyCategoryName
emptyCat['count'] = 0
categoriesSet1.append(emptyCat) 

nextCategoryId = emptyCategoryId + 1
    
nFoundMetadata = 0
nTotalBoxes = 0
nImageLevelEmpties = 0
nSequenceLevelAnnotations = 0
nRedundantBoxes = 0

# For each image
#
# iImage = 0; imageID = set1ImageIDs[iImage]
for iImage,imageID in enumerate(set1ImageIDs):
    
    im = imageIdToImage[imageID]
    imagesSet1.append(im)
    
    # E.g. Set1\\1.80-Coiban_Agouti\\SEQ83155\\SEQ83155_IMG_0010.JPG
    relPath = im['file_name']

    # Find the species name
    tokens = os.path.normpath(relPath).split(os.sep)
    speciesTag = tokens[1]
    tokens = speciesTag.split('-',1)
    assert(len(tokens) == 2)
    categoryName = tokens[1].lower()
    
    category = None
    categoryId = None
    
    if categoryName not in categoryNameToId:
        
        categoryId = nextCategoryId
        nextCategoryId += 1
        categoryNameToId[categoryName] = categoryId
        newCat = {}
        newCat['id'] = categoryNameToId[categoryName]
        newCat['name'] = categoryName
        newCat['count'] = 0
        categoriesSet1.append(newCat) 
        idToCategory[categoryId] = newCat
        category = newCat
        
    else:
        
        categoryId = categoryNameToId[categoryName]
        category = idToCategory[categoryId]
        
        # This image may still be empty...
        # category['count'] = category['count'] + 1
                
    # If we have bounding boxes, create image-level annotations    
    if relPath in relPathToMetadataSet1:
        
        nFoundMetadata += 1
        
        # filename, number of bounding boxes, boxes (four values per box)
        imageMetadata = relPathToMetadataSet1[relPath]
        
        # Make sure the relative filename matches, allowing for the fact that
        # some of the filenames in the metadata aren't quite right
        fn = imageMetadata[0]
        s = relPath.replace('Set1','').replace('\\','/')[1:]
        if (fn != s):
            s = s.replace('IMG_','IMG')
            
        assert(fn == s)
        
        nBoxes = int(imageMetadata[1])
        
        if nBoxes == 0:
            
            ann = {}
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = im['id']
            ann['category_id'] = emptyCategoryId
            ann['sequence_level_annotation'] = False
            annotationsSet1.append(ann)
            emptyCat['count'] = emptyCat['count'] + 1
            nImageLevelEmpties += 1
            
        else:
            
            # This image is non-empty
            category['count'] = category['count'] + 1
            
            for iBox in range(0,nBoxes):
                                
                boxCoords = imageMetadata[2+(iBox*4):6+(iBox*4)]
                boxCoords = list(map(int, boxCoords))
                
                # Some redundant bounding boxes crept in, don't add them twice
                bRedundantBox = False
                
                # Check this bbox against previous bboxes
                #
                # Inefficient?  Yes.  In an important way?  No.
                for iBoxComparison in range(0,iBox):
                    assert iBox != iBoxComparison                        
                    boxCoordsComparison = imageMetadata[2+(iBoxComparison*4):6+(iBoxComparison*4)]
                    boxCoordsComparison = list(map(int, boxCoordsComparison))
                    if boxCoordsComparison == boxCoords:
                        print('Warning: redundant box on image {}'.format(fn))
                        bRedundantBox = True
                        nRedundantBoxes += 1
                        break
                
                if bRedundantBox:
                    continue
                    
                # Bounding box values are in absolute coordinates, with the origin 
                # at the upper-left of the image, as [xmin1 ymin1 xmax1 ymax1].
                #
                # Convert to floats and to x/y/w/h, as per CCT standard
                bboxW = boxCoords[2] - boxCoords[0]
                bboxH = boxCoords[3] - boxCoords[1]
                
                box = [boxCoords[0], boxCoords[1], bboxW, bboxH]
                box = list(map(float, box))
                
                ann = {}
                ann['id'] = str(uuid.uuid1())
                ann['image_id'] = im['id']
                ann['category_id'] = categoryId
                ann['sequence_level_annotation'] = False
                ann['bbox'] = box
                annotationsSet1.append(ann)
                nTotalBoxes += 1
            
            # ...for each box
            
        # if we do/don't have boxes for this image
        
    # Else create a sequence-level annotation
    else:
        
        ann = {}
        ann['id'] = str(uuid.uuid1())
        ann['image_id'] = im['id']
        ann['category_id'] = categoryId
        ann['sequence_level_annotation'] = True
        annotationsSet1.append(ann)
        nSequenceLevelAnnotations += 1
        
# ...for each image
        
print('Finished processing set 1, found metadata for {} of {} images\nCreated {} annotations and {} boxes in {} categories\nFound {} redundant annotations'.format(
        nFoundMetadata,len(set1ImageIDs),len(annotationsSet1),nTotalBoxes,len(categoriesSet1),nRedundantBoxes))

assert len(annotationsSet1) == nSequenceLevelAnnotations + nTotalBoxes + nImageLevelEmpties
assert len(set1ImageIDs) == nSequenceLevelAnnotations + nFoundMetadata


#%% Create categories and annotations for set 2

imagesSet2 = []
categoriesSet2 = []
annotationsSet2 = []

categoryNameToId = {}
idToCategory = {}

emptyCat = {}
assert emptyCategoryId == 0
emptyCat['id'] = emptyCategoryId
emptyCat['name'] = emptyCategoryName
emptyCat['count'] = 0
categoriesSet2.append(emptyCat) 

humanCat = {}
humanCat['id'] = 1
humanCat['name'] = 'human'
humanCat['count'] = 0
categoriesSet2.append(humanCat) 

animalCat = {}
animalCat['id'] = 2
animalCat['name'] = 'animal'
animalCat['count'] = 0
categoriesSet2.append(animalCat) 

unknownCat = {}
unknownCat['id'] = 3
unknownCat['name'] = 'unknown'
unknownCat['count'] = 0
categoriesSet2.append(unknownCat) 

categoryMappingsSet2 = {0:humanCat,1:animalCat}
sequenceClassCategoryMappingsSet2 = {'human':humanCat,'animal':animalCat,
                                     'mixed':unknownCat,'empty':emptyCat}

nFoundMetadata = 0
nImageLevelEmpties = 0
nSequenceLevelAnnotations = 0
nTotalBoxes = 0

# For each image
#
# iImage = 0; imageID = set2ImageIDs[iImage]
for iImage,imageID in enumerate(set2ImageIDs):
    
    im = imageIdToImage[imageID]    
    imagesSet2.append(im)
    
    # E.g. 'Set2\\p1d100\\p1d100s10i1.JPG'
    relPath = im['file_name']

    # Find the sequence ID, sanity check filename against what we stored
    tokens = os.path.normpath(relPath).split(os.sep)
    sequenceID = tokens[1]
    assert(sequenceID == im['seq_id'])
    
    # If we have bounding boxes or an explicit empty label, create image-level annotations    
    if relPath in relPathToMetadataSet2:
        
        nFoundMetadata += 1
        
        # filename, number of bounding boxes, labeled boxes (five values per box)
        imageMetadata = relPathToMetadataSet2[relPath]
        
        # Make sure the relative filename matches, allowing for the fact that
        # some of the filenames in the metadata aren't quite right
        fn = os.path.normpath(imageMetadata[0])
        s = os.path.normpath(relPath.replace('Set2\\',''))
        assert(fn == s)
        
        nBoxes = int(imageMetadata[1])
        
        if nBoxes == 0:
            
            ann = {}
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = im['id']
            ann['category_id'] = emptyCat['id']
            ann['sequence_level_annotation'] = False
            annotationsSet2.append(ann)
            nImageLevelEmpties += 1
            
        else:
            
            for iBox in range(0,nBoxes):
                                
                boxCoords = imageMetadata[2+(iBox*5):7+(iBox*5)]
                boxCoords = list(map(int, boxCoords))
                
                boxClass = boxCoords[0]
                boxCat = categoryMappingsSet2[boxClass]
                categoryId = boxCat['id']
                
                boxCoords = boxCoords[1:]
                
                # Bounding box values are in absolute coordinates, with the origin 
                # at the upper-left of the image, as [xmin1 ymin1 xmax1 ymax1].
                #
                # Convert to floats and to x/y/w/h
                bboxW = boxCoords[2] - boxCoords[0]
                bboxH = boxCoords[3] - boxCoords[1]
                
                box = [boxCoords[0], boxCoords[1], bboxW, bboxH]
                box = list(map(float, box))
                
                ann = {}
                ann['id'] = str(uuid.uuid1())
                ann['image_id'] = im['id']
                ann['category_id'] = categoryId
                ann['sequence_level_annotation'] = False
                ann['bbox'] = box
                annotationsSet2.append(ann)
                nTotalBoxes += 1
                
            # ...for each box
            
        # ...if we do/don't have boxes for this image
        
    # Else create a sequence-level annotation
    else:
        
        if sequenceID not in set2SequenceToClass:
            sequenceClass = 'empty'
        else:
            sequenceClass = set2SequenceToClass[sequenceID]
        category = sequenceClassCategoryMappingsSet2[sequenceClass]
        categoryId = category['id']
        
        ann = {}
        ann['id'] = str(uuid.uuid1())
        ann['image_id'] = im['id']
        ann['category_id'] = categoryId
        ann['sequence_level_annotation'] = True
        annotationsSet2.append(ann)
        nSequenceLevelAnnotations += 1
        
# ...for each image
        
print('Finished processing set 2, found metadata for {} of {} images, created {} annotations and {} boxes in {} categories'.format(
        nFoundMetadata,len(set2ImageIDs),len(annotationsSet2),nTotalBoxes,len(categoriesSet2)))

assert len(annotationsSet2) == nSequenceLevelAnnotations + nTotalBoxes + nImageLevelEmpties
assert len(set2ImageIDs) == nSequenceLevelAnnotations + nFoundMetadata
    
    
#%% The 'count' field isn't really meaningful, delete it

# It's really the count of image-level annotations, not total images assigned to a class
for d in categoriesSet1:
    del d['count']
    
    
#%% Write output .json files

data = {}
data['info'] = infoSet1
data['images'] = imagesSet1
data['annotations'] = annotationsSet1
data['categories'] = categoriesSet1
json.dump(data, open(outputJsonFilenameSet1,'w'), indent=4)    
print('Finished writing json to {}'.format(outputJsonFilenameSet1))

data = {}
data['info'] = infoSet2
data['images'] = imagesSet2
data['annotations'] = annotationsSet2
data['categories'] = categoriesSet2
json.dump(data, open(outputJsonFilenameSet2,'w'), indent=4)
print('Finished writing json to {}'.format(outputJsonFilenameSet2))


#%% Sanity-check final set 1 .json file

from data_management.databases import sanity_check_json_db
options = sanity_check_json_db.SanityCheckOptions()
sortedCategories,data = sanity_check_json_db.sanity_check_json_db(outputJsonFilenameSet1, options)
sortedCategories

# python sanity_check_json_db.py --bCheckImageSizes --baseDir "E:\wildlife_data\missouri_camera_traps" "E:\wildlife_data\missouri_camera_traps\missouri_camera_traps_set1.json"


#%% Generate previews

from visualization import visualize_db

output_dir = os.path.join(baseDir,'preview')

options = visualize_db.DbVizOptions()
options.num_to_visualize = 1000
options.sort_by_filename = False
options.classes_to_exclude = None

htmlOutputFile,_ = visualize_db.process_images(outputJsonFilenameSet1,output_dir,baseDir,options)
os.startfile(htmlOutputFile)

# Generate previewse:\wildlife_data\missouri_camera_traps\missouri_camera_traps_set1.json" "e:\wildlife_data\missouri_camera_traps\preview" "e:\wildlife_data\missouri_camera_traps" --num_to_visualize 1000

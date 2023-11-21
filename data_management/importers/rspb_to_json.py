#
# rspb_to_json.py
#
# Convert the .csv file provided for the RSPB data set to a 
# COCO-camera-traps .json file
#

#%% Constants and environment

import pandas as pd
import os
import glob
import json
import re
import uuid
import tqdm
import time
import ntpath
import humanfriendly
import PIL

from data_management.databases import sanity_check_json_db
from visualization import visualize_db

# [location] is an obfuscation
baseDir = r'e:\wildlife_data\rspb_gola_data'
metadataFile = os.path.join(baseDir,'gola_camtrapr_master_renaming_table_2019-01-31.csv')
outputFile = os.path.join(baseDir,'rspb_gola_labeled.json')
imageBaseDir = os.path.join(baseDir,'gola_camtrapr_data')
imageFlatDir = os.path.join(baseDir,'gola_camtrapr_data_flat')
unmatchedImagesFile = os.path.join(baseDir,'unmatchedImages.txt')
assert(os.path.isdir(imageBaseDir))


#%% Create info struct

info = {}
info['year'] = 2019
info['version'] = 1
info['description'] = 'COCO style database for RSPB gola data'
info['secondary_contributor'] = 'Converted to COCO .json by Dan Morris'
info['contributor'] = 'RSPB'


#%% Read source data

metadataTable = pd.read_csv(metadataFile)

print('Read {} columns and {} rows from metadata file'.format(len(metadataTable.columns),
      len(metadataTable)))

# metadataTable.columns.values
#
# array(['Project', 'inDir', 'FileName', 'Station', 'Camera',
#        'StationCameraFileName', 'DateTimeOriginal', 'DateReadable',
#        'outDir', 'filename_new', 'fileExistsAlready', 'CopyStatus',
#        'Species'], dtype=object)

metadataTable[['Species']] = metadataTable[['Species']].fillna(value='unlabeled')

# We'll populate these later 
metadataTable['sequenceID'] = ''
metadataTable['frameNumber'] = ''
metadataTable['filePath'] = ''

failedCopies = metadataTable[~metadataTable.CopyStatus]
print('Removing {} rows that were failed copies'.format(len(failedCopies)))

metadataTable = metadataTable[metadataTable.CopyStatus]

species = list(metadataTable.Species)
uniqueSpecies = set(species)

print('Read {} unique species in {} rows'.format(len(uniqueSpecies),len(metadataTable)))

speciesMappings = {}

# keys should be lowercase
speciesMappings['blank'] = 'empty'
speciesMappings[''] = 'unlabeled'


#%% Enumerate images, confirm filename uniqueness

imageFullPaths = glob.glob(os.path.join(imageBaseDir,r'**\*.JPG'),recursive=True)

print('Counted {} images'.format(len(imageFullPaths)))

filenamesOnly = set()

for p in imageFullPaths:
    
    fn = ntpath.basename(p)
    assert fn not in filenamesOnly
    filenamesOnly.add(fn)
    
print('Finished uniqueness checking')


#%% Update metadata filenames to include site and camera folders, check existence
#
# Takes ~1min

filenamesToRows = {}

startTime = time.time()

newRows = []
matchFailures = []

# iRow = 0; row = metadataTable.iloc[iRow]
for iRow,row in tqdm.tqdm(metadataTable.iterrows(), total=metadataTable.shape[0]):
    
    baseFn = row['filename_new']
    station = row['Station']
    
    filenamesToRows[baseFn] = iRow
    
    # There's a bug in the metadata; the 'camera' column isn't correct.  
    # camera = row['Camera']
    # These appear as, e.g., '3.22e12'    
    # camera = str(int(float(camera)))
    
    # Let's pull this out of the file name instead
    #
    # Filenames look like one of the following:
    #
    # A1__03224850850507__2015-11-28__10-45-04(1).JPG
    # Bayama2PH__C05__NA(NA).JPG
    pat = '^(?P<station>.+?)__(?P<camera>.+?)__((?P<date>.+?)__)?(?P<time>[^_\()]+?)\((?P<frame>.+?)\)\.JPG'
    match = re.match(pat,baseFn)
    if match is None:
        raise ValueError('Regex failure at row {}: {}'.format(iRow,baseFn))
    assert(station == match.group('station'))
    camera = match.group('camera')
    row['Camera'] = camera
    
    assert match.group('station') is not None
    assert match.group('camera') is not None
    assert match.group('frame') is not None
    
    if match.group('date') is None:
        imgDate = ''
    else:
        imgDate = match.group('date')
        
    if match.group('time') is None:
        imgTime = ''
    else:
        imgTime = match.group('time')
    
    frame = -1
    try:
        frame = int(match.group['frame'])
    except:
        pass
    row['frameNumber'] = frame
    
    fn = os.path.join(station,camera,baseFn)
    fullPath = os.path.join(imageBaseDir,fn)
    row['filePath'] = fn
    # assert(os.path.isfile(fullPath))
    if not os.path.isfile(fullPath):
        print('Failed to match image {}'.format(fullPath))
        matchFailures.append(fullPath)
        continue
    
    # metadataTable.iloc[iRow] = row
    newRows.append(row)

elapsed = time.time() - startTime

# Re-assemble into an updated table
metadataTable = pd.DataFrame(newRows)

print('Finished checking file existence, extracting metadata in {}, couldn''t find {} images'.format(
      humanfriendly.format_timespan(elapsed),len(matchFailures)))
       
    
#%% Check for images that aren't included in the metadata file

imagesNotInMetadata = []

# Enumerate all images
for iImage,imagePath in enumerate(imageFullPaths):
    
    fn = ntpath.basename(imagePath)
    if(fn not in filenamesToRows):
        imagesNotInMetadata.append(imagePath)

print('Finished matching {} images, failed to match {}'.format(
        len(imageFullPaths),len(imagesNotInMetadata)))

# Write to a text file
with open(unmatchedImagesFile, 'w') as f:
    for fn in imagesNotInMetadata:
        f.write('{}\n'.format(fn))
        
        
#%% Create CCT dictionaries

# Also gets image sizes, so this takes ~6 minutes
#
# Implicitly checks images for overt corruptness, i.e. by not crashing.

images = []
annotations = []

# Map categories to integer IDs (that's what COCO likes)
nextCategoryID = 1
categoriesToCategoryId = {'empty':0}
categoriesToCounts = {'empty':0}

# For each image
#
# Because in practice images are 1:1 with annotations in this data set,
# this is also a loop over annotations.

startTime = time.time()

# iRow = 0; row = metadataTable.iloc[iRow]
for iRow,row in tqdm.tqdm(metadataTable.iterrows(), total=metadataTable.shape[0]):
    
    im = {}
        
    # A1__03224850850507__2015-11-28__10-45-04(1).JPG
    fn = row['filename_new']
    assert '.JPG' in fn
    fn = fn.replace('.JPG','')
    im['id'] = fn
    
    # 'A1\\03224850850507\\A1__03224850850507__2015-11-28__10-45-04(1).JPG'
    im['file_name'] = row['filePath']
    
    # Not currently populated
    im['seq_id'] = row['sequenceID']
    
    # Often -1, sometimes a semi-meaningful int
    im['frame_num'] = row['frameNumber']
    
    # A1
    im['site']= row['Station']    
    
    # 03224850850507
    im['camera'] = row['Camera']
    
    # In variable form, but sometimes '28/11/2015 10:45'
    im['datetime'] = row['DateTimeOriginal']
    
    images.append(im)
    
    # Check image height and width
    imagePath = os.path.join(imageBaseDir,im['file_name'])
    assert(os.path.isfile(imagePath))
    pilImage = PIL.Image.open(imagePath)
    width, height = pilImage.size
    im['width'] = width
    im['height'] = height

    category = row['Species'].lower()
    if category in speciesMappings:
        category = speciesMappings[category]
    
    # Have we seen this category before?
    if category in categoriesToCategoryId:
        categoryID = categoriesToCategoryId[category]
        categoriesToCounts[category] += 1
    else:
        categoryID = nextCategoryID
        categoriesToCategoryId[category] = categoryID
        categoriesToCounts[category] = 0
        nextCategoryID += 1
    
    # Create an annotation
    ann = {}
    
    # The Internet tells me this guarantees uniqueness to a reasonable extent, even
    # beyond the sheer improbability of collisions.
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']    
    ann['category_id'] = categoryID
    
    annotations.append(ann)
    
# ...for each image
    
# Convert categories to a CCT-style dictionary

categories = []

for category in categoriesToCounts:
    
    print('Category {}, count {}'.format(category,categoriesToCounts[category]))
    categoryID = categoriesToCategoryId[category]
    cat = {}
    cat['name'] = category
    cat['id'] = categoryID
    categories.append(cat)    
    
elapsed = time.time() - startTime

print('Finished creating CCT dictionaries in {}'.format(
      humanfriendly.format_timespan(elapsed)))
    

#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data,open(outputFile,'w'),indent=4)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
        len(images),len(annotations),len(categories)))


#%% Check database integrity

options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = imageBaseDir
options.bCheckImageSizes = False
options.bFindUnusedImages = False
sanity_check_json_db.sanity_check_json_db(outputFile, options)


#%% Preview a few images to make sure labels were passed along sensibly

db_path = outputFile
output_dir = os.path.join(baseDir,'label_preview')
image_base_dir = imageBaseDir
options = visualize_db.DbVizOptions()
options.num_to_visualize = 100
htmlOutputFile = visualize_db.process_images(db_path,output_dir,image_base_dir,options)
    

#%% One-time processing step: copy images to a flat directory for annotation

if False:
    
    #%%
    
    from shutil import copyfile
    os.makedirs(imageFlatDir,exist_ok=True)
    
    for sourcePath in tqdm.tqdm(imageFullPaths):
        fn = ntpath.basename(sourcePath)
        targetPath = os.path.join(imageFlatDir,fn)
        assert not os.path.isfile(targetPath)
        copyfile(sourcePath,targetPath)
        
    print('Copied {} files'.format(len(imageFullPaths)))
    
#
# sanity_check_json_db.py
#
# Does some sanity-checking and computes basic statistics on a db, specifically:
#
# * Verifies that required fields are present and have the right types
# * Verifies that annotations refer to valid images
# * Verifies that annotations refer to valid categories
# * Verifies that image, category, and annotation IDs are unique 
#
# * Optionally checks file existence
#
# * Finds un-annotated images
# * Finds unused categories
#
# * Prints a list of categories sorted by count

#%% Constants and environment

import json
import os
from tqdm import tqdm
from operator import itemgetter
from multiprocessing.pool import ThreadPool
from PIL import Image

nThreads = 10


#%% Functions

# If baseDir is non-empty, checks image existence
class SanityCheckOptions:
    
    baseDir = ''
    bCheckImageSizes = False
    bFindUnusedImages = False
    iMaxNumImages = -1
    
defaultOptions = SanityCheckOptions()


def checkImageExistenceAndSize(image,options=None):

    if options is None:
        
        options = defaultOptions
        
    filePath = os.path.join(options.baseDir,image['file_name'])
    if not os.path.isfile(filePath):
        print('Image path {} does not exist'.format(filePath))
        return False
    
    if options.bCheckImageSizes:
        if not ('height' in image and 'width' in image):
            print('Missing image size in {}'.format(filePath))
            return False

        width, height = Image.open(filePath).size
        if (not (width == image['width'] and height == image['height'])):
            print('Size mismatch for image {}: {} (reported {},{}, actual {},{})'.format(
                    image['id'], filePath, image['width'], image['height'], width, height))
            return False
        
    return True

  
def sanityCheckJsonDb(jsonFile, options=None):
    
    if options is None:   
        
        options = SanityCheckOptions()
    
    print(options.__dict__)
    
    assert os.path.isfile(jsonFile), '.json file {} does not exist'.format(jsonFile)
    
    print('\nProcessing .json file {}'.format(jsonFile))
    
    baseDir = options.baseDir
        
    ##%% Read .json file, sanity-check fields
    
    print('Reading .json {} with base dir [{}]...'.format(
            jsonFile,baseDir))
    
    with open(jsonFile,'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    # info = data['info']
    assert 'info' in data
    
    if len(baseDir) > 0:
        
        assert os.path.isdir(baseDir), 'Base directory {} does not exist'.format(baseDir)
        
    ##%% Build dictionaries, checking ID uniqueness and internal validity as we go
    
    imageIdToImage = {}
    annIdToAnn = {}
    catIdToCat = {}
    
    print('Checking categories...')
    
    for cat in tqdm(categories):
        
        # Confirm that required fields are present
        assert 'name' in cat
        assert 'id' in cat
        
        assert isinstance(cat['id'],int), 'Illegal category ID type'
        assert isinstance(cat['name'],str), 'Illegal category name type'
        
        catId = cat['id']
        
        # Confirm ID uniqueness
        assert catId not in catIdToCat
        catIdToCat[catId] = cat
        cat['_count'] = 0
        
    # ...for each category
        
    print('\nChecking images...')
    
    if options.iMaxNumImages > 0 and len(images) > options.iMaxNumImages:
        
        print('Trimming image list to {}'.format(options.iMaxNumImages))
        images = images[0:options.iMaxNumImages]
        
    imagePathsInJson = set()
    
    for image in tqdm(images):
        
        image['_count'] = 0
        
        # Confirm that required fields are present
        assert 'file_name' in image
        assert 'id' in image

        imagePathsInJson.add(image['file_name'])
        
        assert isinstance(image['file_name'],str), 'Illegal image filename type'
        assert isinstance(image['id'],str), 'Illegal image ID type'
        
        imageId = image['id']        
        
        # Confirm ID uniqueness
        assert imageId not in imageIdToImage, 'Duplicate image ID {}'.format(imageId)
        
        imageIdToImage[imageId] = image
        
        if 'height' in image:
            assert 'width' in image, 'Image with height but no width: {}'.format(image['id'])
        
        if 'width' in image:
            assert 'height' in image, 'Image with width but no height: {}'.format(image['id'])
    
    # Are we checking for unused images?
    if (len(baseDir) > 0) and options.bFindUnusedImages:    
        
        print('Enumerating images...')
        
        # Recursively enumerate images
        imagePaths = []
        for root, dirs, files in os.walk(baseDir):
            for file in files:
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    relDir = os.path.relpath(root, baseDir)
                    relFile = os.path.join(relDir,file)
                    if len(relFile) > 2 and \
                        (relFile[0:2] == './' or relFile[0:2] == '.\\'):                     
                            relFile = relFile[2:]
                    imagePaths.append(relFile)
          
        unusedFiles = []
        
        for p in imagePaths:
            if p not in imagePathsInJson:
                print('Image {} is unused'.format(p))
                unusedFiles.append(p)
                
    # Are we checking file existence and/or image size?
    if (len(baseDir) > 0) and options.bCheckImageSizes:
        
        print('Checking image existence and image sizes...')
        
        pool = ThreadPool(nThreads)
        # results = pool.imap_unordered(lambda x: fetch_url(x,nImages), indexedUrlList)
        defaultOptions.baseDir = options.baseDir
        defaultOptions.bCheckImageSizes = options.bCheckImageSizes
        results = tqdm(pool.imap(checkImageExistenceAndSize, images), total=len(images))
        
        for iImage,r in enumerate(results):
            if not r:
                print('Image validation error for image {}'.format(iImage))
                            
    # ...for each image
    
    print('Checking annotations...')
    
    for ann in tqdm(annotations):
    
        # Confirm that required fields are present
        assert 'image_id' in ann
        assert 'id' in ann
        assert 'category_id' in ann
        
        assert isinstance(ann['id'],str), 'Illegal annotation ID type'
        assert isinstance(ann['category_id'],int), 'Illegal annotation category ID type'
        assert isinstance(ann['image_id'],str), 'Illegal annotation image ID type'
        
        annId = ann['id']        
        
        # Confirm ID uniqueness
        assert annId not in annIdToAnn
        annIdToAnn[annId] = ann
    
        # Confirm validity
        assert ann['category_id'] in catIdToCat
        assert ann['image_id'] in imageIdToImage
    
        imageIdToImage[ann['image_id']]['_count'] += 1
        catIdToCat[ann['category_id']]['_count'] +=1 
        
    # ...for each annotation
        
        
    ##%% Print statistics
    
    # Find un-annotated images and multi-annotation images
    nUnannotated = 0
    nMultiAnnotated = 0
    
    for image in images:
        if image['_count'] == 0:
            nUnannotated += 1
        elif image['_count'] > 1:
            nMultiAnnotated += 1
            
    print('Found {} unannotated images, {} images with multiple annotations'.format(
            nUnannotated,nMultiAnnotated))
    
    nUnusedCategories = 0
    
    # Find unused categories
    for cat in categories:
        if cat['_count'] == 0:
            print('Unused category: {}'.format(cat['name']))
            nUnusedCategories += 1
    
    print('Found {} unused categories'.format(nUnusedCategories))
            
    print('\nDB contains {} images, {} annotations, and {} categories\n'.format(
            len(images),len(annotations),len(categories)))
    
    # Prints a list of categories sorted by count
    
    # https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary
    
    sortedCategories = sorted(categories, key=itemgetter('_count'), reverse=True)
    
    print('Categories and counts:\n')
    
    for cat in sortedCategories:
        print('{:6} {}'.format(cat['_count'],cat['name']))
    
    print('')
    
    return sortedCategories, data

# ...def sanityCheckJsonDb()
    

#%% Command-line driver
    
import argparse

def main():
    
    # python sanity_check_json_db.py "e:\wildlife_data\wellington_data\wellington_camera_traps.json" --baseDir "e:\wildlife_data\wellington_data\images" --bFindUnusedImages --bCheckImageSizes
    
    # Here the '-u' prevents buffering, which makes tee happier
    #
    # python -u sanity_check_json_db.py '/datadrive1/nacti_metadata.json' --baseDir '/datadrive1/nactiUnzip/' --bFindUnusedImages --bCheckImageSizes | tee ~/nactiTest.out
    
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonFile')
    parser.add_argument('--bCheckImageSizes', action='store_true')
    parser.add_argument('--bFindUnusedImages', action='store_true')
    parser.add_argument('--baseDir',action="store", type=str, default='')
    parser.add_argument('--iMaxNumImages',action="store", type=int, default=-1)
    
    args = parser.parse_args()    
    sanityCheckJsonDb(args.jsonFile,args)


if __name__ == '__main__':
    
    main()


#%% Interactive driver(s)

if False:
    
    #%%
    
    # Sanity-check .json files for LILA
    options = SanityCheckOptions()
    jsonFiles = [r'd:\temp\CaltechCameraTraps.json',
                 r'd:\temp\wellington_camera_traps.json',
                 r'd:\temp\nacti_metadata.json',
                 r'd:\temp\SnapshotSerengeti.json']
    
    # Sanity-check one file with all the bells and whistles
    jsonFiles = [r'e:\wildlife_data\wellington_data\wellington_camera_traps.json']; jsonFile = jsonFiles[0]; baseDir = r'e:\wildlife_data\wellington_data\images'
    options = SanityCheckOptions()
    options.baseDir = baseDir
    options.bCheckImageSizes = False
    options.bFindUnusedImages = True
    
    # options.iMaxNumImages = 10    
    
    for jsonFile in jsonFiles:
        
        sortedCategories,data = sanityCheckJsonDb(jsonFile, options)
        
    
      
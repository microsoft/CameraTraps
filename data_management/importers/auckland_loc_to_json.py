#
# auckland_loc_to_json.py
#
# Take a directory of images in which species labels are encoded by folder
# names, and produces a COCO-style .json file
#

# %% Constants and imports

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
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
baseDir = r'e:\wildlifeblobssc\Maukahuka_Auckland_Island'
outputJsonFilename = os.path.join(baseDir, 'auckland_loc.json')
outputCsvFilename = os.path.join(baseDir, 'auckland_loc.csv')


outputEncoding = 'utf-8'

bLoadFileListIfAvailable = False

info = {}
info['year'] = 2019
info['version'] = '1.0'
info['description'] = 'Auckaland Loc Camera Traps'
info['contributor'] = 'Vardhan Duvvuri'
info['date_created'] = str(datetime.date.today())

maxFiles = -1
bReadImageSizes = True
bUseExternalRemappingTable = False


# %% Enumerate files, read image sizes

# Each element will be a list of relative path/full path/width/height
fileInfo = []
nonImages = []
nFiles = 0

print('Enumerating files from {} to {}'.format(baseDir, outputCsvFilename))
image_files = find_images(baseDir, bRecursive=True)

print('Enumerated {} images'.format(len(image_files)))
with io.open(outputCsvFilename, "w", encoding=outputEncoding) as outputFileHandle:
    for fname in image_files:
        nFiles = nFiles + 1
        if maxFiles >= 0 and nFiles > maxFiles:
            print('Warning: early break at {} files'.format(maxFiles))
            break
        fullPath = fname
        relativePath = os.path.relpath(fullPath, baseDir)
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
                               '"' + fullPath + '"' + ',' + str(w) + ',' + str(h) + '\n')
        # ...for each image file
# ...csv file output
print("Finished writing {} file names to {}".format(nFiles, outputCsvFilename))

# %% Enumerate classes

# Maps classes to counts
classList = {}

for iRow, row in enumerate(fileInfo):
    try:
        fullPath = row[0]
        folder = row[0].split('\\')
        if folder[0] == '2_Testing':
            className = 'test'
        else:
            className = folder[-3]
        if className.startswith('2_'):
            className = className.replace('2_', '')
        className = className.lower().strip()
        if className in classList:
            classList[className] += 1
        else:
            classList[className] = 1
        row.append(className)
    except:
        print(row[0])
        import pdb;pdb.set_trace()

classNames = list(classList.keys())

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
for iRow, row in enumerate(fileInfo):

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
    if (className) != 'test':
        im['behavior'] = row[0].split('\\')[-2]
    images.append(im)
    imageIdToImage[im['id']] = im

    ann = {}
    ann['id'] = str(uuid.uuid1())
    ann['image_id'] = im['id']
    ann['category_id'] = categoryId
    annotations.append(ann)

    cat = idToCategory[categoryId]
    cat['count'] += 1

#%% Write output .json

data = {}
data['info'] = info
data['images'] = images
data['annotations'] = annotations
data['categories'] = categories

json.dump(data, open(outputJsonFilename, 'w'), indent=4)

print('Finished writing json to {}'.format(outputJsonFilename))


fn = outputJsonFilename
options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = baseDir
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True

sortedCategories, data = sanity_check_json_db.sanity_check_json_db(fn, options)


#%% Preview labels


viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = None
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file, image_db = visualize_db.process_images(db_path=outputJsonFilename,
                                                         output_dir=os.path.join(
                                                            baseDir, 'preview'),
                                                         image_base_dir=baseDir,
                                                        options=viz_options)
os.startfile(html_output_file)

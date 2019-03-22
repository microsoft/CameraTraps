#
# find_problematic_detections.py
#
# Looks through a sequence of detections in .csv+.json format, and finds candidates
# that might be "problematic false positives", i.e. that random branch that it
# really thinks is an animal.
#
# Currently the unit within which images are compared is a *directory*.
#

#%% Constants and imports

import csv
import os
import json
import jsonpickle
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
                        
# from ai4eutils; this is assumed to be on the path, as per repo convention
import write_html_image_list

# For bounding-box rendering functions
# 
# Definitely a heaviweight import just for this; some refactoring
# is likely necessary here.
from detection.run_tf_detector import render_bounding_box

# from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool
# import multiprocessing
# import joblib

inputCsvFilename = r'D:\temp\tigers_20190308_all_output.csv'
imageBase = r'd:\wildlife_data\tigerblobs'
outputBase = r'd:\temp\suspiciousDetections'

headers = ['image_path','max_confidence','detections']

# Don't consider detections with confidence lower than this as suspicious
confidenceThreshold = 0.85

# What's the IOU threshold for considering two boxes the same?
iouThreshold = 0.925

# How many occurrences of a single location (as defined by the IOU threshold)
# are required before we declare it suspicious?
occurrenceThreshold = 10

# Set to zero to disable parallelism
nWorkers = 10 # joblib.cpu_count()

debugMaxDir = -1
debugMaxRenderDir = -1
debugMaxDetection = -1
debugMaxInstance = -1
bParallelizeComparisons = True
bParallelizeRendering = True    

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings("ignore", "Metadata warning", UserWarning)
    

#%% Helper functions

def prettyPrintObject(obj,bPrint=True):
    '''
    Prints an arbitrary object as .json
    '''    
    # _ = prettyPrintObject(obj)
                
    # Sloppy that I'm making a module-wide change here...
    jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
    a = jsonpickle.encode(candidateLocation)
    s = '{}'.format(a)
    if bPrint:
        print(s)
    return s
    

imageExtensions = ['.jpg','.jpeg','.gif','.png']
    
def isImageFile(s):
    '''
    Check a file's extension against a hard-coded set of image file extensions    '
    '''
    ext = os.path.splitext(s)[1]
    return ext.lower() in imageExtensions
    
    
def findImageStrings(strings):
    '''
    Given a list of strings that are potentially image file names, look for strings
    that actually look like image file names (based on extension).
    '''
    imageStrings = []
    bIsImage = [False] * len(strings)
    for iString,f in enumerate(strings):
        bIsImage[iString] = isImageFile(f)
        if bIsImage[iString]:
            continue
        else:
            imageStrings.append(f)
    return imageStrings,bIsImage

    
def findImages(dirName):
    '''
    Find all files in a directory that look like image file names
    '''
    strings = os.listdir(dirName)
    return findImageStrings(strings)
    

# Adapted from:
#
# https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1: [x1,y1,x2,y2]
    bb2: [x1,y1,x2,y2]
    
    The (x1, y1) position is at the top left corner.
    The (x2, y2) position is at the bottom right corner.
    
    Returns
    -------
    float in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]    

    # Determine the coordinates of the intersection rectangle    
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area.
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


#%% Load file

# Each row is filename, max confidence, bounding box info

allRows = []

with open(inputCsvFilename) as f:
    reader = csv.reader(f, delimiter=',')
    iRow = 0
    for row in reader:
        iRow += 1
        assert(len(row) == 3)
        # Parse the detection info into an array
        if iRow > 1:
            row[2] = json.loads(row[2])
        allRows.append(row)

# [ymin, xmin, ymax, xmax, confidence], where (xmin, ymin) is the upper-left

# Remove header row
headerRow = allRows[0]
assert(headerRow == headers)        
allRows = allRows[1:]

print('Read {} rows from {}'.format(len(allRows),inputCsvFilename))


#%% Separate files into directories

rowsByDirectory = {}

# row = allRows[0]
for row in allRows:
    
    relativePath = row[0]
    dirName = os.path.dirname(relativePath)
    
    if not dirName in rowsByDirectory:
        rowsByDirectory[dirName] = []
        
    rowsByDirectory[dirName].append(row)
    
print('Finished separating {} files into {} directories'.format(len(allRows),
      len(rowsByDirectory)))


#%% Look for matches (one directory)

class IndexedDetection:
    
    iRow = -1
    iDetection = -1
    filename = ''
    bbox = []
    
    def __init__(self, iRow, iDetection, filename, bbox):
        self.iRow = iRow
        self.iDetection = iDetection
        self.filename = filename
        self.bbox = bbox

    def __repr__(self):
        s = prettyPrintObject(self,False)
        return s
    
class DetectionLocation:
    
    # list of IndexedDetections
    instances = []
    bbox = []

    def __init__(self, instance, bbox):
        self.instances = [instance]
        self.bbox = bbox
    
    def __repr__(self):
        s = prettyPrintObject(self,False)
        return s
    
pbar = None

def findMatchesInDirectory(dirName):

    if pbar is not None:
        pbar.update()
        
    # List of DetectionLocations
    candidateDetections = []
    
    rows = rowsByDirectory[dirName]
    
    for iRow,row in enumerate(rows):
    
        filename = row[0]
        if not isImageFile(filename):
            continue
        
        # For each detection above threshold
        maxP = float(row[1])
        
        if maxP >= confidenceThreshold:
        
            # Array of arrays, where each element is:
            #
            # [ymin, xmin, ymax, xmax, confidence], where (xmin, ymin) is the upper-left            
            detections = row[2]
            assert len(detections) > 0
                        
            # For each detection in this image
            for iDetection,detection in enumerate(detections):
                
                instance = IndexedDetection(iRow,iDetection,row[0],detection)
                        
                bFoundSimilarDetection = False
                
                # For each detection in our candidate list
                for iCandidate,candidate in enumerate(candidateDetections):
                
                    # Is this a match?                    
                    iou = get_iou(detection,candidate.bbox)
                    
                    if iou >= iouThreshold:                                        
                        
                        bFoundSimilarDetection = True
                        
                        # If so, add this example to the list for this detection
                        candidate.instances.append(instance)
                                                
                # If we found no matches, add this to the candidate list
                if not bFoundSimilarDetection:
                    
                    candidate = DetectionLocation(instance, detection)
                    candidateDetections.append(candidate)

    return candidateDetections


#%% Look for matches

# For each directory
if debugMaxDir > 0:
    print('TRIMMING TO {} DIRECTORIES FOR DEBUGGING'.format(debugMaxDir))
    
dirsToSearch = list(rowsByDirectory.keys())[0:debugMaxDir]
    
allCandidateDetections = [None] * len(dirsToSearch)

if bParallelizeComparisons:
        
    pbar = None
    for iDir,dirName in enumerate(tqdm(dirsToSearch)):        
        allCandidateDetections[iDir] = findMatchesInDirectory(dirName)
         
else:

    pbar = tqdm(total=len(dirsToSearch))
    allCandidateDetections = Parallel(n_jobs=nWorkers,prefer='threads')(delayed(findMatchesInDirectory)(dirName) for dirName in tqdm(dirsToSearch))
        
print('Finished looking for similar bounding boxes')    
    

#%% Find suspicious locations based on match results

suspiciousDetections = [None] * len(dirsToSearch)

nImagesWithSuspiciousDetections = 0
nSuspiciousDetections = 0

# For each directory
#
# iDir = 51
for iDir in range(len(dirsToSearch)):

    # A list of DetectionLocation objects
    suspiciousDetectionsThisDir = []    
    
    # A list of DetectionLocation objects
    candidateDetectionsThisDir = allCandidateDetections[iDir]
    
    for iLocation,candidateLocation in enumerate(candidateDetectionsThisDir):
        
        # occurrenceList is a list of file/detection pairs
        nOccurrences = len(candidateLocation.instances)
        
        if nOccurrences < occurrenceThreshold:
            continue
        
        nImagesWithSuspiciousDetections += nOccurrences
        nSuspiciousDetections += 1
        
        suspiciousDetectionsThisDir.append(candidateLocation)
        # Find the images corresponding to this bounding box, render boxes
    
    suspiciousDetections[iDir] = suspiciousDetectionsThisDir

print('Finished searching for problematic detections, found {} unique detections on {} images that are suspicious'.format(
  nSuspiciousDetections,nImagesWithSuspiciousDetections))    


#%% Render problematic locations with html

plt.close('all')

def renderImagesForDirectory(iDir):
    
    if pbar is not None:
        pbar.update()
    
    if debugMaxRenderDir > 0 and iDir > debugMaxRenderDir:
        print('WARNING: DEBUG BREAK AFTER RENDERING {} DIRS'.format(debugMaxRenderDir))
        return None
    
    dirName = 'dir{:0>4d}'.format(iDir)
    
    # suspiciousDetectionsThisDir is a list of DetectionLocation objects
    suspiciousDetectionsThisDir = suspiciousDetections[iDir]
    
    if len(suspiciousDetectionsThisDir) == 0:
        return None
    
    print('Processing directory {} of {}'.format(iDir,nDirs))
    
    dirBaseDir = os.path.join(outputBase,dirName)
    os.makedirs(dirBaseDir,exist_ok=True)
    
    directoryDetectionHtmlFiles = []
    directoryDetectionExampleImages = []
    directoryDetectionTitles = []
        
    # For each problematic detection in this directory
    #
    # iDetection = 0; detection = suspiciousDetectionsThisDir[iDetection];
    nDetections = len(suspiciousDetectionsThisDir)
    for iDetection,detection in enumerate(suspiciousDetectionsThisDir):
    
        if debugMaxDetection > 0 and iDetection > debugMaxDetection:
            print('WARNING: DEBUG BREAK AFTER RENDERING {} DETECTIONS'.format(debugMaxDetection))
            break
        
        nInstances = len(detection.instances)
        print('Processing detection {} of {} ({} instances)'.format(
                iDetection,nDetections,nInstances))
        detectionName = 'detection{:0>4d}'.format(iDetection)
        detectionBaseDir = os.path.join(dirBaseDir,detectionName)
        os.makedirs(detectionBaseDir,exist_ok=True)
        
        # _ = prettyPrintObject(detection)
        assert(nInstances >= occurrenceThreshold)
                        
        imageFileNames = []
        titles = []
        
        # Render images
        
        # iInstance = 0; instance = detection.instances[iInstance]
        for iInstance,instance in enumerate(detection.instances):
            
            if debugMaxInstance > 9 and iInstance > debugMaxInstance:
                print('WARNING: DEBUG BREAK AFTER RENDERING {} INSTANCES'.format(debugMaxInstance))
                break
            
            imageRelativeFilename = 'image{:0>4d}.jpg'.format(iInstance)
            imageOutputFilename = os.path.join(detectionBaseDir,
                                               imageRelativeFilename)
            imageFileNames.append(imageRelativeFilename)
            titles.append(instance.filename)
            # Render bounding boxes
            # def render_bounding_boxes(boxes, scores, classes, inputFileNames, outputFileNames=[],
            #              confidenceThreshold=0.9):
            
            """
            Render bounding boxes on the image files specified in [inputFileNames].  
            
            [boxes] and [scores] should be in the format returned by generate_detections, 
            specifically [top, left, bottom, right] in normalized units, where the
            origin is the upper-left.
            """

            inputFileName = os.path.join(imageBase,instance.filename)
            assert(os.path.isfile(inputFileName))
            render_bounding_box(instance.bbox,1,None,inputFileName,imageOutputFilename,0,10)

        # ...for each instance

        # Write html for this detection
        detectionHtmlFile = os.path.join(detectionBaseDir,'index.html')
        
        options = write_html_image_list.write_html_image_list()
        options['imageStyle'] = 'max-width:700px;'
        write_html_image_list.write_html_image_list(detectionHtmlFile, 
                                                    imageFileNames, titles, options)

        directoryDetectionHtmlFiles.append(detectionHtmlFile)
        directoryDetectionExampleImages.append(os.path.join(detectionName,imageFileNames[0]))
        title = '<a href="{}">{}</a>'.format(detectionHtmlFile,detectionName)
        directoryDetectionTitles.append(title)

    # ...for each detection
        
    # Write the html file for this directory
    directoryHtmlFile = os.path.join(dirBaseDir,'index.html')
    
    write_html_image_list.write_html_image_list(directoryHtmlFile, 
                                                    directoryDetectionExampleImages, 
                                                    directoryDetectionTitles, options)    
    
    return directoryHtmlFile

nDirs = len(dirsToSearch)
directoryHtmlFiles = [None] * nDirs

if bParallelizeRendering:

    pbar = tqdm(total=nDirs)
    # pbar = None
    
    directoryHtmlFiles = Parallel(n_jobs=nWorkers,prefer='threads')(delayed(
            renderImagesForDirectory)(iDir) for iDir in tqdm(range(nDirs)))
    
else:

    pbar = None
    
    # For each directory
    # iDir = 51
    for iDir in range(nDirs):
                
        # Add this directory to the master list of html files
        directoryHtmlFiles[iDir] = renderImagesForDirectory(iDir)
    
    # ...for each directory

#%% 
    
# Write master html file
masterHtmlFile = os.path.join(outputBase,'index.html')   

with open(masterHtmlFile,'w') as fHtml:
    
    fHtml.write('<html><body>\n')
    fHtml.write('<h2><b>Suspicious detections by directory</b></h2></br>\n')
    for iDir,dirHtmlFile in enumerate(directoryHtmlFiles):
        if dirHtmlFile is None:
            continue
        relPath = os.path.relpath(dirHtmlFile,outputBase)
        dirName = dirsToSearch[iDir]
        fHtml.write('<a href={}>{}</a><br/>\n'.format(relPath,dirName))
    fHtml.write('</body></html>\n')


#%% Write .csv output
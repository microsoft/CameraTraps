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
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from joblib import Parallel, delayed

# from ai4eutils; this is assumed to be on the path, as per repo convention
import write_html_image_list

# For bounding-box rendering functions
# 
# Definitely a heaviweight import just for this; some refactoring
# is likely necessary here.
from detection.run_tf_detector import render_bounding_boxes

# from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool
# import multiprocessing
# import joblib

inputCsvFilename = r'D:\temp\WIItigers_20190308_all_output.csv'
imageBase = r'D:\wildlife_data\tigerblobs'

headers = ['image_path','max_confidence','detections']

confidenceThreshold = 0.8
iouThreshold = 0.95

# Set to zero to disable parallelism
nWorkers = 8 # joblib.cpu_count()

    
#%% Helper functions

imageExtensions = ['.jpg','.jpeg','.gif','.png']
    
def isImageFile(s):
    
    ext = os.path.splitext(s)[1]
    return ext.lower() in imageExtensions
    
    
def findImageStrings(strings):
    
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
    
print('Finished separating {} files into {} directories'.format(len(allRows),len(rowsByDirectory)))


#%% Look for matches (one directory)

pbar = None

def findMatchesInDirectory(dirName):

    if pbar is not None:
        pbar.update()
        
    # A list of all unique-ish bounding boxes 
    candidateDetections = []
    
    # For each bounding box, a list of image/index pairs where we saw this bounding box
    candidateDetectionOccurrenceIndices = []
    
    rows = rowsByDirectory[dirName]
    
    for iRow,row in enumerate(rows):
    
        filename = row[0]
        if not isImageFile(filename):
            continue
        
        # For each detection above threshold
        maxP = float(row[1])
        
        if maxP >= confidenceThreshold:
        
            detections = row[2]
            assert len(detections) > 0
            
            # [ymin, xmin, ymax, xmax, confidence], where (xmin, ymin) is the upper-left
            
            # For each detection in this image
            for iDetection,detection in enumerate(detections):
                
                bFoundSimilarDetection = False
                
                # For each detection in our candidate list
                for iCandidate,candidate in enumerate(candidateDetections):
                
                    # Is this a match?
                    iou = get_iou(detection,candidate)
                    
                    if iou >= iouThreshold:                                        
                        
                        bFoundSimilarDetection = True
                        
                        # If so, add this example to the list for this detection
                        candidateDetectionOccurrenceIndices[iCandidate].append([iRow,iDetection])
                        
                                            
                # If we found no matches, add this to the candidate list
                if not bFoundSimilarDetection:
                    
                    candidateDetections.append(detection)
                    candidateDetectionOccurrenceIndices.append([[iRow,iDetection]])

    return candidateDetections,candidateDetectionOccurrenceIndices


#%% Look for matches

# For each directory
# dirName = (list(rowsByDirectory.keys()))[0]

dirsToSearch = rowsByDirectory
# dirsToSearch = list(rowsByDirectory.keys())[0:100]
    
allCandidateDetections = [None] * len(dirsToSearch)
allOccurrenceIndices = [None] * len(dirsToSearch)

if nWorkers == 0:
        
    for iDir,dirName in enumerate(tqdm(dirsToSearch)):
        
        candidateDetections,candidateDetectionOccurrenceIndices = findMatchesInDirectory(dirName)
        allCandidateDetections[iDir] = candidateDetections
        allOccurrenceIndices[iDir] = candidateDetectionOccurrenceIndices

else:

    pbar = tqdm(total=len(dirsToSearch))
    allResults = Parallel(n_jobs=nWorkers,prefer='threads')(delayed(findMatchesInDirectory)(dirName) for dirName in tqdm(dirsToSearch))

    for iResult,result in enumerate(allResults):
        allCandidateDetections[iResult] = result[0]
        allOccurrenceIndices[iResult] = result[1]
        
print('Finished looking for similar bounding boxes')    
    

#%% Find potentially-problematic locations

# For each directory

occurrenceThreshold = 10

allProblematicDetections = []

nImagesWithSuspiciousDetections = 0
nSuspiciousDetections = 0

# iDir = 0
for iDir in range(len(dirsToSearch)):

    problematicDetectionsThisDir = []    
    
    # Each element is a unique-ish detection, represented as a list of 
    # image/index pairs where we saw that detection
    candidateDetectionOccurrenceIndices = allOccurrenceIndices[iDir]
    
    for iCandidate,occurenceList in enumerate(candidateDetectionOccurrenceIndices):
        
        # occurrenceList is a list of file/detection pairs
        nOccurrences = len(occurenceList)
        
        if nOccurrences < occurrenceThreshold:
            continue
        
        nImagesWithSuspiciousDetections += nOccurrences
        nSuspiciousDetections += 1
        
        problematicDetectionsThisDir.append(candidateDetectionOccurrenceIndices)
        # Find the images corresponding to this bounding box, render boxes
                
    allProblematicDetections.append(problematicDetectionsThisDir)

print('Finished searching for problematic detections, found {} unique detections on {} images that are suspicious'.format(
        nSuspiciousDetections,nImagesWithSuspiciousDetections))    


#%% Render problematic locations
"""    
def render_bounding_boxes(boxes, scores, classes, inputFileNames, outputFileNames=[],
                          confidenceThreshold=0.9):
"""    
    """
    Render bounding boxes on the image files specified in [inputFileNames].  
    
    [boxes] and [scores] should be in the format returned by generate_detections, 
    specifically [top, left, bottom, right] in normalized units, where the
    origin is the upper-left.
    """

#%% Write html for those images
    
# def write_html_image_list(filename=None,imageFilenames=None,titles=(),options={}):
options = write_html_image_list.write_html_image_list()
options['imageStyle'] = 'max-width:700px;'
write_html_image_list.write_html_image_list(OUTPUT_HTML_TEST_FILE, imageFilenames, titles, options)
os.startfile(OUTPUT_HTML_TEST_FILE)

#%% Write output
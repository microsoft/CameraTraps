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

inputCsvFilename = r'D:\temp\WIItigers_20190308_all_output.csv'
imageBase = r'D:\wildlife_data\tigerblobs'

headers = ['image_path','max_confidence','detections']

confidenceThreshold = 0.8
iouThreshold = 0.95


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


#%% Look for matches

# For each directory
# dirName = (list(rowsByDirectory.keys()))[0]
for dirName in rowsByDirectory:
    
    #%%

    candidateDetections = []
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
                        
                        assert not bFoundSimilarDetection
                        
                        bFoundSimilarDetection = True
                        
                        # If so, add this example to the list for this detection
                        candidateDetectionOccurrenceIndices[iCandidate].append([iRow,iDetection])
                        
                                            
                # If we found no matches, add this to the candidate list
                if not bFoundSimilarDetection:
                    
                    candidateDetections.append(detection)
                    candidateDetectionOccurrenceIndices.append([[iRow,iDetection]])
                
print('Finished looking for similar bounding boxes')    
    

#%% Render problematic locations
    

#%% Write output
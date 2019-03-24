#
# find_problematic_detections.py
#
# Looks through a sequence of detections in .csv+.json format, and finds candidates
# that might be "problematic false positives", i.e. that random branch that it
# really thinks is an animal.
#
# Currently the unit within which images are compared is a *directory*.
#

#%% Imports and environment

import csv
import os
import json
import jsonpickle
import warnings

from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime
from PIL import Image, ImageDraw
      
# from ai4eutils; this is assumed to be on the path, as per repo convention
import write_html_image_list

# Imports I'm not using but use when I tinker with parallelization
#
# from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool
# import multiprocessing
# import joblib

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings("ignore", "Metadata warning", UserWarning)


#%% Classes

class SuspiciousDetectionOptions:
    
    # inputCsvFilename = r'D:\temp\tigers_20190308_all_output.csv'
    
    # Only relevant for HTML rendering
    imageBase = ''
    outputBase = ''
    
    # Don't consider detections with confidence lower than this as suspicious
    confidenceThreshold = 0.85
    
    # What's the IOU threshold for considering two boxes the same?
    iouThreshold = 0.925
    
    # How many occurrences of a single location (as defined by the IOU threshold)
    # are required before we declare it suspicious?
    occurrenceThreshold = 10
    
    # Set to zero to disable parallelism
    nWorkers = 10 # joblib.cpu_count()
    
    # Ignore "suspicious" detections larger than some size; these are often animals
    # taking up the whole image.  This is expressed as a fraction of the image size.
    maxSuspiciousDetectionSize = 0.35
    
    bRenderHtml = False
    
    debugMaxDir = -1
    debugMaxRenderDir = -1
    debugMaxRenderDetection = -1
    debugMaxRenderInstance = -1
    bParallelizeComparisons = True
    bParallelizeRendering = True
        
    # State variables
    pbar = None

    # Constants
    expectedHeaders = ['image_path','max_confidence','detections']
    
    
class SuspiciousDetectionResults:

    allRows = None
    rowsByDirectory = None
    suspiciousDetections = None
    masterHtmlFile = None
    

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
    

#%% Helper functions

def render_bounding_box(bbox,inputFileName,imageOutputFilename,linewidth):  
    
    im = Image.open(inputFileName)
    imW = im.width; imH = im.height
    
    # [top, left, bottom, right] in normalized units, where the origin is the upper-left.
    draw = ImageDraw.Draw(im)
    x0 = bbox[1] * imW; x1 = bbox[3] * imW
    y0 = bbox[0] * imH; y1 = bbox[2] * imH
    draw.rectangle([x0,y0,x1,y1],width=linewidth,outline='#ff0000')
    del draw
    im.save(imageOutputFilename)
    

def prettyPrintObject(obj,bPrint=True):
    '''
    Prints an arbitrary object as .json
    '''    
    # _ = prettyPrintObject(obj)
                
    # Sloppy that I'm making a module-wide change here...
    jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
    a = jsonpickle.encode(obj)
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


#%% Look for matches (one directory) (function)

def findMatchesInDirectory(dirName,options,rowsByDirectory):

    if options.pbar is not None:
        options.pbar.update()
        
    # List of DetectionLocations
    candidateDetections = []
    
    rows = rowsByDirectory[dirName]
    
    # iRow = 0; row = rows[iRow]
    for iRow,row in enumerate(rows):
    
        filename = row[0]
        if not isImageFile(filename):
            continue
        
        # Don't bother checking images with no detections above threshold
        maxP = float(row[1])        
        if maxP < options.confidenceThreshold:
            continue
        
        # Array of arrays, where each element is:
        #
        # [ymin, xmin, ymax, xmax, confidence], where (xmin, ymin) is the upper-left            
        detections = row[2]
        assert len(detections) > 0
        
        # For each detection in this image
        for iDetection,detection in enumerate(detections):
            
            assert len(detection) == 5
            
            # Is this detection too big to be suspicious?
            h = detection[2] - detection[0]
            w = detection[3] - detection[1]
            area = h * w
            
            # These are relative coordinates
            assert area >= 0.0 and area <= 1.0
            
            if area > options.maxSuspiciousDetectionSize:
                # print('Ignoring very large detection with area {}'.format(area))
                continue
        
            confidence = detection[4]
            assert confidence >= 0.0 and confidence <= 1.0
            if confidence < options.confidenceThreshold:
                continue
            
            instance = IndexedDetection(iRow,iDetection,row[0],detection)
                    
            bFoundSimilarDetection = False
            
            # For each detection in our candidate list
            for iCandidate,candidate in enumerate(candidateDetections):
            
                # Is this a match?                    
                iou = get_iou(detection,candidate.bbox)
                
                if iou >= options.iouThreshold:                                        
                    
                    bFoundSimilarDetection = True
                    
                    # If so, add this example to the list for this detection
                    candidate.instances.append(instance)
                    
            # ...for each detection on our candidate list
            
            # If we found no matches, add this to the candidate list
            if not bFoundSimilarDetection:
                
                candidate = DetectionLocation(instance, detection)
                candidateDetections.append(candidate)

        # ...for each detection
                
    # ...for each row
    
    return candidateDetections

# ...def findMatchesInDirectory(dirName)
    

#%% Render problematic locations to html (function)

def renderImagesForDirectory(iDir,directoryHtmlFiles,suspiciousDetections,options):
    
    nDirs = len(directoryHtmlFiles)
    
    if options.pbar is not None:
        options.pbar.update()
    
    if options.debugMaxRenderDir > 0 and iDir > options.debugMaxRenderDir:
        return None
    
    dirName = 'dir{:0>4d}'.format(iDir)
    
    # suspiciousDetectionsThisDir is a list of DetectionLocation objects
    suspiciousDetectionsThisDir = suspiciousDetections[iDir]
    
    if len(suspiciousDetectionsThisDir) == 0:
        return None
    
    timeStr = datetime.now().strftime('%H:%M:%S')
    print('Processing directory {} of {} ({})'.format(iDir,nDirs,timeStr))
    
    dirBaseDir = os.path.join(options.outputBase,dirName)
    os.makedirs(dirBaseDir,exist_ok=True)
    
    directoryDetectionHtmlFiles = []
    directoryDetectionExampleImages = []
    directoryDetectionTitles = []
        
    # For each problematic detection in this directory
    #
    # iDetection = 0; detection = suspiciousDetectionsThisDir[iDetection];
    nDetections = len(suspiciousDetectionsThisDir)
    for iDetection,detection in enumerate(suspiciousDetectionsThisDir):
    
        if options.debugMaxRenderDetection > 0 and iDetection > options.debugMaxRenderDetection:
            break
        
        nInstances = len(detection.instances)
        print('Processing detection {} of {} ({} instances)'.format(
                iDetection,nDetections,nInstances))
        detectionName = 'detection{:0>4d}'.format(iDetection)
        detectionBaseDir = os.path.join(dirBaseDir,detectionName)
        os.makedirs(detectionBaseDir,exist_ok=True)
        
        # _ = prettyPrintObject(detection)
        assert(nInstances >= options.occurrenceThreshold)
                        
        imageFileNames = []
        titles = []
        
        # Render images
        
        # iInstance = 0; instance = detection.instances[iInstance]
        for iInstance,instance in enumerate(detection.instances):
            
            if options.debugMaxRenderInstance > 9 and iInstance > options.debugMaxRenderInstance:
                break
            
            imageRelativeFilename = 'image{:0>4d}.jpg'.format(iInstance)
            imageOutputFilename = os.path.join(detectionBaseDir,
                                               imageRelativeFilename)
            imageFileNames.append(imageRelativeFilename)
            confidence = instance.bbox[4]
            confidenceStr = '{:.2f}'.format(confidence)
            t = confidenceStr + ' (' + instance.filename + ')'
            titles.append(t)
            
            inputFileName = os.path.join(options.imageBase,instance.filename)
            if not os.path.isfile(inputFileName):
                print('Warning: could not find file {}'.format(inputFileName))
            else:
                # render_bounding_box(instance.bbox,1,None,inputFileName,imageOutputFilename,0,10)
                render_bounding_box(instance.bbox,inputFileName,imageOutputFilename,15)
                
        # ...for each instance

        # Write html for this detection
        detectionHtmlFile = os.path.join(detectionBaseDir,'index.html')
        
        htmlOptions = write_html_image_list.write_html_image_list()
        htmlOptions['imageStyle'] = 'max-width:700px;'
        write_html_image_list.write_html_image_list(detectionHtmlFile, 
                                                    imageFileNames, titles, htmlOptions)

        directoryDetectionHtmlFiles.append(detectionHtmlFile)
        directoryDetectionExampleImages.append(os.path.join(detectionName,imageFileNames[0]))
        detectionHtmlFileRelative = os.path.relpath(detectionHtmlFile,dirBaseDir)
        title = '<a href="{}">{}</a>'.format(detectionHtmlFileRelative,detectionName)
        directoryDetectionTitles.append(title)

    # ...for each detection
        
    # Write the html file for this directory
    directoryHtmlFile = os.path.join(dirBaseDir,'index.html')
    
    write_html_image_list.write_html_image_list(directoryHtmlFile, 
                                                    directoryDetectionExampleImages, 
                                                    directoryDetectionTitles, htmlOptions)    
    
    return directoryHtmlFile

# ...def renderImagesForDirectory(iDir)
    

#%% Main function
    
def findSuspiciousDetections(inputCsvFilename,outputCsvFilename,options=None):
    
    ##%% Input handling
    
    if options is None:
        options = SuspiciousDetectionOptions()

    toReturn = SuspiciousDetectionResults()
            
    ##%% Load file
    
    # Each row is filename, max confidence, bounding box info
    
    allRows = []
    
    print('Reading input file {}'.format(inputCsvFilename))
    
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
    assert(headerRow == options.expectedHeaders)        
    allRows = allRows[1:]
    toReturn.allRows = allRows
    
    print('Read {} rows from {}'.format(len(allRows),inputCsvFilename))


    ##%% Separate files into directories
    
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
    
    toReturn.rowsByDirectory = rowsByDirectory
    
    
    ##%% Look for matches
    
    # For each directory
        
    dirsToSearch = list(rowsByDirectory.keys())[0:options.debugMaxDir]
        
    allCandidateDetections = [None] * len(dirsToSearch)
    
    if not options.bParallelizeComparisons:
            
        options.pbar = None
        # iDir = 0; dirName = dirsToSearch[iDir]
        for iDir,dirName in enumerate(tqdm(dirsToSearch)):        
            allCandidateDetections[iDir] = findMatchesInDirectory(dirName,options,rowsByDirectory)
             
    else:
    
        options.pbar = tqdm(total=len(dirsToSearch))
        allCandidateDetections = Parallel(n_jobs=options.nWorkers,prefer='threads')(delayed(findMatchesInDirectory)(dirName,options,rowsByDirectory) for dirName in tqdm(dirsToSearch))
            
    print('Finished looking for similar bounding boxes')    
    

    ##%% Find suspicious locations based on match results
    
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
            
            if nOccurrences < options.occurrenceThreshold:
                continue
            
            nImagesWithSuspiciousDetections += nOccurrences
            nSuspiciousDetections += 1
            
            suspiciousDetectionsThisDir.append(candidateLocation)
            # Find the images corresponding to this bounding box, render boxes
        
        suspiciousDetections[iDir] = suspiciousDetectionsThisDir
    
    print('Finished searching for problematic detections, found {} unique detections on {} images that are suspicious'.format(
      nSuspiciousDetections,nImagesWithSuspiciousDetections))    
    
    toReturn.suspiciousDetections = suspiciousDetections
    
    if options.bRenderHtml:
        
        ##%% Render problematic locations with html (loop)
    
        nDirs = len(dirsToSearch)
        directoryHtmlFiles = [None] * nDirs
              
        if options.bParallelizeRendering:
        
            # options.pbar = tqdm(total=nDirs)
            options.pbar = None
            
            directoryHtmlFiles = Parallel(n_jobs=options.nWorkers,prefer='threads')(delayed(
                    renderImagesForDirectory)(iDir,directoryHtmlFiles,suspiciousDetections,options) for iDir in tqdm(range(nDirs)))
            
        else:
        
            options.pbar = None
            
            # For each directory
            # iDir = 51
            for iDir in range(nDirs):
                        
                # Add this directory to the master list of html files
                directoryHtmlFiles[iDir] = renderImagesForDirectory(iDir,directoryHtmlFiles,suspiciousDetections,options)
            
            # ...for each directory


        ##%% Write master html file
            
        masterHtmlFile = os.path.join(options.outputBase,'index.html')   
        toReturn.masterHtmlFile = masterHtmlFile
        
        with open(masterHtmlFile,'w') as fHtml:
            
            fHtml.write('<html><body>\n')
            fHtml.write('<h2><b>Suspicious detections by directory</b></h2></br>\n')
            for iDir,dirHtmlFile in enumerate(directoryHtmlFiles):
                if dirHtmlFile is None:
                    continue
                relPath = os.path.relpath(dirHtmlFile,options.outputBase)
                dirName = dirsToSearch[iDir]
                fHtml.write('<a href={}>{}</a><br/>\n'.format(relPath,dirName))
            fHtml.write('</body></html>\n')

    # ...if we're rendering html
    
    
    ##%% Write revised .csv output
    
    return toReturn

# ...findSuspiciousDetections()
    
#%% Interactive driver
    
if False:

    #%%     
    
    options = SuspiciousDetectionOptions()
    options.bRenderHtml = True
    options.imageBase = r'd:\wildlife_data\tigerblobs'
    options.outputBase = r'd:\temp\suspiciousDetections'
    
    options.debugMaxDir = -1
    options.debugMaxRenderDir = -1
    options.debugMaxRenderDetection = -1
    options.debugMaxRenderInstance = -1
    
    inputCsvFilename = r'D:\temp\tigers_20190308_all_output.csv'
    outputCsvFilename = ''
    
    # def findSuspiciousDetections(inputCsvFilename,outputCsvFilename,options=None)
    findSuspiciousDetections(inputCsvFilename,outputCsvFilename,options)
    
    
#%% Command-line driver

import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFile')
    parser.add_argument('outputFile')
    parser.add_argument('--imageBase', action='store', type=str, 
                        help='Image base dir, only relevant if renderHtml is True')
    parser.add_argument('--outputBase', action='store', type=str, 
                        help='Html output dir, only relevant if renderHtml is True')
    parser.add_argument('--confidenceThreshold',action="store", type=float, default=0.85, 
                        help='Detection confidence threshold; don\'t process anything below this')
    parser.add_argument('--occurrenceThreshold',action="store", type=int, default=10, 
                        help='More than this many near-identical detections in a group (e.g. a folder) is considered suspicious')
    parser.add_argument('--nWorkers',action="store", type=int, default=10, 
                        help='Level of parallelism for rendering or IOU computation')
    parser.add_argument('--maxSuspiciousDetectionSize',action="store", type=float, 
                        default=0.35, help='Detections larger than this fraction of image area are not considered suspicious')
    parser.add_argument('--renderHtml', action='store', type=bool, default=False, 
                        dest='bRenderHtml', help='Should we render HTML output?')
    
    parser.add_argument('--debugMaxDir', action='store', type=int, default=-1)                        
    parser.add_argument('--debugMaxRenderDir', action='store', type=int, default=-1)
    parser.add_argument('--debugMaxRenderDetection', action='store', type=int, default=-1)
    parser.add_argument('--debugMaxRenderInstance', action='store', type=int, default=-1)
    
    parser.add_argument('--bParallelizeComparisons', action='store', type=bool, default=True)
    parser.add_argument('--bParallelizeRendering', action='store', type=bool, default=True)
    
    args = parser.parse_args()    
    
    findSuspiciousDetections(args.inputFile,args.outputFile,options)

if __name__ == '__main__':
    
    main()

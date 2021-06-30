########
#
# repeat_detections_core.py
#
# Core utilities shared by find_repeat_detections and remove_repeat_detections.
#
########

# %% Imports and environment

import os
import warnings
from datetime import datetime
from itertools import compress

import jsonpickle
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# from ai4eutils; this is assumed to be on the path, as per repo convention
import write_html_image_list
import path_utils

from api.batch_processing.postprocessing.load_api_results import load_api_results, write_api_results
from api.batch_processing.postprocessing.postprocess_batch_results import is_sas_url
from api.batch_processing.postprocessing.postprocess_batch_results import relative_sas_url

from visualization.visualization_utils import open_image, render_detection_bounding_boxes
import ct_utils

# Imports I'm not using but use when I tinker with parallelization
#
# from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool
# import multiprocessing
# import joblib

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)
# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings('ignore', 'Metadata warning', UserWarning)


#%% Constants

DETECTION_INDEX_FILE_NAME = 'detectionIndex.json'


#%% Classes

class RepeatDetectionOptions:

    # inputFlename = r'D:\temp\tigers_20190308_all_output.csv'

    # Relevant for rendering HTML or filtering folder of images
    #
    # imageBase can also be a SAS URL, in which case some error-checking is
    # disabled.
    imageBase = ''
    outputBase = ''

    # Don't consider detections with confidence lower than this as suspicious
    confidenceMin = 0.8

    # Don't consider detections with confidence higher than this as suspicious
    confidenceMax = 1.0

    # What's the IOU threshold for considering two boxes the same?
    iouThreshold = 0.9

    # How many occurrences of a single location (as defined by the IOU threshold)
    # are required before we declare it suspicious?
    occurrenceThreshold = 15

    # Ignore "suspicious" detections larger than some size; these are often animals
    # taking up the whole image.  This is expressed as a fraction of the image size.
    maxSuspiciousDetectionSize = 0.2

    # A list of classes we don't want to treat as suspicious. Each element is an int.
    excludeClasses = []  # [annotation_constants.detector_bbox_category_name_to_id['person']]

    nWorkers = 10  # joblib.cpu_count()

    viz_target_width = 800

    # Load detections from a filter file rather than finding them from the detector output

    # .json file containing detections, should be called detectionIndex.json in the filtering_* folder 
    # produced in the first pass
    filterFileToLoad = ''

    # (optional) List of filenames remaining after deletion of identified 
    # repeated detections that are actually animals.  This should be a flat
    # text file, one relative filename per line.  See enumerate_images().
    filteredFileListToLoad = None

    # Turn on/off optional outputs
    bRenderHtml = False
    bWriteFilteringFolder = True

    debugMaxDir = -1
    debugMaxRenderDir = -1
    debugMaxRenderDetection = -1
    debugMaxRenderInstance = -1
    bParallelizeComparisons = True
    bParallelizeRendering = True
    
    # Determines whether bounding-box rendering errors (typically network errors) should
    # be treated as failures    
    bFailOnRenderError = False
    
    bPrintMissingImageWarnings = True
    missingImageWarningType = 'once'  # 'all'

    # Box rendering options
    lineThickness = 10
    boxExpansion = 2
    
    # State variables
    pbar = None

    # Replace filename tokens after reading, useful when the directory structure
    # has changed relative to the structure the detector saw
    filenameReplacements = {}

    # How many folders up from the leaf nodes should we be going to aggregate images?
    nDirLevelsFromLeaf = 0


class RepeatDetectionResults:
    """
    The results of an entire repeat detection analysis
    """

    # The data table (Pandas DataFrame), as loaded from the input json file via 
    # load_api_results()
    detectionResults = None

    # The other fields in the input json file, loaded via load_api_results()
    otherFields = None

    # The data table after modification
    detectionResultsFiltered = None

    # dict mapping folder names to whole rows from the data table
    rowsByDirectory = None

    # dict mapping filenames to rows in the master table
    filenameToRow = None

    # An array of length nDirs, where each element is a list of DetectionLocation 
    # objects for that directory that have been flagged as suspicious
    suspiciousDetections = None

    masterHtmlFile = None

    filterFile = None


class IndexedDetection:

    def __init__(self, iDetection=-1, filename='', bbox=[], confidence=-1, category='unknown'):
        """
        A single detection event on a single image

        Args:
            iDetection: order in API output file
            filename: path to the image of this detection
            bbox: [x_min, y_min, width_of_box, height_of_box]
        """
        self.iDetection = iDetection
        self.filename = filename
        self.bbox = bbox
        self.confidence = confidence
        self.category = category

    def __repr__(self):
        s = ct_utils.pretty_print_object(self, False)
        return s


class DetectionLocation:
    """
    A unique-ish detection location, meaningful in the context of one
    directory
    """

    def __init__(self, instance, detection, relativeDir):
        self.instances = [instance]  # list of IndexedDetections
        self.bbox = detection['bbox']
        self.relativeDir = relativeDir
        self.sampleImageRelativeFileName = ''

    def __repr__(self):
        s = ct_utils.pretty_print_object(self, False)
        return s
    
    def to_api_detection(self):
        """
        Converts to a 'detection' dictionary, making the semi-arbitrary assumption that
        the first instance is representative of confidence.
        """
        detection = {'conf':self.instances[0].confidence,'bbox':self.bbox,'category':self.instances[0].category}
        return detection


##%% Helper functions

def enumerate_images(dirName,outputFileName=None):
    """
    Non-recursively enumerates all image files in *dirName* to the text file 
    *outputFileName*, as relative paths.  This is used to produce a file list
    after removing true positives from the image directory.
    
    Not used directly in this module, but provides a consistent way to enumerate
    files in the format expected by this module.
    """
    imageList = path_utils.find_images(dirName)
    imageList = [os.path.basename(fn) for fn in imageList]
    
    if outputFileName is not None:
        with open(outputFileName,'w') as f:
            for s in imageList:
                f.write(s + '\n')
            
    return imageList
    

def render_bounding_box(detection, inputFileName, outputFileName, lineWidth=5, expansion=0):
    
    im = open_image(inputFileName)
    d = detection.to_api_detection()
    render_detection_bounding_boxes([d],im,thickness=lineWidth,expansion=expansion,confidence_threshold=-10)
    im.save(outputFileName)


##%% Look for matches (one directory) (function)

def find_matches_in_directory(dirName, options, rowsByDirectory):
        
    if options.pbar is not None:
        options.pbar.update()

    # List of DetectionLocations
    candidateDetections = []

    rows = rowsByDirectory[dirName]

    # iDirectoryRow = 0; row = rows.iloc[iDirectoryRow]
    for iDirectoryRow, row in rows.iterrows():

        filename = row['file']
        if not ct_utils.is_image_file(filename):
            continue

        # Don't bother checking images with no detections above threshold
        maxP = float(row['max_detection_conf'])
        if maxP < options.confidenceMin:
            continue

        # Array of dicts, where each element is
        # {
        #   'category': '1',  # str value, category ID
        #   'conf': 0.926,  # confidence of this detections
        #   'bbox': [x_min, y_min, width_of_box, height_of_box]  # (x_min, y_min) is upper-left,
        #                                                           all in relative coordinates and length
        # }
        detections = row['detections']
        if isinstance(detections,float):
            assert isinstance(row['failure'],str)
            print('Skipping failed image {} ({})'.format(filename,row['failure']))
            continue
        
        assert len(detections) > 0

        # For each detection in this image
        for iDetection, detection in enumerate(detections):
            
            assert 'category' in detection and 'conf' in detection and 'bbox' in detection

            confidence = detection['conf']
            
            # This is no longer strictly true; I sometimes run RDE in stages, so
            # some probabilities have already been made negative
            # assert confidence >= 0.0 and confidence <= 1.0
            assert confidence >= -1.0 and confidence <= 1.0
            
            if confidence < options.confidenceMin:
                continue
            if confidence > options.confidenceMax:
                continue

            # Optionally exclude some classes from consideration as suspicious
            if len(options.excludeClasses) > 0:
                iClass = int(detection['category'])
                if iClass in options.excludeClasses:
                    continue

            bbox = detection['bbox']
            confidence = detection['conf']
            
            # Is this detection too big to be suspicious?
            w, h = bbox[2], bbox[3]
            
            if (w == 0 or h == 0):
                # print('Illegal zero-size bounding box on image {}'.format(filename))
                continue
            
            area = h * w

            # These are relative coordinates
            assert area >= 0.0 and area <= 1.0, 'Illegal bounding box area {}'.format(area)

            if area > options.maxSuspiciousDetectionSize:
                # print('Ignoring very large detection with area {}'.format(area))
                continue

            category = detection['category']
            
            instance = IndexedDetection(iDetection=iDetection,
                                        filename=row['file'], bbox=bbox, 
                                        confidence=confidence, category=category)

            bFoundSimilarDetection = False

            # For each detection in our candidate list
            for iCandidate, candidate in enumerate(candidateDetections):

                # Is this a match?                    
                try:
                    iou = ct_utils.get_iou(bbox, candidate.bbox)
                except Exception as e:
                    print('Warning: IOU computation error on boxes ({},{},{},{}),({},{},{},{}): {}'.format(
                        bbox[0],bbox[1],bbox[2],bbox[3],
                        candidate.bbox[0],candidate.bbox[1],candidate.bbox[2],candidate.bbox[3], str(e)))
                    continue

                if iou >= options.iouThreshold:
                    
                    bFoundSimilarDetection = True

                    # If so, add this example to the list for this detection
                    candidate.instances.append(instance)

                    # We *don't* break here; we allow this instance to possibly
                    # match multiple candidates.  There isn't an obvious right or
                    # wrong here.

            # ...for each detection on our candidate list

            # If we found no matches, add this to the candidate list
            if not bFoundSimilarDetection:
                candidate = DetectionLocation(instance, detection, dirName)
                candidateDetections.append(candidate)

        # ...for each detection

    # ...for each row

    return candidateDetections

# ...def find_matches_in_directory(dirName)

    
##%% Render problematic locations to html (function)

def render_images_for_directory(iDir, directoryHtmlFiles, suspiciousDetections, options):
    
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
    print('Processing directory {} of {} ({})'.format(iDir, nDirs, timeStr))

    dirBaseDir = os.path.join(options.outputBase, dirName)
    os.makedirs(dirBaseDir, exist_ok=True)

    directoryDetectionHtmlFiles = []
    directoryDetectionImageInfo = []

    # For each problematic detection in this directory
    #
    # iDetection = 0; detection = suspiciousDetectionsThisDir[iDetection];
    nDetections = len(suspiciousDetectionsThisDir)
    bPrintedMissingImageWarning = False

    # iDetection = 0; detection = suspiciousDetectionsThisDir[0]
    for iDetection, detection in enumerate(suspiciousDetectionsThisDir):

        if options.debugMaxRenderDetection > 0 and iDetection > options.debugMaxRenderDetection:
            break

        nInstances = len(detection.instances)
        print('Processing detection {} of {} ({} instances)'.format(
            iDetection, nDetections, nInstances))
        detectionName = 'detection{:0>4d}'.format(iDetection)
        detectionBaseDir = os.path.join(dirBaseDir, detectionName)
        os.makedirs(detectionBaseDir, exist_ok=True)

        # _ = pretty_print_object(detection)
        assert (nInstances >= options.occurrenceThreshold)

        imageInfo = []

        # Render images

        # iInstance = 0; instance = detection.instances[iInstance]
        for iInstance, instance in enumerate(detection.instances):

            if options.debugMaxRenderInstance >= 0 and iInstance >= options.debugMaxRenderInstance:
                break

            imageRelativeFilename = 'image{:0>4d}.jpg'.format(iInstance)
            imageOutputFilename = os.path.join(detectionBaseDir,
                                               imageRelativeFilename)
            thisImageInfo = {}
            thisImageInfo['filename'] = imageRelativeFilename
            confidence = instance.confidence
            confidenceStr = '{:.2f}'.format(confidence)
            t = confidenceStr + ' (' + instance.filename + ')'
            thisImageInfo['title'] = t
            imageInfo.append(thisImageInfo)

            if not is_sas_url(options.imageBase):
                inputFileName = os.path.join(options.imageBase, instance.filename)
                if not os.path.isfile(inputFileName):
                    if options.bPrintMissingImageWarnings:
                        if (options.missingImageWarningType == 'all') or (not bPrintedMissingImageWarning):
                            print('Warning: could not find file {}'.format(inputFileName))
                            bPrintedMissingImageWarning = True
                    continue
            else:
                inputFileName = relative_sas_url(options.imageBase, instance.filename)
            render_bounding_box(detection, inputFileName, imageOutputFilename, lineWidth=options.lineThickness, 
                                expansion=options.boxExpansion)

        # ...for each instance

        # Write html for this detection
        detectionHtmlFile = os.path.join(detectionBaseDir, 'index.html')

        htmlOptions = write_html_image_list.write_html_image_list()
        htmlOptions['defaultImageStyle'] = 'max-width:650px;'
        write_html_image_list.write_html_image_list(detectionHtmlFile, imageInfo, htmlOptions)

        thisDirectoryImageInfo = {}
        directoryDetectionHtmlFiles.append(detectionHtmlFile)

        # Use the first image from this detection (arbitrary) as the canonical example
        # that we'll render for the directory-level page.
        thisDirectoryImageInfo['filename'] = os.path.join(detectionName, imageInfo[0]['filename'])
        detectionHtmlFileRelative = os.path.relpath(detectionHtmlFile, dirBaseDir)
        title = '<a href="{}">{}</a>'.format(detectionHtmlFileRelative, detectionName)
        thisDirectoryImageInfo['title'] = title
        directoryDetectionImageInfo.append(thisDirectoryImageInfo)

    # ...for each detection

    # Write the html file for this directory
    directoryHtmlFile = os.path.join(dirBaseDir, 'index.html')

    htmlOptions = write_html_image_list.write_html_image_list()
    htmlOptions['defaultImageStyle'] = 'max-width:650px;'
    write_html_image_list.write_html_image_list(directoryHtmlFile,
                                                directoryDetectionImageInfo,
                                                htmlOptions)

    return directoryHtmlFile

# ...def render_images_for_directory(iDir)


##%% Update the detection table based on suspicious results, write .csv output

def update_detection_table(RepeatDetectionResults, options, outputFilename=None):
    
    detectionResults = RepeatDetectionResults.detectionResults

    # An array of length nDirs, where each element is a list of DetectionLocation 
    # objects for that directory that have been flagged as suspicious
    suspiciousDetectionsByDirectory = RepeatDetectionResults.suspiciousDetections

    nBboxChanges = 0

    print('Updating output table')

    # For each directory
    for iDir, directoryEvents in enumerate(suspiciousDetectionsByDirectory):

        # For each suspicious detection group in this directory
        for iDetectionEvent, detectionEvent in enumerate(directoryEvents):

            locationBbox = detectionEvent.bbox

            # For each instance of this suspicious detection
            for iInstance, instance in enumerate(detectionEvent.instances):

                instanceBbox = instance.bbox

                # This should match the bbox for the detection event
                iou = ct_utils.get_iou(instanceBbox, locationBbox)
                
                # The bbox for this instance should be almost the same as the bbox
                # for this detection group, where "almost" is defined by the IOU
                # threshold.
                assert iou >= options.iouThreshold
                # if iou < options.iouThreshold:
                #    print('IOU warning: {},{}'.format(iou,options.iouThreshold))

                assert instance.filename in RepeatDetectionResults.filenameToRow
                iRow = RepeatDetectionResults.filenameToRow[instance.filename]
                row = detectionResults.iloc[iRow]
                rowDetections = row['detections']
                detectionToModify = rowDetections[instance.iDetection]

                # Make sure the bounding box matches
                assert (instanceBbox[0:3] == detectionToModify['bbox'][0:3])

                # Make the probability negative, if it hasn't been switched by
                # another bounding box
                if detectionToModify['conf'] >= 0:
                    detectionToModify['conf'] = -1 * detectionToModify['conf']
                    nBboxChanges += 1

            # ...for each instance

        # ...for each detection

    # ...for each directory       

    # Update maximum probabilities

    # For each row...
    nProbChanges = 0
    nProbChangesToNegative = 0
    nProbChangesAcrossThreshold = 0

    for iRow, row in detectionResults.iterrows():

        detections = row['detections']
        if isinstance(detections,float):
            assert isinstance(row['failure'],str)
            continue
        
        if len(detections) == 0:
            continue

        maxPOriginal = float(row['max_detection_conf'])
        
        # No longer strictly true; sometimes I run RDE on RDE output
        # assert maxPOriginal >= 0
        assert maxPOriginal >= -1.0

        maxP = None
        nNegative = 0

        for iDetection, detection in enumerate(detections):
            
            p = detection['conf']

            if p < 0:
                nNegative += 1

            if (maxP is None) or (p > maxP):
                maxP = p
        
        if abs(maxP - maxPOriginal) > 1e-3:

            # We should only be making detections *less* likely
            assert maxP < maxPOriginal
            # row['max_confidence'] = str(maxP)
            detectionResults.at[iRow, 'max_detection_conf'] = maxP

            nProbChanges += 1

            if (maxP < 0) and (maxPOriginal >= 0):
                nProbChangesToNegative += 1

            if (maxPOriginal >= options.confidenceMin) and (maxP < options.confidenceMin):
                nProbChangesAcrossThreshold += 1

            # Negative probabilities should be the only reason maxP changed, so
            # we should have found at least one negative value
            assert nNegative > 0

        # ...if there was a meaningful change to the max probability for this row

    # ...for each row

    # If we're also writing output...
    if outputFilename is not None and len(outputFilename) > 0:
        write_api_results(detectionResults, RepeatDetectionResults.otherFields, outputFilename)

    print(
        'Finished updating detection table\nChanged {} detections that impacted {} maxPs ({} to negative) ({} across confidence threshold)'.format(
            nBboxChanges, nProbChanges, nProbChangesToNegative, nProbChangesAcrossThreshold))

    return detectionResults

# ...def update_detection_table(RepeatDetectionResults,options)


##%% Main function

def find_repeat_detections(inputFilename, outputFilename=None, options=None):
    
    ##%% Input handling

    if options is None:
        options = RepeatDetectionOptions()

    toReturn = RepeatDetectionResults()

    
    # Check early to avoid problems with the output folder
    
    if options.bWriteFilteringFolder or options.bRenderHtml:
        assert options.outputBase is not None and len(options.outputBase) > 0
        os.makedirs(options.outputBase,exist_ok=True)


    # Load file

    detectionResults, otherFields = load_api_results(inputFilename, normalize_paths=True,
                                         filename_replacements=options.filenameReplacements)
    toReturn.detectionResults = detectionResults
    toReturn.otherFields = otherFields

    # detectionResults[detectionResults['failure'].notna()]
        
    # Before doing any real work, make sure we can *probably* access images
    # This is just a cursory check on the first image, but it heads off most 
    # problems related to incorrect mount points, etc.  Better to do this before
    # spending 20 minutes finding repeat detections.  
    
    if options.bWriteFilteringFolder or options.bRenderHtml:
        if not is_sas_url(options.imageBase):
            row = detectionResults.iloc[0]
            relativePath = row['file']
            for s in options.filenameReplacements.keys():
                relativePath = relativePath.replace(s,options.filenameReplacements[s])
            absolutePath = os.path.join(options.imageBase,relativePath)
            assert os.path.isfile(absolutePath), 'Could not find file {}'.format(absolutePath)
        
        
    ##%% Separate files into directories

    # This will be a map from a directory name to smaller data frames
    rowsByDirectory = {}

    # This is a mapping back into the rows of the original table
    filenameToRow = {}

    # TODO: in the case where we're loading an existing set of FPs after manual filtering,
    # we should load these data frames too, rather than re-building them from the input.

    print('Separating files into directories...')

    # iRow = 0; row = detectionResults.iloc[0]
    for iRow, row in detectionResults.iterrows():
        relativePath = row['file']
        dirName = os.path.dirname(relativePath)
        
        if len(dirName) == 0:
            assert options.nDirLevelsFromLeaf == 0, 'Can''t use the dirLevelsFromLeaf option with flat filenames'
        else:
            if options.nDirLevelsFromLeaf > 0:
                iLevel = 0
                while (iLevel < options.nDirLevelsFromLeaf):
                    iLevel += 1
                    dirName = os.path.dirname(dirName)
            assert len(dirName) > 0

        if not dirName in rowsByDirectory:
            # Create a new DataFrame with just this row
            # rowsByDirectory[dirName] = pd.DataFrame(row)
            rowsByDirectory[dirName] = []

        rowsByDirectory[dirName].append(row)

        assert relativePath not in filenameToRow
        filenameToRow[relativePath] = iRow

    # Convert lists of rows to proper DataFrames
    dirs = list(rowsByDirectory.keys())
    for d in dirs:
        rowsByDirectory[d] = pd.DataFrame(rowsByDirectory[d])

    toReturn.rowsByDirectory = rowsByDirectory
    toReturn.filenameToRow = filenameToRow

    print('Finished separating {} files into {} directories'.format(len(detectionResults),
                                                                    len(rowsByDirectory)))


    ##% Look for matches (or load them from file)

    dirsToSearch = list(rowsByDirectory.keys())
    if options.debugMaxDir > 0:
        dirsToSearch = dirsToSearch[0:options.debugMaxDir]

    # length-nDirs list of lists of DetectionLocation objects
    suspiciousDetections = [None] * len(dirsToSearch)

    # Are we actually looking for matches, or just loading from a file?
    if len(options.filterFileToLoad) == 0:

        # We're actually looking for matches...
        print('Finding similar detections...')

        allCandidateDetections = [None] * len(dirsToSearch)

        if not options.bParallelizeComparisons:

            options.pbar = None
            # iDir = 0; dirName = dirsToSearch[iDir]
            for iDir, dirName in enumerate(tqdm(dirsToSearch)):
                allCandidateDetections[iDir] = find_matches_in_directory(dirName, options, rowsByDirectory)

        else:

            options.pbar = tqdm(total=len(dirsToSearch))
            allCandidateDetections = Parallel(n_jobs=options.nWorkers, prefer='threads')(
                delayed(find_matches_in_directory)(dirName, options, rowsByDirectory) for dirName in tqdm(dirsToSearch))

        print('\nFinished looking for similar bounding boxes')

        ##%% Find suspicious locations based on match results

        print('Filtering out repeat detections...')

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

            for iLocation, candidateLocation in enumerate(candidateDetectionsThisDir):

                # occurrenceList is a list of file/detection pairs
                nOccurrences = len(candidateLocation.instances)

                if nOccurrences < options.occurrenceThreshold:
                    continue

                nImagesWithSuspiciousDetections += nOccurrences
                nSuspiciousDetections += 1

                suspiciousDetectionsThisDir.append(candidateLocation)
                # Find the images corresponding to this bounding box, render boxes

            suspiciousDetections[iDir] = suspiciousDetectionsThisDir

        print(
            'Finished searching for repeat detections\nFound {} unique detections on {} images that are suspicious'.format(
                nSuspiciousDetections, nImagesWithSuspiciousDetections))

    else:

        print('Bypassing detection-finding, loading from {}'.format(options.filterFileToLoad))

        # Load the filtering file
        detectionIndexFileName = options.filterFileToLoad
        sIn = open(detectionIndexFileName, 'r').read()
        suspiciousDetections = jsonpickle.decode(sIn)
        filteringBaseDir = os.path.dirname(options.filterFileToLoad)
        assert len(suspiciousDetections) == len(dirsToSearch)

        nDetectionsRemoved = 0
        nDetectionsLoaded = 0

        # We're skipping detection-finding, but to see which images are actually legit false
        # positives, we may be looking for physical files or loading from a text file.        
        fileList = None
        if options.filteredFileListToLoad is not None:
            with open(options.filteredFileListToLoad) as f:
                fileList = f.readlines()
                fileList = [x.strip() for x in fileList]
            nSuspiciousDetections = sum([len(x) for x in suspiciousDetections])
            print('Loaded false positive list from file, will remove {} of {} suspicious detections'.format(
                len(fileList), nSuspiciousDetections))

        # For each directory
        # iDir = 0; detections = suspiciousDetections[0]
        #
        # suspiciousDetections is an array of DetectionLocation objects,
        # one per directory.            
        for iDir, detections in enumerate(suspiciousDetections):

            bValidDetection = [True] * len(detections)
            nDetectionsLoaded += len(detections)

            # For each detection that was present before filtering
            # iDetection = 0; detection = detections[iDetection]
            for iDetection, detection in enumerate(detections):

                # Are we checking the directory to see whether detections were actually false
                # positives, or reading from a list?
                if fileList is None:
                    
                    # Is the image still there?                
                    imageFullPath = os.path.join(filteringBaseDir, detection.sampleImageRelativeFileName)

                    # If not, remove this from the list of suspicious detections
                    if not os.path.isfile(imageFullPath):
                        nDetectionsRemoved += 1
                        bValidDetection[iDetection] = False

                else:
                    
                    if detection.sampleImageRelativeFileName not in fileList:
                        nDetectionsRemoved += 1
                        bValidDetection[iDetection] = False

            # ...for each detection

            nRemovedThisDir = len(bValidDetection) - sum(bValidDetection)
            if nRemovedThisDir > 0:
                print('Removed {} of {} detections from directory {}'.format(nRemovedThisDir,
                                                                             len(detections), iDir))

            detectionsFiltered = list(compress(detections, bValidDetection))
            suspiciousDetections[iDir] = detectionsFiltered

        # ...for each directory

        print('Removed {} of {} total detections via manual filtering'.format(nDetectionsRemoved, nDetectionsLoaded))

    # ...if we are/aren't finding detections (vs. loading from file)

    toReturn.suspiciousDetections = suspiciousDetections

    if options.bRenderHtml:

        # Render problematic locations with html (loop)

        print('Rendering html')

        nDirs = len(dirsToSearch)
        directoryHtmlFiles = [None] * nDirs

        if options.bParallelizeRendering:

            # options.pbar = tqdm(total=nDirs)
            options.pbar = None

            directoryHtmlFiles = Parallel(n_jobs=options.nWorkers, prefer='threads')(delayed(
                render_images_for_directory)(iDir, directoryHtmlFiles, suspiciousDetections, options) for iDir in
                                                                                     tqdm(range(nDirs)))

        else:

            options.pbar = None

            # For each directory
            # iDir = 51
            for iDir in range(nDirs):
                # Add this directory to the master list of html files
                directoryHtmlFiles[iDir] = render_images_for_directory(iDir, directoryHtmlFiles, suspiciousDetections,
                                                                       options)

            # ...for each directory

        # Write master html file

        masterHtmlFile = os.path.join(options.outputBase, 'index.html')
        os.makedirs(options.outputBase, exist_ok=True)
        toReturn.masterHtmlFile = masterHtmlFile

        with open(masterHtmlFile, 'w') as fHtml:

            fHtml.write('<html><body>\n')
            fHtml.write('<h2><b>Repeat detections by directory</b></h2></br>\n')

            for iDir, dirHtmlFile in enumerate(directoryHtmlFiles):

                if dirHtmlFile is None:
                    continue

                relPath = os.path.relpath(dirHtmlFile, options.outputBase)
                dirName = dirsToSearch[iDir]

                # Remove unicode characters before formatting
                relPath = relPath.encode('ascii', 'ignore').decode('ascii')
                dirName = dirName.encode('ascii', 'ignore').decode('ascii')

                fHtml.write('<a href={}>{}</a><br/>\n'.format(relPath, dirName))

            fHtml.write('</body></html>\n')

    # ...if we're rendering html

    toReturn.allRowsFiltered = update_detection_table(toReturn, options, outputFilename)
    
    # Create filtering directory
    if options.bWriteFilteringFolder:

        print('Creating filtering folder...')

        dateString = datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
        filteringDir = os.path.join(options.outputBase, 'filtering_' + dateString)
        os.makedirs(filteringDir, exist_ok=True)

        # iDir = 0; suspiciousDetectionsThisDir = suspiciousDetections[iDir]
        for iDir, suspiciousDetectionsThisDir in enumerate(tqdm(suspiciousDetections)):

            # suspiciousDetectionsThisDir is a list of DetectionLocation objects
            # iDetection = 0; detection = suspiciousDetectionsThisDir[0]
            for iDetection, detection in enumerate(suspiciousDetectionsThisDir):
                
                instance = detection.instances[0]
                relativePath = instance.filename
                outputRelativePath = 'dir{:0>4d}_det{:0>4d}_n{:0>4d}.jpg'.format(iDir, iDetection, len(detection.instances))
                outputFullPath = os.path.join(filteringDir, outputRelativePath)
                
                if is_sas_url(options.imageBase):
                    inputFullPath = relative_sas_url(options.imageBase, relativePath)
                else:
                    inputFullPath = os.path.join(options.imageBase, relativePath)
                    assert (os.path.isfile(inputFullPath)), 'Not a file: {}'.format(inputFullPath)
                    
                try:
                    render_bounding_box(detection, inputFullPath, outputFullPath,
                                        lineWidth=options.lineThickness, expansion=options.boxExpansion)
                except Exception as e:
                    print('Warning: error rendering bounding box from {} to {}: {}'.format(
                        inputFullPath,outputFullPath,e))                    
                    if options.bFailOnRenderError:
                        raise
                detection.sampleImageRelativeFileName = outputRelativePath

        # Write out the detection index
        detectionIndexFileName = os.path.join(filteringDir, DETECTION_INDEX_FILE_NAME)
        jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)
        s = jsonpickle.encode(suspiciousDetections)
        with open(detectionIndexFileName, 'w') as f:
            f.write(s)
        toReturn.filterFile = detectionIndexFileName

        print('Done')

    # ...if we're writing filtering info

    return toReturn

# ...find_repeat_detections()


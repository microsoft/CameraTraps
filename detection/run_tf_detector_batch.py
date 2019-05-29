######
#
# run_tf_detector_batch.py
#
# Runs a TensorFlow detection model on images, writing the results to a file
# in the same format produced by our batch API:
#
# https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing
#
# This enables the results to be used in our post-processing pipeline; see
# postprocess_batch_results.py .
#
# See the "test driver" cell for example invocation.
#
######

#%% Constants, imports, environment

import time
import glob
import sys
import argparse
import os
import json
import pickle
import inspect
import tempfile
import warnings
from itertools import compress

import tensorflow as tf
import numpy as np
import humanfriendly
import PIL
from tqdm import tqdm
import pandas as pd

from detection.detection_eval.load_api_results import write_api_results

DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Suppress excessive tensorflow output
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Write results to a temporary file every N images, in case something crashes;
# set to <= 0 to disable this feature.
DEFAULT_CHECKPOINT_N_IMAGES = -1

CHECKPOINT_SUBDIR = 'detector_batch'

# ignoring all "PIL cannot read EXIF metainfo for the images" warnings
warnings.filterwarnings('ignore', '(Possibly )?corrupt EXIF data', UserWarning)
# Metadata Warning, tag 256 had too many entries: 42, expected 1
warnings.filterwarnings('ignore', 'Metadata warning', UserWarning)


#%% Classes

class BatchDetectionOptions:
    
    detectorFile = None
    
    # Can be a singe image file, a text file containing a list of images, or a 
    # directory
    imageFile = None
    outputFile = None
    threshold = DEFAULT_CONFIDENCE_THRESHOLD
    recursive = False
    forceCpu = False
    checkpointFrequency = DEFAULT_CHECKPOINT_N_IMAGES
    resumeFromCheckpoint = None
    

class CheckPointState:
    
    iImage = 0
    boxes = []
    scores = []
    classes = []
    bValidImage = None
    
    def __init__(self,nImages):
                
        self.bValidImage = [True] * nImages
        self.classes = [None] * nImages
        self.scores = [None] * nImages
        self.boxes = [None] * nImages
        
        
#%% Core detection functions

def load_model(checkpoint):
    """
    Load a detection model (i.e., create a graph) from a .pb file
    """

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    return detection_graph


def generate_detections(detection_graph,images,options):
    """
    boxes,scores,classes,images = generate_detections(detection_graph,images)

    Run an already-loaded detector network on a set of images.

    [images] should be a list of filenames.

    Boxes are returned in relative coordinates as (top, left, bottom, right); 
    x,y origin is the upper-left.
    
    [boxes] will be returned as a numpy array of size nImages x nDetections x 4.
    
    [scores] and [classes] will each be returned as a numpy array of size nImages x nDetections.    
    
    [images] will be returned as a list of files that were actually processed, possibly a subset
    of the input parameter [images].
    """

    if not isinstance(images,list):
        images = [images]
        
    nImages = len(images)
    firstImage = 0
    
    if options.resumeFromCheckpoint is not None:
        print('Loading state from checkpoint {}'.format(options.resumeFromCheckpoint))
        cpState = pickle.load(open(options.resumeFromCheckpoint,'rb'))
        
        # "Wind back the clock" by one image, just to avoid edge effects like crashing during
        # serialization.
        firstImage = max(cpState.iImage - 1,0)
        print('Resuming from image {}'.format(firstImage))
    else:
        cpState = CheckPointState(nImages)
        
    print('Running detector...')    
    startTime = time.time()
    firstImageCompleteTime = startTime
    
    with detection_graph.as_default():
        
        with tf.Session(graph=detection_graph) as sess:
            
            for iImage,image in tqdm(enumerate(images)): 
                
                # Skip images we've loaded from a checkpoint or otherwise preprocessed
                if iImage < firstImage:
                    continue
                
                assert isinstance(image,str)
                
                if not os.path.isfile(image):
                    print('Warning: can''t find file {}, skipping'.format(image))
                    cpState.bValidImage[iImage] = False
                    continue
                
                # Load the image as an nparray of size h,w,nChannels            
                imageNP = PIL.Image.open(image).convert("RGB"); imageNP = np.array(imageNP)
                # image = mpimg.imread(image)
                
                # This shouldn't be necessary when loading with PIL and converting to RGB
                nChannels = imageNP.shape[2]
                if nChannels > 3:
                    print('Warning: trimming channels from image')
                    imageNP = imageNP[:,:,0:3]
                
                imageNP_expanded = np.expand_dims(imageNP, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                box = detection_graph.get_tensor_by_name('detection_boxes:0')
                score = detection_graph.get_tensor_by_name('detection_scores:0')
                clss = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                
                # Run inference on this image
                (box, score, clss, num_detections) = sess.run(
                        [box, score, clss, num_detections],
                        feed_dict={image_tensor: imageNP_expanded})

                cpState.boxes[iImage] = box
                cpState.scores[iImage] = score
                cpState.classes[iImage] = clss
            
                if iImage == 0:
                    firstImageCompleteTime = time.time()
                
                if options.checkpointFrequency > 0:
                    
                    if (iImage % options.checkpointFrequency) == 0:
                    
                        tempDir = os.path.join(tempfile.gettempdir(),CHECKPOINT_SUBDIR)
                        os.makedirs(tempDir,exist_ok=True)
                        
                        # jsonpickle.set_encoder_options('json', sort_keys=False, indent=0)
                        # s = jsonpickle.encode(cp)
                        f = tempfile.NamedTemporaryFile(dir=tempDir,delete=False)
                        print('Checkpointing {} images to {}...'.format(iImage+1,f.name),end='')
                        cpState.iImage = iImage
                        pickle.dump(cpState,f)
                        f.close()
                        print('...done')
                                            
            # ...for each image                
    
        # ...with tf.Session

    # ...with detection_graph.as_default()
    
    images = list(compress(images, cpState.bValidImage))
    boxes = list(compress(cpState.boxes, cpState.bValidImage))
    scores = list(compress(cpState.scores, cpState.bValidImage))
    classes = list(compress(cpState.classes, cpState.bValidImage))
    
    nImages = len(images)
    
    elapsed = time.time() - startTime
    if nImages == 1:
        print("Finished running detector in {}".format(humanfriendly.format_timespan(elapsed)))
    else:
        firstImageElapsed = firstImageCompleteTime - startTime
        remainingImagesElapsed = elapsed - firstImageElapsed
        remainingImagesTimePerImage = remainingImagesElapsed/(nImages-1)
        
        print("Finished running detector on {} images in {} ({} for the first image, {} for each subsequent image)".format(len(images),
              humanfriendly.format_timespan(elapsed),
              humanfriendly.format_timespan(firstImageElapsed),
              humanfriendly.format_timespan(remainingImagesTimePerImage)))
    
    nDetections = -1
    
    # "boxes" has shape:
    #
    # 1,nDetections,4
    #
    # This implicitly banks on TF giving us back a fixed number of boxes, we'll assert on this
    # later to make sure this doesn't silently break in the future.
    
    assert(len(boxes)==nImages)    
    # iBox = 0; box = boxes[iBox]
    for iBox,box in enumerate(boxes):
        nDetectionsThisBox = box.shape[1]
        assert (nDetections == -1 or nDetectionsThisBox == nDetections), 'Detection count mismatch'
        nDetections = nDetectionsThisBox
        assert(box.shape[0] == 1)
    
    # "scores" is a length-nImages list of elements with size 1,nDetections
    assert(len(scores) == nImages)
    for iScore,score in enumerate(scores):
        assert score.shape[0] == 1
        assert score.shape[1] == nDetections
        
    # "classes" is a length-nImages list of elements with size 1,nDetections
    #
    # Still as floats, but really representing ints
    assert(len(classes) == nImages)
    for iClass,c in enumerate(classes):
        assert c.shape[0] == 1
        assert c.shape[1] == nDetections
            
    # Squeeze out the empty axis
    boxes = np.squeeze(np.array(boxes),axis=1)
    scores = np.squeeze(np.array(scores),axis=1)
    classes = np.squeeze(np.array(classes),axis=1).astype(int)
    
    # boxes is nImages x nDetections x 4
    assert(len(boxes.shape) == 3)
    assert(boxes.shape[0] == nImages)
    assert(boxes.shape[1] == nDetections)
    assert(boxes.shape[2] == 4)
    
    # scores and classes are both nImages x nDetections
    assert(len(scores.shape) == 2)
    assert(scores.shape[0] == nImages)
    assert(scores.shape[1] == nDetections)
    
    assert(len(classes.shape) == 2)
    assert(classes.shape[0] == nImages)
    assert(classes.shape[1] == nDetections)
    
    return boxes,scores,classes,images


#%% File helper functions

imageExtensions = ['.jpg','.jpeg','.gif','.png']
    
def isImageFile(s):
    """
    Check a file's extension against a hard-coded set of image file extensions    '
    """
    
    ext = os.path.splitext(s)[1]
    return ext.lower() in imageExtensions
    
    
def findImageStrings(strings):
    """
    Given a list of strings that are potentially image file names, look for strings
    that actually look like image file names (based on extension).
    """
    
    imageStrings = []
    bIsImage = [False] * len(strings)
    for iString,f in enumerate(strings):
        bIsImage[iString] = isImageFile(f) 
        if bIsImage[iString]:
            imageStrings.append(f)
        
    return imageStrings

    
def findImages(dirName,bRecursive=False):
    """
    Find all files in a directory that look like image file names
    """
    if bRecursive:
        strings = glob.glob(os.path.join(dirName,'**','*.*'), recursive=True)
    else:
        strings = glob.glob(os.path.join(dirName,'*.*'))
        
    imageStrings = findImageStrings(strings)
    
    return imageStrings


def optionsToImages(options):
    """
    Figure out what images the caller wants to process
    """
    
    # Can be a singe image file, a text file containing a list of images, or a directory
    if os.path.isdir(options.imageFile):
        
        imageFileNames = findImages(options.imageFile,options.recursive)
        imageFileNames.append('asdfasdfasd')
        
    else:
        
        assert os.path.isfile(options.imageFile)
        
        if isImageFile(options.imagefile):
            imageFileNames = [options.imageFile]
        else:
            with open(options.imageFile) as f:
                imageFileNames = f.readlines()
                imageFileNames = [x.strip() for x in imageFileNames] 

    return imageFileNames


def detectorOutputToApiOutput(imageFileNames,options,boxes,scores,classes):
    """
    Converts the output of the TFODAPI detector to the format used by our batch
    API, as a pandas table.    
    """    
    nRows = len(imageFileNames)
    
    boxesOut = [[]] * nRows
    confidences = [0] * nRows
    # iRow = 0
    for iRow in range(0,nRows):
        
        imageBoxes = boxes[iRow]
        imageScores = scores[iRow]
        imageClasses = classes[iRow]
        
        bScoreAboveThreshold = imageScores > options.threshold
        
        if not any(bScoreAboveThreshold):
            continue
        
        imageScores = list(compress(imageScores, bScoreAboveThreshold))
        imageBoxes = list(compress(imageBoxes, bScoreAboveThreshold))
        imageClasses = list(compress(imageClasses, bScoreAboveThreshold))
        
        confidences[iRow] = max(imageScores)
        
        nImageBoxes = len(imageScores)
        imageBoxesOut = []
        
        # Convert detections into 5-element lists, where the fifth element is a class    
        #
        # iBox = 0
        for iBox in range(0,nImageBoxes):
            box = list(imageBoxes[iBox])
            
            # convert from float32 to float
            box = [float(x) for x in box]
            box.append(int(imageClasses[iBox]))
            imageBoxesOut.append(box)
        
        boxesOut[iRow] = imageBoxesOut

    boxStrings = [json.dumps(x) for x in boxesOut]
    
    # Build the output table
    df = pd.DataFrame(np.column_stack([imageFileNames,confidences,boxStrings]),
                      columns=['image_path','max_confidence','detections'])
    return df


#%% Main function

def load_and_run_detector(options,detection_graph=None):
    
    imageFileNames = optionsToImages(options)
    
    if options.forceCpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print('Running detector on {} images'.format(len(imageFileNames)))    
    
    if len(imageFileNames) == 0:        
        print('Warning: no files available')
        return
    
    # Load and run detector on target images
    print('Loading model...')
    startTime = time.time()
    if detection_graph is None:
        detection_graph = load_model(options.detectorFile)
    elapsed = time.time() - startTime
    print("Loaded model in {}".format(humanfriendly.format_timespan(elapsed)))
    
    boxes,scores,classes,imageFileNames = generate_detections(detection_graph,imageFileNames,options)
    
    assert len(boxes) == len(imageFileNames)
    
    print('Writing output...')
    
    df = detectorOutputToApiOutput(imageFileNames,options,boxes,scores,classes)
    write_api_results(df,options.outputFile)
    
    return boxes,scores,classes,imageFileNames


#%% Interactive driver

if False:
             
    #%%
    
    options = BatchDetectionOptions()
    options.detectorFile = r'D:\temp\models\object_detection\megadetector\megadetector_v2.pb'
    options.imageFile = r'D:\temp\demo_images\uw_kachel'    
    options.outputFile = r'd:\temp\detector_out.csv'
    options.recursive = True
    options.checkpointFrequency = 1
    options.resumeFromCheckpoint = r'C:\Users\dan\AppData\Local\Temp\detector_batch\tmp77xdq9dp'
    
    print('Loading model...',end='')
    detection_graph = load_model(options.detectorFile)
    print('...done')
    
    boxes,scores,classes,imageFileNames = load_and_run_detector(options,detection_graph)
    
    #%%
    
    from api.batch_processing.postprocess_batch_results import PostProcessingOptions
    from api.batch_processing.postprocess_batch_results import process_batch_results

    ppoptions = PostProcessingOptions()
    ppoptions.image_base_dir = options.imageFile
    ppoptions.detector_output_file = options.outputFile
    ppoptions.output_dir = os.path.join(ppoptions.image_base_dir,'postprocessing')
    ppresults = process_batch_results(ppoptions)
    os.startfile(ppresults.output_html_file)
    
#%% Command-line driver
   
# Copy all fields from a Namespace (i.e., the output from parse_args) to an object.  
#
# Skips fields starting with _.  Does not check existence in the target object.
def args_to_object(args, obj):
    
    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v);

    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('detectorFile', type=str)
    parser.add_argument('imageFile', action='store', type=str, 
                        help='Can be a singe image file, a text file containing a list of images, or a directory')
    parser.add_argument('outputFile', type=str, 
                       help='Output results file')
    parser.add_argument('--threshold', action='store', type=float, 
                        default=DEFAULT_CONFIDENCE_THRESHOLD, 
                        help='Confidence threshold, don''t render boxes below this confidence')
    parser.add_argument('--recursive', action='store_true', 
                        help='Recurse into directories, only meaningful if using --imageDir')
    parser.add_argument('--forceCpu', action='store_true', 
                        help='Force CPU detection, even if a GPU is available')
    parser.add_argument('--checkpointFrequency', type=int, default=DEFAULT_CHECKPOINT_N_IMAGES,
                        'Checkpoint results to allow restoration from crash points later')
    parser.add_argument('--resumeFromCheckpoint', type=str, default=None,
                        'Initiate inference from the specified checkpoint')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()    
    
    options = BatchDetectionOptions()
    args_to_object(args,options)
        
    load_and_run_detector(options)
    

if __name__ == '__main__':
    
    main()

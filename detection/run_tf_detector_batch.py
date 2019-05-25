######
#
# run_tf_detector_batch.py
#
# Runs a TensorFlow detection model on images, writing the results to a .csv file
# in the same format produced by our batch API:
#
# https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing
#
# See the "test driver" cell for example invocation.
#
######

#%% Constants, imports, environment

import tensorflow as tf
import numpy as np
import humanfriendly
import time
import glob
import sys
import argparse
import PIL
import os
from tqdm import tqdm
import inspect

DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Suppress excessive tensorflow output
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#%% Classes

class BatchDetectionOptions:
    
    detectorFile = None
    
    # Can be a singe image file, a text file containing a list of images, or a 
    # directory
    imageFile = None
    threshold = DEFAULT_CONFIDENCE_THRESHOLD
    recursive = False
    forceCpu = False
    
    
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


def generate_detections(detection_graph,images):
    """
    boxes,scores,classes,= generate_detections(detection_graph,images)

    Run an already-loaded detector network on a set of images.

    [images] should be a list of filenames.

    Boxes are returned in relative coordinates as (top, left, bottom, right); 
    x,y origin is the upper-left.
    
    [boxes] will be returned as a numpy array of size nImages x nDetections x 4.
    
    [scores] and [classes] will each be returned as a numpy array of size nImages x nDetections.    
    """

    if not isinstance(images,list):
        images = [images]
        
    boxes = []
    scores = []
    classes = []
    
    nImages = len(images)

    print('Running detector...')    
    startTime = time.time()
    firstImageCompleteTime = None
    
    with detection_graph.as_default():
        
        with tf.Session(graph=detection_graph) as sess:
            
            for iImage,image in tqdm(enumerate(images)): 
                
                assert isinstance(image,str)
                
                if not os.path.isfile(image):
                    print('Warning: can''t find file {}, skipping'.format(image))
                    continue
                
                # Load the image as an nparray of size h,w,nChannels            
                image = PIL.Image.open(image).convert("RGB"); image = np.array(image)
                # image = mpimg.imread(image)
                
                # This shouldn't be necessary when loading with PIL and converting to RGB
                nChannels = image.shape[2]
                if nChannels > 3:
                    print('Warning: trimming channels from image')
                    imageNP = image[:,:,0:3]
                
                imageNP_expanded = np.expand_dims(imageNP, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                box = detection_graph.get_tensor_by_name('detection_boxes:0')
                score = detection_graph.get_tensor_by_name('detection_scores:0')
                clss = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                
                # Actual detection
                (box, score, clss, num_detections) = sess.run(
                        [box, score, clss, num_detections],
                        feed_dict={image_tensor: imageNP_expanded})

                boxes.append(box)
                scores.append(score)
                classes.append(clss)
            
                if iImage == 0:
                    firstImageCompleteTime = time.time()
                    
            # ...for each image                
    
        # ...with tf.Session

    # ...with detection_graph.as_default()
    
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
    
    nBoxes = len(boxes)
    
    # Currently "boxes" is a list of length nImages, where each element is shaped as
    #
    # 1,nDetections,4
    #
    # This implicitly banks on TF giving us back a fixed number of boxes, let's assert on this
    # to make sure this doesn't silently break in the future.
    nDetections = -1
    # iBox = 0; box = boxes[iBox]
    for iBox,box in enumerate(boxes):
        nDetectionsThisBox = box.shape[1]
        assert (nDetections == -1 or nDetectionsThisBox == nDetections), 'Detection count mismatch'
        nDetections = nDetectionsThisBox
        assert(box.shape[0] == 1)
    
    # "scores" is a length-nImages list of elements with size 1,nDetections
    assert(len(scores) == nImages)
    for(iScore,score) in enumerate(scores):
        assert score.shape[0] == 1
        assert score.shape[1] == nDetections
        
    # "classes" is a length-nImages list of elements with size 1,nDetections
    #
    # Still as floats, but really representing ints
    assert(len(classes) == nBoxes)
    for(iClass,c) in enumerate(classes):
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
    
    return boxes,scores,classes


#%% File helper functions

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
            imageStrings.append(f)
        
    return imageStrings

    
def findImages(dirName,bRecursive=False):
    '''
    Find all files in a directory that look like image file names
    '''
    if bRecursive:
        strings = glob.glob(os.path.join(dirName,'**','*.*'), recursive=True)
    else:
        strings = glob.glob(os.path.join(dirName,'*.*'))
        
    imageStrings = findImageStrings(strings)
    
    return imageStrings


def optionsToImages(options):
    '''
    Figure out what images the caller wants to process
    '''
    
    # Can be a singe image file, a text file containing a list of images, or a directory
    if os.path.isdir(options.imageFile):
        
        imageFileNames = findImages(options.imageFile,options.recursive)
        
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
    '''
    Converts the output of the TFODAPI detector to the format used by our batch
    API, as a pandas table.
    '''
    pass
    

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
    
    boxes,scores,classes = generate_detections(detection_graph,imageFileNames)
    
    assert len(boxes) == len(imageFileNames)
    
    print('Writing output...')
    
    # Todo
    
    return detection_graph


#%% Interactive driver

if False:
    
    #%%
    
    detection_graph = None
    
    #%%
    
    modelFile = r'D:\temp\models\object_detection\megadetector\megadetector_v2.pb'
    imageDir = r'D:\temp\demo_images\b2'    
    imageFileNames = [fn for fn in glob.glob(os.path.join(imageDir,'*.jpg'))
         if (not 'detections' in fn)]
    imageFileNames = [r"D:\temp\test\1\NE2881-9_RCNX0195_xparent.png"]
    
    detection_graph = load_and_run_detector(modelFile,imageFileNames,
                                            confidenceThreshold=DEFAULT_CONFIDENCE_THRESHOLD,
                                            detection_graph=detection_graph)
    

#%% Command-line driver
   
# Copy all fields from a Namespace (i.e., the output from parse_args) to an object.  
#
# Skips fields starting with _.  Does not check existence in the target object.
def args_to_object(args, obj):
    
    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v);

    
def main():
    
    # python run_tf_detector.py "D:\temp\models\object_detection\megadetector\megadetector_v2.pb" --imageFile "D:\temp\demo_images\test\S1_J08_R1_PICT0120.JPG"
    # python run_tf_detector.py "D:\temp\models\object_detection\megadetector\megadetector_v2.pb" --imageDir "d:\temp\demo_images\test"
    # python run_tf_detector.py "d:\temp\models\object_detection\megadetector\megadetector_v3.pb" --imageDir "d:\temp\test\in" --outputDir "d:\temp\test\out"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('detectorFile', type=str)
    parser.add_argument('imageFile', action='store', type=str, 
                        help='Can be a singe image file, a text file containing a list of images, or a directory')
    parser.add_argument('outputFile', type=str, 
                       help='Output results file')
    parser.add_argument('--imageDir', action='store', type=str, default='', 
                        help='Directory to search for images, with optional recursion')
    parser.add_argument('--threshold', action='store', type=float, 
                        default=DEFAULT_CONFIDENCE_THRESHOLD, 
                        help='Confidence threshold, don''t render boxes below this confidence')
    parser.add_argument('--recursive', action='store_true', 
                        help='Recurse into directories, only meaningful if using --imageDir')
    parser.add_argument('--forceCpu', action='store_true', 
                        help='Force CPU detection, even if a GPU is available')
    
    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()    
    
    options = BatchDetectionOptions()
    args_to_object(args,options)
        
    load_and_run_detector(options)
    

if __name__ == '__main__':
    
    main()

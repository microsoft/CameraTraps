######
#
# run_tf_detector.py
#
# Functions to load a TensorFlow detection model, run inference,
# render bounding boxes on images, and write out the resulting
# images (with bounding boxes).
#
# See the "test driver" cell for example invocation.
#
# It's clearly icky that if you give it blah.jpg, it writes the results to 
# blah.detections.jpg, but I'm defending this with the "just a demo script"
# argument.
#
######


#%% Constants, imports, environment

import tensorflow as tf
import numpy as np
import humanfriendly
import time
import matplotlib
import glob
# matplotlib.use('TkAgg')
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import os


#%% Core detection functions

def load_model(checkpoint):
    """
    Load a detection model (i.e., create a graph) from a .pb file
    """

    print('Creating Graph...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('...done')

    return detection_graph


def generate_detections(detection_graph,images):
    """
    boxes,scores,classes,images = generate_detections(detection_graph,images)

    Run an already-loaded detector network on a set of images.

    [images] can be a list of numpy arrays or a list of filenames.  Non-list inputs will be
    wrapped into a list.

    Boxes are returned in relative coordinates as (top, left, bottom, right); x,y origin is the upper-left.
    """

    if not isinstance(images,list):
        images = [images]
    else:
        images = images.copy()

    # Load images if they're not already numpy arrays
    for iImage,image in enumerate(images):
        if isinstance(image,str):
            print('Loading image {}'.format(image))
            # image = Image.open(image).convert("RGBA"); image = np.array(image)
            image = mpimg.imread(image)
            images[iImage] = image
        else:
            assert isinstance(image,np.ndarray)

    boxes = []
    scores = []
    classes = []
    
    nImages = len(images)

    with detection_graph.as_default():
        
        with tf.Session(graph=detection_graph) as sess:
            
            for iImage,imageNP in enumerate(images):                
                
                print('Processing image {} of {}'.format(iImage,nImages))
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
            
            # ...for each image                
            
    boxes = np.squeeze(np.array(boxes))
    scores = np.squeeze(np.array(scores))
    classes = np.squeeze(np.array(classes)).astype(int)

    return boxes,scores,classes,images


#%% Rendering functions

def render_bounding_box(box, score, classLabel, inputFileName, outputFileName=None,
                          confidenceThreshold=0.9,linewidth=2):
    """
    Convenience wrapper to apply render_bounding_boxes to a single image
    """
    outputFileNames = []
    if outputFileName is not None:
        outputFileNames = [outputFileName]
    scores = [[score]]
    boxes = [[box]]
    render_bounding_boxes(boxes,scores,[classLabel],[inputFileName],outputFileNames,
                          confidenceThreshold,linewidth)

def render_bounding_boxes(boxes, scores, classes, inputFileNames, outputFileNames=[],
                          confidenceThreshold=0.9,linewidth=2):
    """
    Render bounding boxes on the image files specified in [inputFileNames].  
    
    [boxes] and [scores] should be in the format returned by generate_detections, 
    specifically [top, left, bottom, right] in normalized units, where the
    origin is the upper-left.    
    
    "classes" is currently unused, it's a placeholder for adding text annotations
    later.
    """

    nImages = len(inputFileNames)
    iImage = 0

    for iImage in range(0,nImages):

        inputFileName = inputFileNames[iImage]

        if iImage >= len(outputFileNames):
            outputFileName = ''
        else:
            outputFileName = outputFileNames[iImage]

        if len(outputFileName) == 0:
            name, ext = os.path.splitext(inputFileName)
            outputFileName = "{}.{}{}".format(name,'_detections',ext)

        image = mpimg.imread(inputFileName)
        iBox = 0; box = boxes[iImage][iBox]
        dpi = 100
        s = image.shape; imageHeight = s[0]; imageWidth = s[1]
        figsize = imageWidth / float(dpi), imageHeight / float(dpi)

        plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1])
        
        # Display the image
        ax.imshow(image)
        ax.set_axis_off()
    
        # plt.show()
        for iBox,box in enumerate(boxes[iImage]):

            score = scores[iImage][iBox]
            if score < confidenceThreshold:
                continue

            # top, left, bottom, right 
            #
            # x,y origin is the upper-left
            topRel = box[0]
            leftRel = box[1]
            bottomRel = box[2]
            rightRel = box[3]
            
            x = leftRel * imageWidth
            y = topRel * imageHeight
            w = (rightRel-leftRel) * imageWidth
            h = (bottomRel-topRel) * imageHeight
            
            # Location is the bottom-left of the rect
            #
            # Origin is the upper-left
            iLeft = x
            iBottom = y
            rect = patches.Rectangle((iLeft,iBottom),w,h,linewidth=linewidth,edgecolor='r',
                                     facecolor='none')
            
            # Add the patch to the Axes
            ax.add_patch(rect)        

        # ...for each box

        # This is magic goop that removes whitespace around image plots (sort of)        
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, 
                            wspace = 0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.axis('tight')
        ax.set(xlim=[0,imageWidth],ylim=[imageHeight,0],aspect=1)
        plt.axis('off')                

        # plt.savefig(outputFileName, bbox_inches='tight', pad_inches=0.0, dpi=dpi, transparent=True)
        plt.savefig(outputFileName, dpi=dpi, transparent=True)
        plt.close()
        # os.startfile(outputFileName)

    # ...for each image

# ...def render_bounding_boxes


def load_and_run_detector(modelFile, imageFileNames, confidenceThreshold=0.85):
    
    # Load and run detector on target images
    detection_graph = load_model(modelFile)
    
    startTime = time.time()
    boxes,scores,classes,images = generate_detections(detection_graph,imageFileNames)
    elapsed = time.time() - startTime
    print("Done running detector on {} files in {}".format(len(images),
          humanfriendly.format_timespan(elapsed)))
    
    assert len(boxes) == len(imageFileNames)
    
    outputFileNames = []
    
    plt.ioff()
    
    render_bounding_boxes(boxes=boxes, scores=scores, classes=classes, 
                          inputFileNames=imageFileNames, outputFileNames=outputFileNames,
                          confidenceThreshold=confidenceThreshold)


#%% Interactive driver

if False:
    
    #%%
    
    modelFile = r'D:\temp\models\object_detection\megadetector\megadetector_v2.pb'
    imageDir = r'D:\temp\demo_images\b2'    
    imageFileNames = [fn for fn in glob.glob(os.path.join(imageDir,'*.jpg'))
         if (not 'detections' in fn)]
    
    load_and_run_detector(modelFile,imageFileNames)
    
    
#%% Command-line driver
    
import argparse

def main():
    
    # python run_tf_detector.py "D:\temp\models\object_detection\megadetector\megadetector_v2.pb" --imageDir "D:\temp\demo_images\b2"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('detectorFile', type=str)
    parser.add_argument('--imageDir', action="store", type=str, default='')
    parser.add_argument('--imageFile', action="store", type=str, default='')
    parser.add_argument('--threshold', action="store", type=float, default=0.85)
    
    args = parser.parse_args()    
    
    if len(args.imageFile) > 0 and len(args.imageDir) > 0:
        raise Exception('Cannot specify both image file and image dir')
    elif len(args.imageFile) > 0:
        imageFileNames = [args.imageFile]
    else:
        imageFileNames = [fn for fn in glob.glob(os.path.join(args.imageDir,'*.jpg')) 
            if (not 'detections' in fn)]
                
    print('Running detector on {} images'.format(len(imageFileNames)))    
    
    load_and_run_detector(modelFile=args.detectorFile, imageFileNames=imageFileNames, 
                          confidenceThreshold=args.threshold)
    

if __name__ == '__main__':
    
    main()

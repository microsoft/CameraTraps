######
#
# detect_and_predict_image.py
#
# Functions to load a TensorFlow detection and a classification model, run inference,
# render bounding boxes on images, and write out the resulting
# images (with bounding boxes and classes).
#
# See the "test driver" cell for example invocation.
#
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
import matplotlib
import PIL
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import os

# Minimum detection confidence for showing a bounding box on the output image
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

# Stick this into filenames before the extension for the rendered result
DETECTION_FILENAME_INSERT = '_detections'

# Number of top-scoring classes to show at each bounding box
NUM_ANNOTATED_CLASSES = 3


#%% Core detection functions

def load_model(checkpoint):
    """
    Load a detection model (i.e., create a graph) from a .pb file
    """

    print('Creating Graph...')
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('...done')

    return graph


def generate_detections(detection_graph,images):
    """
    boxes,scores,classes,images = generate_detections(detection_graph,images)

    Run an already-loaded detector network on a set of images.

    [images] can be a list of numpy arrays or a list of filenames.  Non-list inputs will be
    wrapped into a list.

    Boxes are returned in relative coordinates as (top, left, bottom, right);
    x,y origin is the upper-left.

    [boxes] will be returned as a numpy array of size nImages x nDetections x 4.

    [scores] and [classes] will each be returned as a numpy array of size nImages x nDetections.

    [images] is a set of numpy arrays corresponding to the input parameter [images], which may have
    have been either arrays or filenames.
    """

    if not isinstance(images,list):
        images = [images]
    else:
        images = images.copy()

    # Load images if they're not already numpy arrays
    # iImage = 0; image = images[iImage]
    for iImage,image in enumerate(images):
        if isinstance(image,str):
            print('Loading image {}'.format(image))

            # Load the image as an nparray of size h,w,nChannels

            # There was a time when I was loading with PIL and switched to mpimg,
            # but I can't remember why, and converting to RGB is a very good reason
            # to load with PIL, since mpimg doesn't give any indication of color
            # order, which basically breaks all .png files.
            #
            # So if you find a bug related to using PIL, update this comment
            # to indicate what it was, but also disable .png support.
            image = PIL.Image.open(image).convert("RGB"); image = np.array(image)
            # image = mpimg.imread(image)

            # This shouldn't be necessarily when loading with PIL and converting to RGB
            nChannels = image.shape[2]
            if nChannels > 3:
                print('Warning: trimming channels from image')
                image = image[:,:,0:3]
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

    return boxes,scores,classes,images


def classify_boxes(classification_graph, boxes, scores, classes, images, confidence_threshold, padding_factor=1.6):
    '''
    Takes a classification model and applies it to all detected boxes with a detection confidence
    larger than confidence_threshold.

    Args:
        classification_graph: frozen graph model that includes the TF-slim preprocessing. i.e. it will be given a cropped
                              images with values in [0,1]
        boxes, scores, classes, images: The return values of generate_detections(detection_graph,images) where
                              classes corresponds to the detection class and not the species return by the classification model,
                              and is unused at the moment.

    Returns species_scores with shape len(images) x len(boxes) x num_species. However, num_species is 0 if the
    corresponding box is below the confidence_threshold.
    '''

    nImages = len(images)
    iImage = 0

    species_scores = []

    with classification_graph.as_default():

        with tf.Session(graph=classification_graph) as sess:

            # Get input and output tensors of classification model
            image_tensor = classification_graph.get_tensor_by_name('input:0')
            predictions_tensor = classification_graph.get_tensor_by_name('output:0')
            predictions_tensor = tf.squeeze(predictions_tensor, [0])

            for iImage in range(0,nImages):

                species_scores.append([])
                image = images[iImage]
                assert len(image.shape) == 3
                image_height, image_width, _ = image.shape

                #imsize = cur_image['width'], cur_image['height']
                # Select detections with a confidence larger 0.5
                selection = scores[iImage] > confidence_threshold
                # Get these boxes and convert normalized coordinates to pixel coordinates
                selected_boxes = (boxes[iImage][selection] * np.tile([image_height, image_width], (1,2)))
                # Pad the detected animal to a square box and additionally by PADDING_FACTOR, the result will be in crop_boxes
                # However, we need to make sure that it box coordinates are still within the image
                bbox_sizes = np.vstack([selected_boxes[:,2] - selected_boxes[:,0], selected_boxes[:,3] - selected_boxes[:,1]]).T
                offsets = (padding_factor * np.max(bbox_sizes, axis=1, keepdims=True) - bbox_sizes) / 2
                crop_boxes = selected_boxes + np.hstack([-offsets,offsets])
                crop_boxes = np.maximum(0,crop_boxes).astype(int)
                # For convenience:
                # Create an array with contains the index of the corresponding crop_box for each selected box
                # i.e. [False False 0 False 1 2 3 False False]
                selection_box_idx = np.copy(selection).astype(int)
                selection_box_idx[selection] = range(sum(selection))

                # For each box
                for iBox in range(len(boxes[iImage])):

                    # If this box should be classified
                    if selection[iBox]:
                        crop_box = crop_boxes[selection_box_idx[iBox]]
                        cropped_img = image[crop_box[0]:crop_box[2], crop_box[1]:crop_box[3]]
                        input_image = cropped_img.astype(float) / 255
                        # Run inference
                        predictions = sess.run(predictions_tensor, feed_dict={image_tensor: input_image})
                        species_scores[iImage].append(predictions)

                    # if box should not be classified
                    else:
                        species_scores[iImage].append([])

                # ...for each box

                # species_scores should have shape len(images) x len(boxes) x num_species
                assert len(species_scores[iImage]) == len(boxes[iImage])

            # ...for each image

        # ...with tf.Session

    # with classification_graph

    # species_scores should have shape len(images) x len(boxes) x num_species
    assert len(species_scores) == len(images)

    return species_scores


#%% Rendering functions

def render_bounding_box(box, score, classLabel, inputFileName, outputFileName=None,
                          confidenceThreshold=DEFAULT_CONFIDENCE_THRESHOLD,linewidth=2):
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

def render_bounding_boxes(boxes, scores, species_scores, class_names, input_file_names, output_file_names=[],
                          confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, num_annotated_classes=NUM_ANNOTATED_CLASSES,
                          linewidth=2):
    """
    Render bounding boxes on the image files specified in [input_file_names].

    [boxes] and [scores] should be in the format returned by generate_detections,
    specifically [top, left, bottom, right] in normalized units, where the
    origin is the upper-left.

    [species_scores] should be in the format returned by classify_boxes
    """

    nImages = len(input_file_names)
    iImage = 0

    for iImage in range(0,nImages):

        input_file_name = input_file_names[iImage]

        if iImage >= len(output_file_names):
            output_file_name = ''
        else:
            output_file_name = output_file_names[iImage]

        if len(output_file_name) == 0:
            name, ext = os.path.splitext(input_file_name)
            output_file_name = "{}{}{}".format(name,DETECTION_FILENAME_INSERT,ext)

        image = mpimg.imread(input_file_name)
        iBox = 0; box = boxes[iImage][iBox]
        dpi = 100
        s = image.shape; image_height = s[0]; image_width = s[1]
        figsize = image_width / float(dpi), image_height / float(dpi)

        plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1])

        # Display the image
        ax.imshow(image)
        ax.set_axis_off()

        # plt.show()
        for iBox,box in enumerate(boxes[iImage]):

            score = scores[iImage][iBox]
            if score < confidence_threshold:
                continue

            # top, left, bottom, right
            #
            # x,y origin is the upper-left
            topRel = box[0]
            leftRel = box[1]
            bottomRel = box[2]
            rightRel = box[3]

            x = leftRel * image_width
            y = topRel * image_height
            w = (rightRel-leftRel) * image_width
            h = (bottomRel-topRel) * image_height

            # Generate bounding box text
            box_species_scores = species_scores[iImage][iBox]
            box_text = []
            for species_id in (-box_species_scores).argsort()[:num_annotated_classes]:
                box_text.append('{} {:.1%}'.format(class_names[species_id], box_species_scores[species_id]))
            box_text = '\n'.join(box_text)
            # Choose color based on class
            edge_color = plt.get_cmap('rainbow')(box_species_scores.argmax()/box_species_scores.size)

            # Location is the bottom-left of the rect
            #
            # Origin is the upper-left
            iLeft = x
            # iRight = x + w
            iBottom = y
            # iTop = y + h
            rect = patches.Rectangle((iLeft,iBottom),w,h,linewidth=linewidth,edgecolor=edge_color,
                                     facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add class description
            # First determine best location by finding the corner that is closest to the image center
            # relative corner coordinates
            corners = np.array([[leftRel, bottomRel], [leftRel, topRel], [rightRel, bottomRel], [rightRel, topRel]])
            # relative coordinates of image center
            center = np.array([[0.5, 0.5]])
            # Compute pair-wise squared distance and get the index of the one with minimal distance
            best_corner_idx = ((center - corners)**2).sum(axis=1).argmin()
            # Get the corresponding coordinates ...
            best_corner = corners[best_corner_idx] * np.array([image_width, image_height])
            # ... and alignment for the text box 
            alignment_styles = [dict(horizontalalignment='left', verticalalignment='top'),
                              dict(horizontalalignment='left', verticalalignment='bottom'),
                              dict(horizontalalignment='right', verticalalignment='top'),
                              dict(horizontalalignment='right', verticalalignment='bottom')]
            best_alignment = alignment_styles[best_corner_idx]

            # Plot the text box with background
            ax.annotate(box_text, xy=best_corner, bbox=dict(boxstyle="round", color=edge_color, ec="0.5", alpha=1), **best_alignment)

        # ...for each box

        # This is magic goop that removes whitespace around image plots (sort of)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0,
                            wspace = 0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.axis('tight')
        ax.set(xlim=[0,image_width],ylim=[image_height,0],aspect=1)
        plt.axis('off')

        # plt.savefig(outputFileName, bbox_inches='tight', pad_inches=0.0, dpi=dpi, transparent=True)
        plt.savefig(output_file_name, dpi=dpi, transparent=True)
        plt.close()
        # os.startfile(outputFileName)

    # ...for each image

# ...def render_bounding_boxes


def load_and_run_detector(detector_file, classifier_file, classes_file, image_file_names,
                          confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, num_annotated_classes=NUM_ANNOTATED_CLASSES,
                          detection_graph=None, classification_graph=None):

    if len(image_file_names) == 0:
        print('Warning: no files available')
        return

    # Load and run detector on target images
    if detection_graph is None:
        detection_graph = load_model(detector_file)

    if classification_graph is None:
        classification_graph = load_model(classifier_file)

    startTime = time.time()
    boxes, scores, detection_classes, images = generate_detections(detection_graph,image_file_names)
    species_scores = classify_boxes(classification_graph, boxes, scores, detection_classes, images, confidence_threshold)
    elapsed = time.time() - startTime
    print("Done running detector and classifier on {} files in {}".format(len(images),
          humanfriendly.format_timespan(elapsed)))

    # Read the name of all classes
    with open(classes_file, 'rt') as fi:
        class_names = fi.read().splitlines()
        # remove empty lines
        class_names = [cn for cn in class_names if cn.strip()]

    assert len(boxes) == len(image_file_names)

    output_file_names = []

    plt.ioff()

    render_bounding_boxes(boxes=boxes, scores=scores, species_scores=species_scores, class_names=class_names,
                          input_file_names=image_file_names, output_file_names=output_file_names,
                          confidence_threshold=confidence_threshold, num_annotated_classes=num_annotated_classes)

    return detection_graph, classification_graph


#%% Interactive driver

if False:

    #%%

    detection_graph = None
    classification_graph = None

    #%%

    detection_file = r'/ai4edevfs/models/object_detection/faster_rcnn_inception_resnet_v2_atrous/megadetector/frozen_inference_graph.pb'
    classification_file = r'/ai4edevfs/models/serengeti_cropped/serengeti_cropped_latest_inceptionv4_89.5/all/frozen_inference_graph_w_preprocessing.pb'
    classes_file = r'/ai4edevfs/models/serengeti_cropped/serengeti_cropped_latest_inceptionv4_89.5/classlist.txt'
    image_dir = r'./sample_output'
    image_file_names = [fn for fn in glob.glob(os.path.join(image_dir,'*.JPG'))
         if (not 'detections' in fn)]
    image_file_names = [r'./sample_output/mongoose___S1___R11___R11_R1___S1_R11_R1_PICT0101_1.JPG']

    detection_graph, classification_graph = load_and_run_detector(detection_file, classification_file, classes_file, image_file_names,
                                            DEFAULT_CONFIDENCE_THRESHOLD, NUM_ANNOTATED_CLASSES, detection_graph, classification_graph)


#%% File helper functions

imageExtensions = ['.jpg','.jpeg','.gif','.png']
imageExtensions = imageExtensions + [ext.upper() for ext in imageExtensions]

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


#%% Command-line driver

def main():

    parser = argparse.ArgumentParser(description='Runs both a detector and a classifier on a given image.')
    parser.add_argument('detector_file', type=str, help='Generic detector for animals.',
                       metavar='PATH_TO_DETECTOR')
    parser.add_argument('classifier_file', type=str, help='Frozen graph for classification including pre-processing. The graphs ' + \
                        ' will receive an image with values in [0,1], so double check that you use the correct model. The script ' + \
                        ' `export_inference_graph_serengeti.sh` shows how to create such a model',
                       metavar='PATH_TO_CLASSIFIER_W_PREPROCESSING')
    parser.add_argument('--classes_file', action='store', type=str, help='File with the class names. Each line should contain ' + \
                        ' one name and the first line should correspond to the first output, the second line to the second model output, etc.')
    parser.add_argument('--image_dir', action='store', type=str, default='', help='Directory to search for images, with optional recursion')
    parser.add_argument('--image_file', action='store', type=str, default='', help='Single file to process, mutually exclusive with imageDir')
    parser.add_argument('--threshold', action='store', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Confidence threshold, don't render boxes below this confidence. Default: %.2f"%DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument('--num_annotated_classes', action='store', type=int, default=NUM_ANNOTATED_CLASSES,
                        help='Number of classes to annotated for each bounding box, default: %d'%NUM_ANNOTATED_CLASSES)
    parser.add_argument('--recursive', action='store_true', help='Recurse into directories, only meaningful if using --imageDir')

    if len(sys.argv[1:])==0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if len(args.image_file) > 0 and len(args.image_dir) > 0:
        raise Exception('Cannot specify both image file and image dir')
    elif len(args.image_file) == 0 and len(args.image_dir) == 0:
        raise Exception('Must specify either an image file or an image directory')

    if len(args.image_file) > 0:
        image_file_names = [args.image_file]
    else:
        image_file_names = findImages(args.image_dir,args.recursive)

    # Hack to avoid running on already-detected images
    image_file_names = [x for x in image_file_names if DETECTION_FILENAME_INSERT not in x]

    print('Running detector on {} images'.format(len(image_file_names)))

    load_and_run_detector(detector_file=args.detector_file, classifier_file=args.classifier_file, classes_file=args.classes_file,
                          image_file_names=image_file_names, confidence_threshold=args.threshold, num_annotated_classes=args.num_annotated_classes)


if __name__ == '__main__':

    main()

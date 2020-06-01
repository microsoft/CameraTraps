"""
TODO - Copy this module from the classification folder once it has been refactored so that we do not keep
duplicated functions
"""

import math

import numpy as np
import tensorflow as tf
import tqdm
from animal_detection_classification_api import api_config


class TFClassifier(object):
    def __init__(self, checkpoints, class_names):
        self.models = {}
        self.class_names = {}
        for name, checkpoint in checkpoints.items():
            self.models[name] = self.load_model(checkpoint)
            if name in class_names:
                self.class_names[name] = self.load_class_names(class_names[name])
            else:
                self.class_names[name] = []

            
            self.detection_category_whitelist = api_config.DETECTION_CATEGORY_WHITELIST
            assert all([isinstance(x, str) for x in self.detection_category_whitelist])

            self.padding_factor = api_config.PADDING_FACTOR

            # Minimum detection confidence for showing a bounding box on the output image
            self.default_confidence_threshold = api_config.DEFAULT_CONFIDENCE_THRESHOLD

            # Number of top-scoring classes to show at each bounding box
            self.num_annotated_classes = api_config.NUM_ANNOTATED_CLASSES

            # Number of significant float digits in JSON output
            self.num_significant_digits = api_config.NUM_SIGNIFICANT_DIGITS


    def load_model(self, checkpoint):
        """
        Load a classification model (i.e., create a graph) from a .pb file
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


    def load_class_names(self, file_path):
        """
        Load a class name json file
        """

        print('Loading Class name file ...')
        with open(file_path, 'rt') as fi:
            class_names = fi.read().splitlines()
                # remove empty lines
            class_names = [cn for cn in class_names if cn.strip()]

        return class_names

    def truncate_float(self, x, precision=3):
        """
        Function for truncating a float scalar to the defined precision.
        For example: truncate_float(0.0003214884) --> 0.000321
        This function is primarily used to achieve a certain float representation
        before exporting to JSON

        Args:
        x         (float) Scalar to truncate
        precision (int)   The number of significant digits to preserve, should be
                        greater or equal 1
        """
        
        assert precision > 0

        if np.isclose(x, 0):
            return 0
        else:
            # Determine the factor, which shifts the decimal point of x
            # just behind the last significant digit
            factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
            # Shift decimal point by multiplicatipon with factor, flooring, and 
            # division by factor
            return math.floor(x * factor)/factor


    def classify_boxes(self, images, image_names, detection_json, classification):
        classification_graph = self.models[classification]
        class_names = self.class_names[classification]
        # json_with_classes = self.add_classification_categories(detection_json, class_names)
        classification_predictions = {}

        with classification_graph.as_default():
            with tf.Session(graph=classification_graph) as sess:
                # Get input and output tensors of classification model
                image_tensor = classification_graph.get_tensor_by_name('input:0')
                predictions_tensor = classification_graph.get_tensor_by_name('output:0')
                predictions_tensor = tf.squeeze(predictions_tensor, [0])
                
                # For each image
                n_images = len(images)
                for i_image in tqdm.tqdm(list(range(0,n_images))):
                    images = [np.asarray(image, np.uint8) for image in images]
                    image_data = images[i_image]
                    
                    # Scale pixel values to [0,1]
                    image_data = image_data / 255
                    image_height, image_width, _ = image_data.shape

                    image_description = detection_json[image_names[i_image]]
                    classification_predictions[image_names[i_image]] = list()
                    # For each box
                    n_detections = len(image_description)
                    for i_box in range(n_detections):

                        cur_detection = image_description[i_box]

                        # Skip detections with low confidence
                        if cur_detection[4] < self.default_confidence_threshold:
                            continue

                        # Skip if detection category is not in whitelist
                        if not str(cur_detection[5]) in self.detection_category_whitelist:
                            continue

                        # box ymin, xim, ymax, xmax
                        x_min = cur_detection[1]
                        y_min = cur_detection[0]
                        width_of_box = cur_detection[1] + cur_detection[3]
                        height_of_box = cur_detection[0] + cur_detection[2]

                        # Get current box in relative coordinates and format [x_min, y_min, width_of_box, height_of_box]
                        box_orig = [x_min, y_min, width_of_box, height_of_box]
                        # Convert to [ymin, xmin, ymax, xmax] and
                        # store it as 1x4 numpy array so we can re-use the generic multi-box padding code
                        box_coords = np.array([[box_orig[1],
                                                box_orig[0],
                                                box_orig[1]+box_orig[3],
                                                box_orig[0]+box_orig[2]
                                            ]])
                        # Convert normalized coordinates to pixel coordinates
                        box_coords_abs = (box_coords * np.tile([image_height, image_width], (1,2)))
                        # Pad the detected animal to a square box and additionally by PADDING_FACTOR, the result will be in crop_boxes
                        # However, we need to make sure that it box coordinates are still within the image
                        bbox_sizes = np.vstack([box_coords_abs[:,2] - box_coords_abs[:,0], box_coords_abs[:,3] - box_coords_abs[:,1]]).T
                        offsets = (self.padding_factor * np.max(bbox_sizes, axis=1, keepdims=True) - bbox_sizes) / 2
                        crop_boxes = box_coords_abs + np.hstack([-offsets,offsets])
                        crop_boxes = np.maximum(0,crop_boxes).astype(int)
                        # Get the first (and only) row as our bbox to classify
                        crop_box = crop_boxes[0]
                        # Get the image data for that box
                        cropped_img = image_data[crop_box[0]:crop_box[2], crop_box[1]:crop_box[3]]

                        # Run inference
                        predictions = sess.run(predictions_tensor, feed_dict={image_tensor: cropped_img})
                        current_predicitions = []        
                        # Add the *num_annotated_classes* top scoring classes
                        for class_idx in np.argsort(-predictions)[:self.num_annotated_classes]:
                            class_conf = self.truncate_float(predictions[class_idx].item())         
                            for idx, name in enumerate(class_names):
                                if class_idx == idx:
                                    current_predicitions.append([f'{name}', class_conf])
                                

                        classification_predictions[image_names[i_image]].append(current_predicitions)
                        
        return classification_predictions
    

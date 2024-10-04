# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Video detection demo """
#%% 
# Importing necessary basic libraries and modules
import numpy as np
import supervision as sv

#%% 
# PyTorch imports for tensor operations
import torch
import os
#%% 
# Importing the models, transformations, and utility functions from PytorchWildlife 
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife import utils as pw_utils

#%% 
# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SOURCE_VIDEO_PATH = os.path.join(".","demo_data","videos","opossum_example.MP4")
TARGET_VIDEO_PATH = os.path.join(".","demo_data","videos","opossum_example_processed.MP4")

#%% 
# Initializing the MegaDetectorV6 model for image detection
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="yolov9c")

# Uncomment the following line to use MegaDetectorV5 instead of MegaDetectorV6
#detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True, version="a")

#%% 
# Initializing the model for image classification
classification_model = pw_classification.AI4GOpossum(device=DEVICE, pretrained=True)

#%% 
# Initializing a box annotator for visualizing detections
box_annotator = sv.BoxAnnotator(thickness=4)
lab_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_thickness=4, text_scale=2)

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    """
    Callback function to process each video frame for detection and classification.
    
    Parameters:
    - frame (np.ndarray): Video frame as a numpy array.
    - index (int): Frame index.
    
    Returns:
    annotated_frame (np.ndarray): Annotated video frame.
    """
    
    results_det = detection_model.single_image_detection(frame, img_path=index)

    labels = []

    for xyxy in results_det["detections"].xyxy:
        cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
        results_clf = classification_model.single_image_classification(cropped_image)
        labels.append("{} {:.2f}".format(results_clf["prediction"], results_clf["confidence"]))

    annotated_frame = lab_annotator.annotate(
        scene=box_annotator.annotate(
            scene=frame,
            detections=results_det["detections"],
        ),
        detections=results_det["detections"],
        labels=labels,
    )
    
    return annotated_frame 

# Processing the video and saving the result with annotated detections and classifications
pw_utils.process_video(source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH, callback=callback, target_fps=10)

# %%

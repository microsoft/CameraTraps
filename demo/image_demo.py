# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Demo for image detection"""

#%% 
# Importing necessary basic libraries and modules
import os
# PyTorch imports 
import torch

#%% 
# Importing the model, dataset, transformations and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils

#%% 
# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% 
# Initializing the MegaDetectorV6 model for image detection
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="yolov9c")

# Uncomment the following line to use MegaDetectorV5 instead of MegaDetectorV6
#detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True, version="a")

#%% Single image detection
# Specifying the path to the target image TODO: Allow argparsing
tgt_img_path = os.path.join(".","demo_data","imgs","10050028_0.JPG")

# Performing the detection on the single image
results = detection_model.single_image_detection(tgt_img_path)

# Saving the detection results 
pw_utils.save_detection_images(results, os.path.join(".","demo_output"), overwrite=False)

# Saving the detected objects as cropped images
pw_utils.save_crop_images(results, os.path.join(".","crop_output"), overwrite=False)

#%% Batch detection
""" Batch-detection demo """

# Specifying the folder path containing multiple images for batch detection
tgt_folder_path = os.path.join(".","demo_data","imgs")

# Performing batch detection on the images
results = detection_model.batch_image_detection(tgt_folder_path, batch_size=16)

#%% Output to annotated images
# Saving the batch detection results as annotated images
pw_utils.save_detection_images(results, "batch_output", tgt_folder_path, overwrite=False)

#%% Output to cropped images
# Saving the detected objects as cropped images
pw_utils.save_crop_images(results, "crop_output", tgt_folder_path, overwrite=False)

#%% Output to JSON results
# Saving the detection results in JSON format
pw_utils.save_detection_json(results, os.path.join(".","batch_output.json"),
                             categories=detection_model.CLASS_NAMES,
                             exclude_category_ids=[], # Category IDs can be found in the definition of each model.
                             exclude_file_path=None)

# Saving the detection results in timelapse JSON format
pw_utils.save_detection_timelapse_json(results, os.path.join(".","batch_output_timelapse.json"),
                                       categories=detection_model.CLASS_NAMES,
                                       exclude_category_ids=[], # Category IDs can be found in the definition of each model.
                                       exclude_file_path=tgt_folder_path,
                                       info={"detector": "MegaDetectorV6"})
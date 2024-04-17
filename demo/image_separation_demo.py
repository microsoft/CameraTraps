# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Demo for image separation between positive and negative detections"""

#%% 
# Importing necessary basic libraries and modules
import argparse
import os
import json
import shutil
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

# PyTorch imports 
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data 
from PytorchWildlife import utils as pw_utils

#%% Argument parsing
parser = argparse.ArgumentParser(description="Batch image detection and separation")
parser.add_argument('--image_folder', type=str, default=os.path.join("demo_data","imgs"), help='Folder path containing images for detection')
parser.add_argument('--output_path', type=str, default='folder_separation', help='Path where the outputs will be saved')
parser.add_argument('--file_extension', type=str, default='JPG', help='File extension for images (case sensitive)')
parser.add_argument('--threshold', type=float, default='0.2', help='Confidence threshold to consider a detection as positive')
args = parser.parse_args()

#%% 
# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% 
# Initializing the MegaDetectorV5 model for image detection
detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True)

# Initializing the Yolo-specific transform for the image
transform = pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                               stride=detection_model.STRIDE)

#%% Batch detection
""" Batch-detection demo """

# Creating a dataset of images with the specified transform
dataset = pw_data.DetectionImageFolder(
    args.image_folder,
    transform=pw_trans.MegaDetector_v5_Transform(target_size=detection_model.IMAGE_SIZE,
                                                 stride=detection_model.STRIDE),
    extension=args.file_extension
)

# Creating a DataLoader for batching and parallel processing of the images
loader = DataLoader(dataset, batch_size=32, shuffle=False, 
                    pin_memory=True, num_workers=0, drop_last=False)

# Performing batch detection on the images
results = detection_model.batch_image_detection(loader)

#%% Output to JSON results
# Saving the detection results in JSON format
os.makedirs(args.output_path, exist_ok=True)
json_file = os.path.join(args.output_path, "detection_results.json")
pw_utils.save_detection_json(results, json_file,
                             categories=detection_model.CLASS_NAMES,
                             exclude_category_ids=[], # Category IDs can be found in the definition of each model.
                             exclude_file_path=None)

# Separate the positive and negative detections through file copying:
pw_utils.process_detections(json_file, args.output_path, args.threshold)
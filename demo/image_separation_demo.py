# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Demo for image separation between positive and negative detections"""

#%% 
# Importing necessary basic libraries and modules
import argparse
import os
import torch

# PyTorch imports 
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils

#%% Argument parsing
parser = argparse.ArgumentParser(description="Batch image detection and separation")
parser.add_argument('--image_folder', type=str, default=os.path.join(".","demo_data","imgs"), help='Folder path containing images for detection')
parser.add_argument('--output_path', type=str, default='folder_separation', help='Path where the outputs will be saved')
parser.add_argument('--threshold', type=float, default='0.2', help='Confidence threshold to consider a detection as positive')
args = parser.parse_args()

#%% 
# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% 
# Initializing the MegaDetectorV6 model for image detection
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="yolov9c")

# Uncomment the following line to use MegaDetectorV5 instead of MegaDetectorV6
#detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True, version="a")

#%% Batch detection
""" Batch-detection demo """

# Performing batch detection on the images
results = detection_model.batch_image_detection(args.image_folder, batch_size=16)

#%% Output to JSON results
# Saving the detection results in JSON format
os.makedirs(args.output_path, exist_ok=True)
json_file = os.path.join(args.output_path, "detection_results.json")
pw_utils.save_detection_json(results, json_file,
                             categories=detection_model.CLASS_NAMES,
                             exclude_category_ids=[], # Category IDs can be found in the definition of each model.
                             exclude_file_path=args.image_folder)

# Separate the positive and negative detections through file copying:
pw_utils.detection_folder_separation(json_file, args.image_folder, args.output_path, args.threshold)
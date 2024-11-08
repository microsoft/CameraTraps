# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Demo for Herdnet image detection"""
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
# Initializing the HerdNet model for image detection
detection_model = pw_detection.HerdNet(device=DEVICE)
# If you want to use ennedi dataset weigths, you can use the following line:
# detection_model = pw_detection.HerdNet(device=DEVICE, dataset="ennedi")

#%% Single image detection
img_path = os.path.join(".","demo_data","herdnet_imgs","S_11_05_16_DSC01556.JPG")

# Performing the detection on the single image
results = detection_model.single_image_detection(img=img_path)

#%% Output to annotated images
# Saving the batch detection results as annotated images
pw_utils.save_detection_images_dots(results, os.path.join(".","herdnet_demo_output"), overwrite=False)

#%% Batch image detection
""" Batch-detection demo """

# Specifying the folder path containing multiple images for batch detection
folder_path = os.path.join(".","demo_data","herdnet_imgs")

# Performing batch detection on the images
results = detection_model.batch_image_detection(folder_path, batch_size=1) # NOTE: Only use batch size 1 because each image is divided into patches and this batch is enough. 

#%% Output to annotated images
# Saving the batch detection results as annotated images
pw_utils.save_detection_images_dots(results, "herdnet_demo_batch_output", folder_path, overwrite=False)

# Saving the detection results in JSON format
pw_utils.save_detection_json_as_dots(results, os.path.join(".","herdnet_demo_batch_output.json"),
                             categories=detection_model.CLASS_NAMES,
                             exclude_category_ids=[], # Category IDs can be found in the definition of each model.
                             exclude_file_path=None)

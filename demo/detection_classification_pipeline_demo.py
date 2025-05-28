# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Demo for image detection"""

#%% 
# Importing necessary basic libraries and modules
import os
import numpy as np
from PIL import Image
import supervision as sv

# PyTorch imports 
import torch
from torch.utils.data import DataLoader

#%% 
# Importing the model, dataset, transformations and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils

from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data 

#%% 
# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% 
# Initializing the MegaDetectorV6 model for image detection
# Valid versions are MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e or MDV6-rtdetr-c
detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov10-e")

# Uncomment the following line to use MegaDetectorV5 instead of MegaDetectorV6
# detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True, version="a")

# %%
# Initializing a classification model for image classification
# classification_model = pw_classification.DFNE(device=DEVICE)
classification_model = pw_classification.AI4GAmazonRainforest(device=DEVICE, version='v2')

#%% Single image detection
# Specifying the path to the target image TODO: Allow argparsing
tgt_img_path = os.path.join(".","demo_data","imgs","10050028_0.JPG")

# Performing the detection on the single image
results = detection_model.single_image_detection(tgt_img_path)

clf_conf_thres = 0.8
input_img = np.array(Image.open(tgt_img_path).convert('RGB'))
clf_labels = []
for i, (xyxy, det_id) in enumerate(zip(results["detections"].xyxy, results["detections"].class_id)):
    # Only run classifier when detection class is animal
    if det_id == 0:
        cropped_image = sv.crop_image(image=input_img, xyxy=xyxy)
        results_clf = classification_model.single_image_classification(cropped_image)
        clf_labels.append("{} {:.2f}".format(results_clf["prediction"] if results_clf["confidence"] > clf_conf_thres else "Unknown",
                                             results_clf["confidence"]))
    else:
        clf_labels.append(results["labels"][i])

results["labels"] = clf_labels

# %%
# Saving the detection results 
pw_utils.save_detection_images(results, os.path.join(".","demo_output"), overwrite=False)

# %%# Saving the detected objects as cropped images
pw_utils.save_crop_images(results, os.path.join(".","crop_output"), overwrite=False)

#%% Batch detection
""" Batch-detection demo """

# Specifying the folder path containing multiple images for batch detection
tgt_folder_path = os.path.join(os.getcwd(),"demo_data","classification_examples")

# Performing batch detection on the images
det_results = detection_model.batch_image_detection(tgt_folder_path, batch_size=16)

clf_results = classification_model.batch_image_classification(det_results=det_results)

# %%
merged_results = det_results.copy()
clf_conf_thres = 0.8
clf_counter = 0

# %%
for det in merged_results:
    clf_labels = []
    for i, (xyxy, det_id) in enumerate(zip(det["detections"].xyxy, det["detections"].class_id)):
        if det_id == 0:
            clf_labels.append("{} {:.2f}".format(clf_results[clf_counter]["prediction"] if clf_results[clf_counter]["confidence"] > clf_conf_thres else "Unknown",
                                                 clf_results[clf_counter]["confidence"]))
            clf_counter += 1
        else:
            clf_labels.append(det["labels"][i])
        
    det["labels"] = clf_labels

#%% Output to annotated images
# Saving the batch detection results as annotated images
pw_utils.save_detection_images(merged_results, "batch_output", tgt_folder_path, overwrite=False)

#%% Output to cropped images
# Saving the detected objects as cropped images
pw_utils.save_crop_images(merged_results, "crop_output", tgt_folder_path, overwrite=False)

#%% Output to JSON results
# Saving the detection results in JSON format
pw_utils.save_detection_classification_json(det_results=det_results,
                                            clf_results=clf_results,
                                            det_categories=detection_model.CLASS_NAMES,
                                            clf_categories=classification_model.CLASS_NAMES,
                                            output_path=os.path.join(".","batch_output_classification.json"))

# %%
# Saving the detection results in timelapse JSON format
pw_utils.save_detection_classification_timelapse_json(det_results=det_results,
                                            clf_results=clf_results,
                                            det_categories=detection_model.CLASS_NAMES,
                                            clf_categories=classification_model.CLASS_NAMES,
                                            output_path=os.path.join(".","batch_output_classification_timelapse.json"))
# %%
# Saving the detection results in darwin core CSV format
pw_utils.save_detection_classification_csv_dwc(det_results=det_results,
                                            clf_results=clf_results,
                                            det_categories=detection_model.CLASS_NAMES,
                                            clf_categories=classification_model.CLASS_NAMES,
                                            output_path=os.path.join(".","batch_output_classification_darwincore.csv"),
                                            model_name="MDV6-yolov10-e")
# %%
# Separate the positive and negative detections through file copying
json_file = os.path.join(".","batch_output_classification.json")
output_path = os.path.join(".","folder_separation")
det_threshold = 0.2
clf_threshold = 0.2
overwrite = True
pw_utils.detection_classification_folder_separation(json_file, tgt_folder_path, output_path, det_threshold, clf_threshold, overwrite)
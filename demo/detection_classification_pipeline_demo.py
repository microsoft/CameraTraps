# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#%% 
# Importing necessary basic libraries and modules
import os
import numpy as np
from PIL import Image
import supervision as sv

# PyTorch imports 
import torch
from torch.utils.data import DataLoader

# Importing the model, dataset, transformations and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.data import datasets as pw_data 

#%% 
# Setting the device to use for computations ('cuda' indicates GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#%% Configuration
crop_detections = False

if crop_detections:
    CROP_OUT_DIR = os.path.join(".", "crop_output")
    os.makedirs(CROP_OUT_DIR, exist_ok=True)

USE_MDV5   = True  
MD_VERSION = "MDV6-yolov10-e" # Valid versions are MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e or MDV6-rtdetr-c
CLASSIFIER = "AI4GAmazonRainforest" # Or DFNE

#Thresholds
CLASSIFIER_CONF_THRESH = 0.80
DET_THRESH_FOR_FOLDER  = 0.20  
CLF_THRESH_FOR_FOLDER  = 0.20

BATCH_SIZE = 16

#%% Initialize models

# Initialize detection model
if not USE_MDV5:
    detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version=MD_VERSION)
else:
    detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True, version="a")

# Initialize classification model
if CLASSIFIER.lower().startswith("dfne"):
    classification_model = pw_classification.DFNE(device=DEVICE)
else:
    # default to AI4G Amazon Rainforest v2
    classification_model = pw_classification.AI4GAmazonRainforest(device=DEVICE, version='v2')

#%% Single-Image Detection + Classification

# Example paths. Change these to your data.
SINGLE_IMAGE_PATH = os.path.join(".", "demo_data", "imgs", "10050028_0.JPG")
SINGLE_OUT_DIR    = os.path.join(".", "demo_output")

# Create output directory if needed
os.makedirs(SINGLE_OUT_DIR, exist_ok=True)

assert os.path.exists(SINGLE_IMAGE_PATH), f"Image not found: {SINGLE_IMAGE_PATH}"

# Run detection
results = detection_model.single_image_detection(SINGLE_IMAGE_PATH)

# Optionally run classifier only on detections labeled 'animal' (class id 0 in MD categories)
input_img = np.array(Image.open(SINGLE_IMAGE_PATH).convert('RGB'))
from typing import List
clf_labels = []

for i, (xyxy, det_id) in enumerate(zip(results["detections"].xyxy, results["detections"].class_id)):
    if det_id == 0:  # animal
        cropped_image = sv.crop_image(image=input_img, xyxy=xyxy)
        results_clf = classification_model.single_image_classification(cropped_image)
        label = (
            f'{results_clf["prediction"]} {results_clf["confidence"]:.2f}'
            if results_clf["confidence"] > CLASSIFIER_CONF_THRESH
            else f'Unknown {results_clf["confidence"]:.2f}'
        )
        clf_labels.append(label)
    else:
        clf_labels.append(results["labels"][i])

results["labels"] = clf_labels

# Save annotated image & crops
pw_utils.save_detection_images(results, SINGLE_OUT_DIR, overwrite=False)
print(f"Saved annotated image(s) to: {SINGLE_OUT_DIR}")

if crop_detections:
    pw_utils.save_crop_images(results, CROP_OUT_DIR, overwrite=False)
    print(f"Saved crop(s) to: {CROP_OUT_DIR}")

#%% Batch Detection + Classification

# Example paths. Change these to your data.
BATCH_INPUT_DIR   = os.path.join(".", "demo_data", "classification_examples")
BATCH_DET_OUT_DIR = BATCH_INPUT_DIR.rstrip(os.sep) + "_outputs"

# Create output directory if needed
os.makedirs(BATCH_DET_OUT_DIR, exist_ok=True)

assert os.path.isdir(BATCH_INPUT_DIR), f"Folder not found: {BATCH_INPUT_DIR}"

# Batch detection
det_results = detection_model.batch_image_detection(BATCH_INPUT_DIR, batch_size=BATCH_SIZE)

# Batch classification (runs only for animal detections internally)
clf_results = classification_model.batch_image_classification(det_results=det_results)

# Merge detection + classification labels
merged_results = det_results.copy()
clf_conf_thres = CLASSIFIER_CONF_THRESH
clf_counter = 0

for det in merged_results:
    clf_labels = []
    for i, (xyxy, det_id) in enumerate(zip(det["detections"].xyxy, det["detections"].class_id)):
        if det_id == 0:
            pred = clf_results[clf_counter]["prediction"]
            conf = clf_results[clf_counter]["confidence"]
            label = f"{pred if conf > clf_conf_thres else 'Unknown'} {conf:.2f}"
            clf_labels.append(label)
            clf_counter += 1
        else:
            clf_labels.append(det["labels"][i])
    det["labels"] = clf_labels

# Save outputs
pw_utils.save_detection_images(merged_results, BATCH_DET_OUT_DIR, BATCH_INPUT_DIR, overwrite=False)
if crop_detections:
    pw_utils.save_crop_images(merged_results, CROP_OUT_DIR, BATCH_INPUT_DIR, overwrite=False)

json_out = os.path.join(BATCH_DET_OUT_DIR, "batch_output_classification.json")
json_out_timelapse = os.path.join(BATCH_DET_OUT_DIR, "batch_output_classification_timelapse.json")
csv_out  = os.path.join(BATCH_DET_OUT_DIR, "batch_output_classification.csv")

pw_utils.save_detection_classification_json(
    det_results=det_results,
    clf_results=clf_results,
    det_categories=detection_model.CLASS_NAMES,
    clf_categories=classification_model.CLASS_NAMES,
    output_path=json_out
)

pw_utils.save_detection_classification_timelapse_json(
    det_results=det_results,
    clf_results=clf_results,
    det_categories=detection_model.CLASS_NAMES,
    clf_categories=classification_model.CLASS_NAMES,
    output_path=json_out_timelapse
)

pw_utils.save_detection_classification_csv(
    det_results=det_results,
    clf_results=clf_results,
    det_categories=detection_model.CLASS_NAMES,
    clf_categories=classification_model.CLASS_NAMES,
    output_path=csv_out,
    model_name=MD_VERSION if not USE_MDV5 else "MDV5-a"
)

print("Batch outputs saved:")
print(" - Annotated images:", BATCH_DET_OUT_DIR)
if crop_detections:
    print(" - Crops:", CROP_OUT_DIR)
print(" - JSON:", json_out)
print(" - Timelapse JSON:", json_out_timelapse)
print(" - CSV:", csv_out)

#%% Positive/Negative Folder Separation

FOLDER_SEP_OUT = os.path.join(".", "folder_separation")
os.makedirs(FOLDER_SEP_OUT, exist_ok=True)

json_file = os.path.join(BATCH_DET_OUT_DIR, "batch_output_classification.json")
output_path = FOLDER_SEP_OUT
det_threshold = DET_THRESH_FOR_FOLDER
clf_threshold = CLF_THRESH_FOR_FOLDER
overwrite = True
draw_bboxes = True

assert os.path.isfile(json_file), f"JSON not found (run the batch cell first): {json_file}"

pw_utils.detection_classification_folder_separation(
    json_file,
    BATCH_INPUT_DIR,
    output_path,
    det_threshold,
    clf_threshold,
    overwrite,
    draw_bboxes
)
print(f"Separated images saved under: {output_path}")
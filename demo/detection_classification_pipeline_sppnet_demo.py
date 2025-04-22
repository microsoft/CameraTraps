#%% 
# Importing necessary basic libraries and modules
import os
import copy

# PyTorch imports 
import torch

#%% 
# Importing the model, dataset, transformations and utility functions from PytorchWildlife
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife import utils as pw_utils

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
classification_model = pw_classification.SpeciesNetTFInference(version='v4.0.0a', run_mode='multi_thread')

#%% Single image detection
# Specifying the path to the target image TODO: Allow argparsing
tgt_img_path = os.path.join("demo_data","imgs","10050028_0.JPG")

# Performing the detection on the single image
det_results = detection_model.single_image_detection(tgt_img_path)

# %%
clf_conf_thres = 0.8
clf_labels = []
for i in range(len(det_results['normalized_coords'])):
    r = {
        'img_id': tgt_img_path,
        'normalized_coords': [det_results['normalized_coords'][i]]
        }
    clf_results = classification_model.single_image_classification(tgt_img_path, det_results=r)[0]
    clf_labels.append("{} {:.2f}".format(clf_results["prediction"] if clf_results["confidence"] > clf_conf_thres else "Unknown",
                                         clf_results["confidence"]))
    
det_results["labels"] = clf_labels

# %%
# Saving the detection results 
pw_utils.save_detection_images(det_results, os.path.join(".","demo_output"), overwrite=False)

#%% Batch detection
""" Batch-detection demo """
# Specifying the folder path containing multiple images for batch detection
tgt_folder_path = os.path.join("demo_data","imgs")

# Performing batch detection on the images
det_results = detection_model.batch_image_detection(tgt_folder_path, batch_size=16)

# %%
clf_results = classification_model.batch_image_classification(tgt_folder_path, det_results=copy.deepcopy(det_results))

# %%
merged_results = det_results.copy()
clf_conf_thres = 0.8
clf_counter = 0

for det in merged_results:
    clf_labels = []

    for i, (xyxy, det_id) in enumerate(zip(det["detections"].xyxy, det["detections"].class_id)):
        if det_id == 0:
            clf_labels.append("{} {:.2f}".format(clf_results[clf_counter]["prediction"] if clf_results[clf_counter]["confidence"] > clf_conf_thres else "Unknown",
                                                 clf_results[clf_counter]["confidence"]))
        else:
            clf_labels.append(det["labels"][i])

        clf_counter += 1

    det["labels"] = clf_labels

#%% Output to annotated images
# Saving the batch detection results as annotated images
pw_utils.save_detection_images(merged_results, "batch_output", tgt_folder_path, overwrite=False)

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to run the SpeciesNet model.

Provides a command-line interface to execute the SpeciesNet model on various inputs. It
uses flags for specifying input, output, and run options, allowing the user to run the
model in different modes.
"""

# %%
import multiprocessing as mp

from speciesnet import SpeciesNet
from speciesnet.utils import prepare_instances_dict

#%% 
# Importing necessary basic libraries and modules
import os

# PyTorch imports 
import torch

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

# %%
from operator import itemgetter
def get_by_key(lst, key, value):
    return next(filter(lambda x: itemgetter(key)(x) == value, lst), None)

from PytorchWildlife.models.classification import BaseClassifierInference

class SpeciesNetTFInference(BaseClassifierInference):
    """
    Inference module for the PlainResNet Classifier.
    """
    def __init__(self, version='v4.0.0a', run_mode='multi_thread', geofence=True):
        super(SpeciesNetTFInference, self).__init__()

        self.model_url = 'kaggle:google/speciesnet/keras/{}'.format(version)
        self.run_mode = run_mode 
        self.geofence = geofence

        self.batch_size = 8
        self.progress_bars = True

        try:
            mp.set_start_method('spawn')
        except RuntimeError as e:
            if "context has already been set" in str(e):
                pass  # Context is already set, so skip silently
            else:
                raise

        self.model = SpeciesNet(
            self.model_url,
            components='classifier',
            geofence=self.geofence,
            # target_species_txt=target_species_txt,
            multiprocessing=(self.run_mode == "multi_process"),
        )

    def detections_dict_generation(self, det_results):
        detections_dict = {}

        for det in det_results:
            det['filepath'] = det['img_id']
            det['detections'] = [{'bbox' : [b[0], b[1], b[2] - b[0], b[3] - b[1],]}
                                 for b in det['normalized_coords']]
            detections_dict[det['filepath']] = det 

        return detections_dict

    def results_generation(self, predictions_dict, det_results):
        clf_results = []
        for pred in predictions_dict['predictions']:
            det = get_by_key(det_results, 'img_id', pred['filepath']) 
            for _ in range(len(det['normalized_coords'])):
                clf_results.append({
                    'img_id': pred['filepath'],
                    'prediction': pred['classifications']['classes'][0].split(';')[-1],
                    'confidence': pred['classifications']['scores'][0]
                })
        return clf_results

    def single_image_classification(self, file_path, det_results=None):

        instances_dict = prepare_instances_dict(
            filepaths=[file_path],
        )

        predictions_dict = self.model.classify(
            instances_dict=instances_dict,
            detections_dict=self.detections_dict_generation([det_results]) if det_results else None,
            run_mode=self.run_mode,
            batch_size=1,
            progress_bars=self.progress_bars,
        )
        return self.results_generation(predictions_dict, [det_results])

    def batch_image_classification(self, data_path, batch_size=8, det_results=None):

        instances_dict = prepare_instances_dict(
            folders=[data_path]
        )

        predictions_dict = self.model.classify(
            instances_dict=instances_dict,
            detections_dict=self.detections_dict_generation(det_results) if det_results else None,
            run_mode=self.run_mode,
            batch_size=batch_size,
            progress_bars=self.progress_bars,
        )

        return self.results_generation(predictions_dict, det_results)

#%% 
# Initializing the MegaDetectorV6 model for image detection
# Valid versions are MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e or MDV6-rtdetr-c
# detection_model = pw_detection.MegaDetectorV6(device=DEVICE, pretrained=True, version="MDV6-yolov10-e")

# Uncomment the following line to use MegaDetectorV5 instead of MegaDetectorV6
detection_model = pw_detection.MegaDetectorV5(device=DEVICE, pretrained=True, version="a")

# %%
classification_model = SpeciesNetTFInference(version='v4.0.0a', run_mode='multi_thread')

#%% Single image detection
# Specifying the path to the target image TODO: Allow argparsing
# tgt_img_path = os.path.join(".","demo_data","imgs","10050028_0.JPG")
tgt_img_path = os.path.join("./demo/demo_data/imgs/10050028_0.JPG")

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
# pw_utils.save_detection_images(results, os.path.join(".","demo_output"), overwrite=False)
pw_utils.save_detection_images(det_results, os.path.join("./demo/demo_output"), overwrite=False)

#%% Batch detection
""" Batch-detection demo """
import copy

# Specifying the folder path containing multiple images for batch detection
# tgt_folder_path = os.path.join(".","demo_data","imgs")
tgt_folder_path = "demo/demo_data/imgs"

# Performing batch detection on the images
det_results = detection_model.batch_image_detection(tgt_folder_path, batch_size=16)

# %%
clf_results = classification_model.batch_image_classification(tgt_folder_path, det_results=copy.deepcopy(det_results))

# %%
merged_results = det_results.copy()
clf_conf_thres = 0.8

for det in merged_results:
    clf_counter = 0
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
# %%

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Yolo mit base detector class. """

# Importing basic libraries

import os
import supervision as sv
import numpy as np
from PIL import Image
import wget
import torch

from ..base_detector import BaseDetector
from ....data import datasets as pw_data

import sys
from pathlib import Path

from lightning import Trainer
import yaml
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from yolo.tools.solver import InferenceModel
from yolo.utils.logging_utils import setup

class YOLOMITBase(BaseDetector):
    """
    Base detector class for YOLO MIT framework. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """
    def __init__(self, weights=None, device="cpu", url=None, transform=None):
        """
        Initialize the YOLO MIT detector.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        """
        self.transform = transform
        self.cfg = self._load_cfg()
        self.weights = weights
        self.device = device
        self.url = url
        super(YOLOMITBase, self).__init__(weights=self.weights, device=self.device, url=self.url)

    def _load_cfg(self):
        if self.MODEL_NAME == "MDV6-yolov9s-mit.ckpt":
            if not os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", "config_v9s.yaml")):
                os.makedirs(os.path.join(torch.hub.get_dir(), "checkpoints"), exist_ok=True)
                url = "https://zenodo.org/records/15178680/files/config_v9s.yaml?download=1"
                config_path = wget.download(url, out=os.path.join(torch.hub.get_dir(), "checkpoints"))
            else:
                config_path = os.path.join(torch.hub.get_dir(), "checkpoints", "config_v9s.yaml")
        elif self.MODEL_NAME == "MDV6-yolov9c-mit.ckpt":
            if not os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", "config_v9c.yaml")):
                os.makedirs(os.path.join(torch.hub.get_dir(), "checkpoints"), exist_ok=True)
                url = "https://zenodo.org/records/15178680/files/config_v9c.yaml?download=1"
                config_path = wget.download(url, out=os.path.join(torch.hub.get_dir(), "checkpoints"))
            else:
                config_path = os.path.join(torch.hub.get_dir(), "checkpoints", "config_v9c.yaml")

        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)

        return OmegaConf.create(cfg_dict)

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load the YOLO MIT model weights.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        Raises:
            Exception: If weights are not provided.
        """
        self.cfg.image_size = [self.IMAGE_SIZE, self.IMAGE_SIZE]
        callbacks, loggers, save_path = setup(self.cfg)

        self.trainer = Trainer(
            accelerator="auto",
            max_epochs=getattr(self.cfg.task, "epoch", None),
            precision="16-mixed",
            callbacks=callbacks,
            logger=loggers,
            log_every_n_steps=1,
            gradient_clip_val=10,
            gradient_clip_algorithm="value",
            deterministic=True,
            enable_progress_bar=not getattr(self.cfg, "quite", False),
            default_root_dir=save_path,
            devices=1, 
            num_nodes=1
        )

        self.model = InferenceModel(self.cfg)

        if weights:
            ckpt = torch.load(weights, map_location=torch.device('cpu'))  
            state_dict = {k: v for k, v in ckpt['state_dict'].items() if not k.startswith("ema.")}
            self.model.load_state_dict(state_dict)
        elif url:
            if not os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", self.MODEL_NAME)):
                os.makedirs(os.path.join(torch.hub.get_dir(), "checkpoints"), exist_ok=True)
                weights = wget.download(url, out=os.path.join(torch.hub.get_dir(), "checkpoints"))
            else:
                weights = os.path.join(torch.hub.get_dir(), "checkpoints", self.MODEL_NAME)
            ckpt = torch.load(weights, map_location=torch.device('cpu'))
            state_dict = {k: v for k, v in ckpt['state_dict'].items() if not k.startswith("ema.")}
            self.model.load_state_dict(state_dict)
        else:
            raise Exception("Need weights for inference.")

        results = self.trainer.predict(self.model, return_predictions=True)
        return results

    def results_generation(self, preds, img_id, id_strip=None):
        """
        Generate results for detection based on model predictions.
        
        Args:
            preds (List[torch.Tensor]): 
                Model predictions.
            img_id (str): 
                Image identifier.
            id_strip (str, optional): 
                Strip specific characters from img_id. Defaults to None.

        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        class_id = preds[0][:,0].cpu().numpy().astype(int)
        xyxy = preds[0][:,1:5].cpu().numpy()
        confidence = preds[0][:,5].numpy()

        results = {"img_id": str(img_id).strip(id_strip)}
        results["detections"] = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        results["labels"] = [
            f"{self.CLASS_NAMES[class_id]} {confidence:0.2f}"  
            for _, _, confidence, class_id, _, _ in results["detections"] 
        ]
        
        return results
        

    def single_image_detection(self, img_path, det_conf_thres=0.2, id_strip=None):
        """
        Perform detection on a single image.
        
        Args:
            img_path (str, optional): 
                Image path.
            det_conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """
        self.cfg.task.data.source = img_path
        self.cfg.task.nms.min_confidence = det_conf_thres
        det_results = self._load_model(weights=self.weights, device=self.device, url=self.url)
        return self.results_generation(det_results[0][2], img_path, id_strip)

    def batch_image_detection(self, data_path, batch_size=16, det_conf_thres=0.2, id_strip=None):
        """
        Perform detection on a batch of images.
        
        Args:
            data_path (str): 
                Path containing all images for inference.
            batch_size (int, optional):
                Batch size for inference. Defaults to 16.
            det_conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.
            extension (str, optional):
                Image extension to search for. Defaults to "JPG"

        Returns:
            list: List of detection results for all images.
        """
        self.cfg.task.data.source = data_path
        self.cfg.task.nms.min_confidence = det_conf_thres
        det_results = self._load_model(weights=self.weights, device=self.device, url=self.url)
        
        dataset = pw_data.DetectionImageFolder(
            data_path,
            transform=self.transform,
        )
        
        results = []
        for i in range(len(det_results)):
            res = self.results_generation(det_results[i][2], dataset.images[i], id_strip)
            # Upload the original image and get the size in the format (height, width)
            img = Image.open(dataset.images[i])
            img = np.asarray(img)
            size = img.shape[:2]
            # Normalize the coordinates for timelapse compatibility
            normalized_coords = [[x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]] for x1, y1, x2, y2 in res["detections"].xyxy]
            res["normalized_coords"] = normalized_coords
            results.append(res)

        return results




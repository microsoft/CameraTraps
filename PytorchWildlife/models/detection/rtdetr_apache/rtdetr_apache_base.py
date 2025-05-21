# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" RT-DETR Apache base detector class. """

# Importing basic libraries
import os
import supervision as sv
import wget
import torch
import torch.nn as nn 
import torchvision.transforms as T

from ..base_detector import BaseDetector
from ....data import datasets as pw_data
from PIL import Image

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
from rtdetrv2_pytorch.src.core import YAMLConfig

class RTDETRApacheBase(BaseDetector):
    """
    Base detector class for RTDETRApacheBase framework. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """
    def __init__(self, weights=None, device="cpu", url=None):
        """
        Initialize the RT-DETR apache detector.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        """
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        self.weights = weights
        self.device = device
        self.url = url
        super(RTDETRApacheBase, self).__init__(weights=self.weights, device=self.device, url=self.url)
        self._load_model(weights=self.weights, device=self.device, url=self.url)

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load the RT-DETR apache model weights.
        
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
        if weights:
            resume = weights
        elif url:
            if not os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", self.MODEL_NAME)):
                os.makedirs(os.path.join(torch.hub.get_dir(), "checkpoints"), exist_ok=True)
                resume = wget.download(url, out=os.path.join(torch.hub.get_dir(), "checkpoints"))
            else:
                resume = os.path.join(torch.hub.get_dir(), "checkpoints", self.MODEL_NAME)
        else:
            raise Exception("Need weights for inference.")

        if self.MODEL_NAME == "MDV6-apa-rtdetr-c.pth":
            config = os.path.join(project_root, "rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_megadetector.yml")
        elif self.MODEL_NAME == "MDV6-apa-rtdetr-e.pth":
            config = os.path.join(project_root, "rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r101vd_6x_megadetector.yml")
        else:
            raise ValueError('Select a valid model version: MDV6-apa-rtdetr-c or MDV6-apa-rtdetr-e')
        
        cfg = YAMLConfig(config, resume=resume)
        
        checkpoint = torch.load(resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self, ) -> None:
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
                
            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs
        
        self.model = Model().to(self.device)

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
        class_id = preds[0].cpu().numpy().astype(int)
        xyxy = preds[1].detach().cpu().numpy()
        confidence = preds[2].detach().cpu().numpy()

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
        

    def single_image_detection(self, img, img_path=None, det_conf_thres=0.2, id_strip=None):
        """
        Perform detection on a single image.
        
        Args:
            img (str or ndarray): 
                Image path or ndarray of images.
            img_path (str, optional): 
                Image path or identifier.
            det_conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """
        if type(img) == str:
            if img_path is None:
                img_path = img
            im_pil = Image.open(img_path).convert('RGB')
        else:
            im_pil = Image.fromarray(img)

        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(self.device)
        im_data = self.transform(im_pil)[None].to(self.device)
        labels, boxes, scores = self.model(im_data, orig_size)

        scr = scores[0]
        lab = labels[0][scr > det_conf_thres]
        box = boxes[0][scr > det_conf_thres]
        scrs = scores[0][scr > det_conf_thres]
        
        return self.results_generation([lab, box, scrs], img_path, id_strip)

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
        dataset = pw_data.DetectionImageFolder(
            data_path,
            transform=self.transform,
        )
        
        results = []
        for i in range(len(dataset)):
            im_pil = Image.open(dataset.images[i]).convert('RGB')
            w, h = im_pil.size
            orig_size = torch.tensor([w, h])[None].to(self.device)
            im_data = self.transform(im_pil)[None].to(self.device)

            labels, boxes, scores = self.model(im_data, orig_size)

            scr = scores[0]
            lab = labels[0][scr > det_conf_thres]
            box = boxes[0][scr > det_conf_thres]
            scrs = scores[0][scr > det_conf_thres]
            
            res = self.results_generation([lab, box, scrs], dataset.images[i], id_strip)

            # Normalize the coordinates for timelapse compatibility
            size = orig_size[0].cpu().numpy()
            normalized_coords = [[x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]] for x1, y1, x2, y2 in res["detections"].xyxy]
            res["normalized_coords"] = normalized_coords
            results.append(res)

        return results




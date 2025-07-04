# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" YoloV8 base detector class. """

# Importing basic libraries

import os
import wget
import numpy as np
from tqdm import tqdm
from PIL import Image
import supervision as sv

import torch
from torch.utils.data import DataLoader

from ultralytics.models import yolo, rtdetr

from ..base_detector import BaseDetector
from ....data import transforms as pw_trans
from ....data import datasets as pw_data


class YOLOV8Base(BaseDetector):
    """
    Base detector class for the new ultralytics YOLOV8 framework. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    This base detector class is also compatible with all the new ultralytics models including YOLOV9, 
    RTDetr, and more.
    """
    def __init__(self, weights=None, device="cpu", url=None, transform=None):
        """
        Initialize the YOLOV8 detector.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        """
        super(YOLOV8Base, self).__init__(weights=weights, device=device, url=url)
        self.transform = transform
        self._load_model(weights, self.device, url)

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load the YOLOV8 model weights.
        
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

        if self.MODEL_NAME == 'MDV6b-rtdetrl.pt':
            self.predictor = rtdetr.RTDETRPredictor()
        else:
            self.predictor = yolo.detect.DetectionPredictor()
        # self.predictor.args.device = device # Will uncomment later
        self.predictor.args.imgsz = self.IMAGE_SIZE
        self.predictor.args.save = False # Will see if we want to use ultralytics native inference saving functions.

        if weights:
            self.predictor.setup_model(weights)
        elif url:
            if not os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", self.MODEL_NAME)):
                os.makedirs(os.path.join(torch.hub.get_dir(), "checkpoints"), exist_ok=True)
                weights = wget.download(url, out=os.path.join(torch.hub.get_dir(), "checkpoints"))
            else:
                weights = os.path.join(torch.hub.get_dir(), "checkpoints", self.MODEL_NAME)
            self.predictor.setup_model(weights)
        else:
            raise Exception("Need weights for inference.")
        
        if not self.transform:
            self.transform = pw_trans.MegaDetector_v5_Transform(target_size=self.IMAGE_SIZE,
                                                                stride=self.STRIDE)

    def results_generation(self, preds, img_id, id_strip=None) -> dict:
        """
        Generate results for detection based on model predictions.
        
        Args:
            preds (ultralytics.engine.results.Results): 
                Model predictions.
            img_id (str): 
                Image identifier.
            id_strip (str, optional): 
                Strip specific characters from img_id. Defaults to None.

        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        xyxy = preds.boxes.xyxy.cpu().numpy()
        confidence = preds.boxes.conf.cpu().numpy()
        class_id = preds.boxes.cls.cpu().numpy().astype(int)

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
        

    def single_image_detection(self, img, img_path=None, det_conf_thres=0.2, id_strip=None) -> dict:
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
            img = np.array(Image.open(img_path).convert("RGB"))
        img_size = img.shape

        self.predictor.args.batch = 1
        self.predictor.args.conf = det_conf_thres
        
        det_results = list(self.predictor.stream_inference([img]))

        res = self.results_generation(det_results[0], img_path, id_strip)

        normalized_coords = [[x1 / img_size[1], y1 / img_size[0], x2 / img_size[1], y2 / img_size[0]] 
                             for x1, y1, x2, y2 in res["detections"].xyxy]
        res["normalized_coords"] = normalized_coords
        
        return res

    def batch_image_detection(self, data_source, batch_size: int = 16, det_conf_thres: float = 0.2, id_strip: str = None) -> list[dict]:
        """
        Perform detection on a batch of images.

        Args:
            data_source (str or List[np.ndarray]): Either path containing images for inference or list of numpy arrays (RGB format, shape: H×W×3).
            batch_size (int, optional): Batch size for inference. Defaults to 16.
            det_conf_thres (float, optional): Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): Characters to strip from img_id. Defaults to None.

        Returns:
            list[dict]: List of detection results for all images.
        """
        self.predictor.args.batch = batch_size
        self.predictor.args.conf = det_conf_thres

        # Handle numpy array input
        if isinstance(data_source, (list, np.ndarray)):
            results = []
            num_batches = (len(data_source) + batch_size - 1) // batch_size  # Calculate total batches
            
            with tqdm(total=num_batches) as pbar:
                for start_idx in range(0, len(data_source), batch_size):
                    batch_arrays = data_source[start_idx:start_idx + batch_size]
                    det_results = self.predictor.stream_inference(batch_arrays)
                    
                    for idx, preds in enumerate(det_results):
                        res = self.results_generation(preds, f"{start_idx + idx}", id_strip)
                        # Get size directly from numpy array
                        img_height, img_width = batch_arrays[idx].shape[:2]
                        normalized_coords = [[x1/img_width, y1/img_height, x2/img_width, y2/img_height] 
                                        for x1, y1, x2, y2 in res["detections"].xyxy]
                        res["normalized_coords"] = normalized_coords
                        results.append(res)
                    pbar.update(1)
            return results
        
        # Handle image directory input
        dataset = pw_data.DetectionImageFolder(
            data_source,
            transform=self.transform,
        )

        # Creating a DataLoader for batching and parallel processing of the images
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=0, drop_last=False
                            )
        
        results = []
        with tqdm(total=len(loader)) as pbar:
            for batch_index, (imgs, paths, sizes) in enumerate(loader):
                det_results = self.predictor.stream_inference(paths)
                batch_results = []
                for idx, preds in enumerate(det_results):
                    res = self.results_generation(preds, paths[idx], id_strip)
                    size = preds.orig_shape
                    # Normalize the coordinates for timelapse compatibility
                    normalized_coords = [[x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]] for x1, y1, x2, y2 in res["detections"].xyxy]
                    res["normalized_coords"] = normalized_coords
                    results.append(res)
                pbar.update(1)
                results.extend(batch_results)
        return results

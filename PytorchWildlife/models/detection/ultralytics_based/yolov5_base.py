# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" YoloV5 base detector class. """

# Importing basic libraries

import numpy as np
from tqdm import tqdm
from PIL import Image
import supervision as sv

import torch
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url

from yolov5.utils.general import non_max_suppression, scale_coords

from ..base_detector import BaseDetector
from ....data import transforms as pw_trans
from ....data import datasets as pw_data


class YOLOV5Base(BaseDetector):
    """
    Base detector class for YOLO V5. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """
    def __init__(self, weights=None, device="cpu", url=None, transform=None):
        """
        Initialize the YOLO V5 detector.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
            transform (callable, optional):
                Optional transform to be applied on the image. Defaults to None.
        """
        self.transform = transform
        super(YOLOV5Base, self).__init__(weights=weights, device=device, url=url)
        self._load_model(weights, device, url)

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load the YOLO V5 model weights.
        
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
            checkpoint = torch.load(weights, map_location=torch.device(device))
        elif url:
            checkpoint = load_state_dict_from_url(url, map_location=torch.device(self.device))
        else:
            raise Exception("Need weights for inference.")
        self.model = checkpoint["model"].float().fuse().eval().to(self.device)
        
        if not self.transform:
            self.transform = pw_trans.MegaDetector_v5_Transform(target_size=self.IMAGE_SIZE,
                                                                stride=self.STRIDE)

    def results_generation(self, preds, img_id, id_strip=None):
        """
        Generate results for detection based on model predictions.
        
        Args:
            preds (numpy.ndarray): 
                Model predictions.
            img_id (str): 
                Image identifier.
            id_strip (str, optional): 
                Strip specific characters from img_id. Defaults to None.

        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        results = {"img_id": str(img_id).strip(id_strip)}
        results["detections"] = sv.Detections(
            xyxy=preds[:, :4],
            confidence=preds[:, 4],
            class_id=preds[:, 5].astype(int)
        )
        results["labels"] = [
            f"{self.CLASS_NAMES[class_id]} {confidence:0.2f}"
            for confidence, class_id in zip(results["detections"].confidence, results["detections"].class_id)
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
            img = np.array(Image.open(img_path).convert("RGB"))
        img_size = img.shape
        img = self.transform(img)

        if img_size is None:
            img_size = img.permute((1, 2, 0)).shape # We need hwc instead of chw for coord scaling
        preds = self.model(img.unsqueeze(0).to(self.device))[0]
        preds = torch.cat(non_max_suppression(prediction=preds, det_conf_thres=det_conf_thres), axis=0)
        preds[:, :4] = scale_coords([self.IMAGE_SIZE] * 2, preds[:, :4], img_size).round()
        return self.results_generation(preds.cpu().numpy(), img_path, id_strip)

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

        # Creating a DataLoader for batching and parallel processing of the images
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=0, drop_last=False)

        results = []
        with tqdm(total=len(loader)) as pbar:
            for batch_index, (imgs, paths, sizes) in enumerate(loader):
                imgs = imgs.to(self.device)
                predictions = self.model(imgs)[0].detach().cpu()
                predictions = non_max_suppression(predictions, det_conf_thres=det_conf_thres)

                batch_results = []
                for i, pred in enumerate(predictions):
                    if pred.size(0) == 0:  
                        continue
                    pred = pred.numpy()
                    size = sizes[i].numpy()
                    path = paths[i]
                    original_coords = pred[:, :4].copy()
                    pred[:, :4] = scale_coords([self.IMAGE_SIZE] * 2, pred[:, :4], size).round()
                    # Normalize the coordinates for timelapse compatibility
                    normalized_coords = [[x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]] for x1, y1, x2, y2 in pred[:, :4]]
                    res = self.results_generation(pred, path, id_strip)
                    res["normalized_coords"] = normalized_coords
                    batch_results.append(res)
                pbar.update(1)
                results.extend(batch_results)
            return results

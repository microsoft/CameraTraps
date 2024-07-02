# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" YoloV8 base detector class. """

# Importing basic libraries

import os
from glob import glob
import supervision as sv

from ultralytics.models import yolo

from ..base_detector import BaseDetector


class YOLOV8Base(BaseDetector):
    """
    Base detector class for the new ultralytics YOLOV8 framework. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    This base detector class is also compatible with all the new ultralytics models including YOLOV9, 
    RTDetr, and more.
    """
    def __init__(self, weights=None, device="cpu", url=None):
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

        self.predictor = yolo.detect.DetectionPredictor()
        # self.predictor.args.device = device # Will uncomment later
        self.predictor.args.imgsz = self.IMAGE_SIZE
        self.predictor.args.save = False # Will see if we want to use ultralytics native inference saving functions.

        if weights:
            self.predictor.setup_model(weights)
        elif url:
            raise Exception("URL weights loading not ready for beta testing.")
        else:
            raise Exception("Need weights for inference.")

    def results_generation(self, preds, img_id, id_strip=None):
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
        

    def single_image_detection(self, img_path=None, img_id=None, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a single image.
        
        Args:
            img_path (str): 
                Image path or identifier.
            conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """
        self.predictor.args.batch = 1
        self.predictor.args.conf = conf_thres
        self.predictor.args.verbose = False
        det_results = list(self.predictor.stream_inference([img_path]))

        if img_id is None:
            if type(img_path) == str: #image path
                img_id = img_path
            else: #numpy array
                raise NameError("Please input an img_id.")

        return self.results_generation(det_results[0], img_id, id_strip)

    def batch_image_detection(self, data_path, batch_size=16, conf_thres=0.2, id_strip=None, extension="JPG"):
        """
        Perform detection on a batch of images.
        
        Args:
            data_path (str): 
                Path containing all images for inference.
            batch_size (int, optional):
                Batch size for inference. Defaults to 16.
            conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.
            extension (str, optional):
                Image extension to search for. Defaults to "JPG"

        Returns:
            list: List of detection results for all images.
        """
        self.predictor.args.batch = batch_size
        self.predictor.args.conf = conf_thres
        img_list = glob(os.path.join(data_path, '**/*.{}'.format(extension)), recursive=True)
        det_results = list(self.predictor.stream_inference(img_list))
        results = []
        for idx, preds in enumerate(det_results):
            results.append(self.results_generation(preds, img_list[idx], id_strip))
        return results

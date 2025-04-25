""""
YoloV8 base detector class.
Modified to support PyTorch DDP framework
"""


import os
import time
from glob import glob
import supervision as sv
import numpy as np
import pandas as pd
from PIL import Image
import wget
import torch

from ultralytics.models import yolo, rtdetr
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..base_detector import BaseDetector
from ....data import transforms as pw_trans
from ....data import datasets as pw_data

class YOLOV8_Distributed(BaseDetector):
    """
    Distributed YoloV8 detector class.
    This class provides utility methods for loading the model, generating results,
    and performing batch image detections.
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
        self.transform = transform
        super(YOLOV8_Distributed, self).__init__(weights=weights, device=device, url=url)
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
        # results["detections"] = sv.Detections(
        #     xyxy=xyxy,
        #     confidence=confidence,
        #     class_id=class_id
        # )
        results["detections_xyxy"] = xyxy
        results["detections_confidence"] = confidence
        results["detections_class_id"] = class_id

        # results["labels"] = [
        #     f"{self.CLASS_NAMES[class_id]} {confidence:0.2f}"  
        #     for _, _, confidence, class_id, _, _ in results["detections"] 
        # ]
        
        results["labels"] = [
            f"{self.CLASS_NAMES[cls_id]} {conf:0.2f}"  
            for cls_id, conf in zip(class_id, confidence)
        ]
        
        results["n_animal_detected"] = np.sum(class_id == 0)
        
        return results
    
    def batch_image_detection(self, loader, batch_size, global_rank, local_rank, output_dir, det_conf_thres=0.2, checkpoint_frequency = 1000):

        """
        Perform batch image detection using the YOLOV8 model.
        
        Args:
            loader (torch.utils.data.DataLoader): 
                DataLoader for input images.
            batch_size (int):
                Size of the batch for detection.
            global_rank (int): 
                Global rank of the process.
            local_rank (int): 
                Local rank of the process.
            output_dir (str): 
                Directory to save detection results.
            det_conf_thres (float, optional): 
                Confidence threshold for detections. Defaults to 0.2.
            checkpoint_frequency (int, optional): 
                Frequency of saving intermediate results. Defaults to 1000.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.predictor.args.batch = batch_size
        self.predictor.args.conf = det_conf_thres
        self.predictor.args.device = local_rank

        
        # Create checkpoint directory
        # Track batches and processed items
        results = {
            "img_id": [],
            "detections_xyxy": [],
            "detections_confidence": [],
            "detections_class_id": [],
            "labels": [],
            "n_animal_detected": [],
            "normalized_coords": []
        }

        checkpoint_dir = os.path.join(output_dir, f"checkpoints_rank{global_rank}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        batch_counter = 0
        processed_count = 0
        start_time = time.time()

        for uuids, images in loader:
            batch_counter += 1
            processed_count += len(images)
            # images: tensor of shape [batch_size, 3, H, W]
            # Assuming images are transformed & Standardized
            det_results = self.predictor.stream_inference(images)
            
            for idx, preds in enumerate(det_results):
                res = self.results_generation(preds, uuids[idx])
                
                size = preds.orig_shape
                normalized_coords = [[x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]] for x1, y1, x2, y2 in res["detections_xyxy"]]
                res["normalized_coords"] = normalized_coords
                
                #results.append(res)
                results["img_id"].append(res["img_id"])
                results["detections_xyxy"].append(res["detections_xyxy"].tolist())
                results["detections_confidence"].append(res["detections_confidence"].tolist())
                results["detections_class_id"].append(res["detections_class_id"].tolist())
                results["labels"].append(res["labels"])
                results["n_animal_detected"].append(int(res["n_animal_detected"]))
                results["normalized_coords"].append(res["normalized_coords"])
                
            if batch_counter % checkpoint_frequency == 0:
                elapsed = time.time() - start_time
                print(f"[Rank {global_rank}] Processed {processed_count} images in {elapsed}")
                
                # Save intermediate results
                checkpoint_path = os.path.join(
                    checkpoint_dir, 
                    f"checkpoint_{batch_counter:06d}.parquet"
                )
                
                df = pd.DataFrame({
                    "img_id": results["img_id"],
                    "n_animal_detected": results["n_animal_detected"]
                })
                df.to_parquet(checkpoint_path, index=False)
                print(f"[Rank {global_rank}] Saved checkpoint to {checkpoint_path}")
            
        # Save results to disk
        os.makedirs(output_dir, exist_ok=True)
        df = pd.DataFrame({
                    "img_id": results["img_id"],
                    "n_animal_detected": results["n_animal_detected"]
                })
        out_path = os.path.join(output_dir, f"predictions_rank{global_rank}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[rank {global_rank}] Saved predictions to {out_path}")
        
        return results





        
    
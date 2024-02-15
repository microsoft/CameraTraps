from abc import ABC, abstractmethod
import torch
from torch.hub import load_state_dict_from_url
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from typing import OrderedDict
import numpy as np
from tqdm import tqdm
import supervision as sv

# Import model and utility functions
from yolov5.utils.general import non_max_suppression, scale_coords


class BaseDetector(ABC):
    """
    Base class for object detection models, providing a unified interface for different detectors.
    """

    def __init__(self, weights=None, device="cpu", url=None):
        """
        Initializes the detector with the specified weights and computation device.

        Args:
            weights (str, optional): Path to the model weights file.
            device (str, optional): Computation device ('cpu' or 'cuda').
            url (str, optional): URL to download model weights.
        """
        self.device = device
        self.weights = weights
        self.url = url
        self.model = None
        self._load_model()  # Load the model
        self.model.to(self.device)

    def _load_model(self):
        """
        Generic model loading method for loading weights from a local file or URL.
        """
        if self.weights:
            checkpoint = torch.load(self.weights, map_location=self.device)
        elif self.url:
            checkpoint = load_state_dict_from_url(
                self.url, map_location=self.device)
        else:
            raise ValueError(
                "Weights or URL must be provided for model loading.")
        return self.process_checkpoint(checkpoint)

    @abstractmethod
    def process_checkpoint(self, checkpoint):
        """
        Process the loaded checkpoint specific to the model. Subclasses should override this method.
        Args:
            checkpoint: The loaded checkpoint data.
        
        Returns:
            A model instance with loaded weights
        """
        pass

    def results_generation(self, preds, img_id, id_strip=None):
        """
        Generates a dictionary of detection results from model predictions.

        Args:
            preds (numpy.ndarray): Model predictions.
            img_id (str): Identifier for the input image.
            id_strip (str, optional): Characters to strip from the image identifier.

        Returns:
            dict: Formatted detection results including image ID, detections, and labels.
        """
        img_id_processed = str(img_id).strip(
            id_strip) if id_strip else str(img_id)
        detections = sv.Detections(
            xyxy=preds[:, :4],
            confidence=preds[:, 4],
            class_id=preds[:, 5].astype(int)
        )
        labels = [f"{self.CLASS_NAMES[class_id]} {confidence:.2f}"
                  for _, _, confidence, class_id, _ in detections]
        results = {"img_id": img_id_processed,
                   "detections": detections, "labels": labels}
        return results

    @abstractmethod
    def single_image_detection(self, img, img_size=None, img_path=None, conf_thres=0.2, id_strip=None):
        """
        Detects objects in a single image. This method should be overridden by subclasses.

        Args:
            img (torch.Tensor or any image representation): Input image.
            img_size (tuple, optional): Original size of the input image.
            img_path (str, optional): Path or identifier of the input image.
            conf_thres (float, optional): Confidence threshold for detections.
            id_strip (str, optional): Characters to strip from the image identifier.

        Returns:
            dict: Detection results.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    @abstractmethod
    def batch_image_detection(self, dataloader, conf_thres=0.2, id_strip=None):
        """
        Detects objects in a batch of images. This method should be overridden by subclasses.

        Args:
            dataloader (DataLoader): DataLoader containing batches of images.
            conf_thres (float, optional): Confidence threshold for detections.
            id_strip (str, optional): Characters to strip from image identifiers.

        Returns:
            list: A list of detection results for all images in the batch.
        """
        raise NotImplementedError("Subclass must implement abstract method")


class YOLOv5Detector(BaseDetector):

    def process_checkpoint(self, checkpoint):
        """
        YOLOv5 specific checkpoint processing.
        """
        model = checkpoint["model"].float().fuse().eval()
        return model

    def single_image_detection(self, img, img_size=None, img_path=None, conf_thres=0.2, id_strip=None):
        """
        Performs object detection on a single image using YOLOv5.

        Args are as defined in the BaseDetector class.
        """
        if img_size is None:
            # HWC format for coordinate scaling
            img_size = img.permute((1, 2, 0)).shape
        preds = self.model(img.unsqueeze(0).to(self.device))[0]
        preds = torch.cat(non_max_suppression(
            preds, conf_thres=conf_thres), axis=0)
        preds[:, :4] = scale_coords(
            img_size, preds[:, :4], img.shape[-2:]).round()
        return self.results_generation(preds.cpu().numpy(), img_path, id_strip)

    def batch_image_detection(self, dataloader, conf_thres=0.2, id_strip=None):
        """
        Performs object detection on a batch of images using YOLOv5.

        Args are as defined in the BaseDetector class.
        """
        results = []
        with tqdm(total=len(dataloader), desc="Processing batches") as pbar:
            for imgs, paths, sizes in dataloader:
                imgs = imgs.to(self.device)
                preds = self.model(imgs)[0]
                preds = non_max_suppression(preds, conf_thres=conf_thres)

                for i, pred in enumerate(preds):
                    if pred is not None and len(pred):
                        pred[:, :4] = scale_coords(
                            imgs[i].shape[1:], pred[:, :4], sizes[i]).round()
                        results.extend([self.results_generation(
                            pred.cpu().numpy(), path, id_strip) for path in paths])
                pbar.update(1)
        return results


class HerdNetDetector(BaseDetector):

    def process_checkpoint(self, checkpoint):
        """
        Loads HerdNet model weights either from a local file or a URL.
        """
        from .herdnet.model import HerdNet

        classes = checkpoint['classes']
        self.num_classes = len(classes) + 1

        self.model = HerdNet(num_classes=self.num_classes,
                             down_ratio=self.DOWN_RATIO)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        # Remove the 'model.' prefix
        for k, v in state_dict.items():
            name = k[6:]
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.float().eval()

    def single_image_detection(self, img, img_size=None, img_path=None, conf_thres=0.2, id_strip=None):
        """
        Performs object detection on a single image using HerdNet.

        Args are as defined in the BaseDetector class.
        """
        from .herdnet import utilities

        # Convert input image to tensor if not already
        if isinstance(img, torch.Tensor):
            img_tensor = img.to(self.device)
        else:
            img_tensor = ToTensor()(img).to(self.device)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        self.model.eval()

        # Create patches from the input image
        patches, (ncol, nrow) = utilities.make_patches(
            img_tensor[0], self.PATCHSIZE, self.OVERLAP)

        # Perform inference on the patches to generate maps
        maps = utilities.infer_patches(self, patches)

        # Reassemble the maps into the original image's coordinate system
        patched_map = utilities.patch_maps(
            img_tensor[0], maps, self.DOWN_RATIO, self.PATCHSIZE, self.OVERLAP, ncol, nrow)
        patched_map = utilities.reduce(
            self, img_tensor[0], self.DOWN_RATIO, self.PATCHSIZE, self.OVERLAP, patched_map)

        # Upsample the patched map back to the original image resolution
        patched_map = F.interpolate(
            patched_map, scale_factor=self.DOWN_RATIO, mode='bilinear', align_corners=True)

        # Split the patched map into heatmap and class map
        heatmap = patched_map[:, :1, :, :]
        clsmap = patched_map[:, 1:, :, :]

        # Perform Local Maxima Detection Strategy (LDMS) on the maps to extract detections
        counts, locs, labels, scores, dscores = utilities.process_maps(
            heatmap, clsmap, kernel_size=(3, 3), adapt_ts=100.0/255.0, neg_ts=0.1)

        # Format the detections into a structured array
        width, height = 25, 25  # Example dimensions, adjust as necessary
        xyxy_locs = [[x - width // 2, y - height // 2, x +
                      width // 2, y + height // 2] for x, y in locs[0]]

        locs_np = np.array(xyxy_locs)
        dscores_np = np.array(dscores[0]).reshape(-1, 1)
        labels_np = np.array(labels[0]).astype(int).reshape(-1, 1)

        # Concatenate locations, detection scores, and labels
        preds = np.concatenate([locs_np, dscores_np, labels_np], axis=1)

        # Generate and return the final results
        return self.results_generation(preds, img_path, id_strip)

    def batch_image_detection(self, dataloader, conf_thres=0.2, id_strip=None):
        """
        Performs object detection on a batch of images using HerdNet.

        Args are as defined in the BaseDetector class.
        """
        batch_results = []
        with tqdm(total=len(dataloader)) as pbar:
            for imgs, paths, sizes in dataloader:
                for img, path, size in zip(imgs, paths, sizes):
                    result = self.single_image_detection(
                        img, img_path=path, conf_thres=conf_thres, id_strip=id_strip)
                    batch_results.append(result)
                    pbar.update(1)
        return batch_results

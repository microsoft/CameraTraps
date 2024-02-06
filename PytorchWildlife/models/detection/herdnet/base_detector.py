# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Herdnet base detector class. """

# Importing basic libraries

from typing import OrderedDict
import numpy as np
from tqdm import tqdm
import supervision as sv
import torch
from torch.hub import load_state_dict_from_url
from .model import HerdNet

import torchvision.transforms
import torch.nn.functional as F
from .utilities import make_patches, infer_patches, patch_maps, reduce, process_maps


class HerdNetBase:
    """
    Base detector class for HerdNet. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """

    # Placeholder class-level attributes to be defined in derived classes
    CLASS_NAMES = None
    TRANSFORM = None
    NUM_CLASSES = None
    DOWN_RATIO = None
    IMAGE_SIZE = None
    STRIDE = None

    def __init__(self, weights=None, device="cpu", url=None):
        """
        Initialize the Herdnet V5 detector.

        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        """
        self.model = None
        self.device = device
        self._load_model(weights, self.device, url)
        self.model.to(self.device)

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load the HerdNet model weights.

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
            checkpoint = load_state_dict_from_url(
                url, map_location=torch.device(self.device))
        else:
            raise Exception("Need weights for inference.")

        classes = checkpoint['classes']
        self.num_classes = len(classes) + 1

        self.model = HerdNet(num_classes=self.num_classes,
                             down_ratio=self.DOWN_RATIO)

        # Load the state dictionary
        state_dict = checkpoint['model_state_dict']

        # Adjust the keys
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[6:]  # Remove the 'model.' prefix
            new_state_dict[name] = v

        # Load the adjusted state dict
        self.model.load_state_dict(new_state_dict)

        # Convert the model to FP32 and switch to eval mode
        self.model = self.model.float().eval()

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
            for _, _, confidence, class_id, _ in results["detections"]
        ]
        return results

    def single_image_detection(self, img,  img_size=None, img_path=None, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a single image.

        Args:
            img (torch.Tensor): 
                Input image tensor.
            img_size (tuple): 
                Original image size.
            img_path (str): 
                Image path or identifier.
            conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """
        #Load and preprocess the image
        if isinstance(img, torch.Tensor):
            img_tensor = img.unsqueeze(0).to(self.device)
        else:
            img_tensor = torchvision.transforms.ToTensor()(img)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

        self.model.eval()

        # Create patches
        patches, (ncol, nrow) = make_patches(
            img_tensor[0], self.PATCHSIZE, self.OVERLAP)

        # Inference to get maps
        maps = infer_patches(self, patches)

        # Patch the maps into initial coordinates system
        patched_map = patch_maps(
            img_tensor[0], maps, self.DOWN_RATIO, self.PATCHSIZE, self.OVERLAP, ncol, nrow)
        patched_map = reduce(
            self, img_tensor[0], self.DOWN_RATIO, self.PATCHSIZE, self.OVERLAP, patched_map)

        # Upsample
        patched_map = F.interpolate(
            patched_map, scale_factor=self.DOWN_RATIO, mode='bilinear', align_corners=True)

        # Split into heatmap and clsmap
        heatmap = patched_map[:, :1, :, :]
        clsmap = patched_map[:, 1:, :, :]

        #Perform Local Maxima Detection Strategy (LDMS)
        counts, locs, labels, scores, dscores = process_maps(heatmap, clsmap, kernel_size=(
            3, 3), adapt_ts=100.0/255.0, neg_ts=0.1) 

        # Format predictions
        # formatted_locs = [[x, y, x, y] for x, y in locs[0]]
        width, height = 25, 25
        xyxy_locs = []
        for a, b in locs[0]:
            x1 = a - width // 2
            y1 = b - height // 2
            x2 = a + width // 2
            y2 = b + height // 2
            xyxy_locs.append([x1, y1, x2, y2])

        locs_np = np.array(xyxy_locs)
        dscores_np = np.array(dscores[0]).reshape(-1, 1)
        labels_np = np.array(labels[0]).astype(int).reshape(-1, 1)

        # Concatenate locs, dscores and labels
        preds = np.concatenate([locs_np, dscores_np, labels_np], axis=1)

        return self.results_generation(preds, img_path, id_strip)

    def batch_image_detection(self, dataloader, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a batch of images.

        Args:
            dataloader (DataLoader): DataLoader containing image batches.
            conf_thres (float, optional): Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): Characters to strip from img_id. Defaults to None.

        Returns:
            list: List of detection results for all images.
        """
        batch_results = []

        with tqdm(total=len(dataloader)) as pbar:
            for batch in dataloader:
                imgs, paths, sizes = batch
                # Iterate over each image in the batch
                for img, path, size in zip(imgs, paths, sizes):
                    # img is now a single image tensor of shape [C, H, W]
                    # Perform detection on the single image
                    result = self.single_image_detection(
                        img, img_path=path, conf_thres=conf_thres, id_strip=id_strip)
                    batch_results.append(result)
                    pbar.update(1)

        return batch_results

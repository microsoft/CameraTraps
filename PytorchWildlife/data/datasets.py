# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from glob import glob
from pathlib import Path
from typing import List

from PIL import Image
import numpy as np
import supervision as sv
import torch
from torch.utils.data import Dataset

# Making the DetectionImageFolder class available for import from this module
__all__ = [
    "DetectionImageFolder",
    ]


class DetectionImageFolder(Dataset):
    """
    A PyTorch Dataset for loading images from a specified directory.
    Each item in the dataset is a tuple containing the image data, 
    the image's path, and the original size of the image.
    """

    def __init__(self, image_dir, transform=None, extensions=("jpg",)):
        """
        Initializes the dataset.

        Parameters:
            image_dir (str): Path to the directory containing the images.
            transform (callable, optional): Optional transform to be applied on the image.
            extensions (tuple, optional): Tuple of extensions to be used for filtering the images.
        """
        self.image_dir = image_dir

        # Listing and sorting all image files in the specified directory
        self.images = self.find_images(Path(image_dir), extensions)
        print(self.images)

        self.transform = transform

    def __getitem__(self, idx):
        """
        Retrieves an image from the dataset.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Contains the image data, the image's path, and its original size.
        """
        # Get image filename and path
        img_path = self.images[idx]

        # Load and convert image to RGB
        img = Image.open(img_path).convert("RGB")
        img_size_ori = img.size[::-1]
        
        # Apply transformation if specified
        if self.transform:
            img = self.transform(img)

        return img, img_path, torch.tensor(img_size_ori)
    
    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.images)

    def find_images(self, path_to_images: Path, extensions: List[str]):
        """
        Finds all images in the given path.

        Args:
            path_to_images (Path): Path to the images.
            extensions (List[str]): List of extensions to search for.
        Returns:
            List[Path]: Sorted list of paths to the images.
        """
        all_files = list(path_to_images.rglob("*.*"))

        # all extensions to lowercase (needs for case-insensitive search)
        extensions = [ext.lower() for ext in extensions]

        # remain images with extensions
        image_paths = [str(path) for path in all_files if path.suffix.lower().replace(".", "") in extensions]

        return sorted(image_paths)


class DetectionCrops(Dataset):

    def __init__(self, detection_results, transform=None, path_head=None, animal_cls_id=0):

        self.detection_results = detection_results
        self.transform = transform
        self.path_head = path_head
        self.animal_cls_id = animal_cls_id # This determins which detection class id represents animals.
        self.img_ids = []
        self.xyxys = []

        self.load_detection_results()

    def load_detection_results(self):
        for det in self.detection_results:
            for xyxy, det_id in zip(det["detections"].xyxy, det["detections"].class_id):
                # Only run recognition on animal detections
                if det_id == self.animal_cls_id:
                    self.img_ids.append(det["img_id"])
                    self.xyxys.append(xyxy)

    def __getitem__(self, idx):
        """
        Retrieves an image from the dataset.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Contains the image data and the image's path.
        """

        # Get image path and corresponding bbox xyxy for cropping
        img_id = self.img_ids[idx]
        xyxy = self.xyxys[idx]

        img_path = os.path.join(self.path_head, img_id) if self.path_head else img_id
        
        # Load and crop image with supervision
        img = sv.crop_image(np.array(Image.open(img_path).convert("RGB")),
                            xyxy=xyxy)
        
        # Apply transformation if specified
        if self.transform:
            img = self.transform(Image.fromarray(img))

        return img, img_path

    def __len__(self):
        return len(self.img_ids)
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
from torchvision import transforms
from yolov5.utils.augmentations import letterbox

# Making the provided classes available for import from this module
__all__ = [
    "MegaDetector_v5_Transform",
    "Classification_Inference_Transform"
]

class MegaDetector_v5_Transform:
    """
    A transformation class to preprocess images for the MegaDetector v5 model.
    This includes resizing, transposing, and normalization operations.
    This is a required transformation for the YoloV5 model.

    """

    def __init__(self, target_size=1280, stride=32):
        """
        Initializes the transform.

        Args:
            target_size (int): Desired size for the image's longest side after resizing.
            stride (int): Stride value for resizing.
        """
        self.target_size = target_size
        self.stride = stride

    def __call__(self, np_img):
        """
        Applies the transformation on the provided image.

        Args:
            np_img (np.ndarray): Input image as a numpy array.

        Returns:
            torch.Tensor: Transformed image.
        """
        # Resize and pad the image using the letterbox function
        img = letterbox(np_img, new_shape=self.target_size, stride=self.stride, auto=False)[0]

        # Transpose and convert image to PyTorch tensor
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0

        return img


class Classification_Inference_Transform:
    """
    A transformation class to preprocess images for classification inference.
    This includes resizing, normalization, and conversion to a tensor.
    """
    # Normalization constants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, target_size=224):
        """
        Initializes the transform.

        Args:
            target_size (int): Desired size for the height and width after resizing.
        """
        # Define the sequence of transformations
        self.trans = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __call__(self, img):
        """
        Applies the transformation on the provided image.

        Args:
            img (PIL.Image.Image): Input image in PIL format.

        Returns:
            torch.Tensor: Transformed image.
        """
        img = self.trans(img)
        return img

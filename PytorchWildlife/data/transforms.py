# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
from torchvision import transforms
#from yolov5.utils.augmentations import letterbox
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

# Making the provided classes available for import from this module
__all__ = [
    "MegaDetector_v5_Transform",
    "Classification_Inference_Transform"
]


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad an image to a desired shape while keeping the aspect ratio unchanged. 

    This function is commonly used in object detection tasks to prepare images for models like YOLOv5. 
    It resizes the image to fit into the new shape with the correct aspect ratio and then pads the rest.

    Parameters:
    im (PIL.Image.Image or torch.Tensor): The input image. It can be a PIL image or a PyTorch tensor.
    new_shape (tuple, optional): The target size of the image, in the form (height, width). Defaults to (640, 640).
    color (tuple, optional): The color used for padding. Defaults to (114, 114, 114).
    auto (bool, optional): Adjust padding to ensure the padded image is a multiple of stride. Defaults to True.
    scaleFill (bool, optional): If True, scales the image to fill the new shape, ignoring the aspect ratio. Defaults to False.
    scaleup (bool, optional): Allow the function to scale up the image. Defaults to True.
    stride (int, optional): The stride used in the model. The padding is adjusted to be a multiple of this stride. Defaults to 32.

    Returns:
    tuple: A tuple containing:
        - The transformed image as a torch.Tensor.
        - The scale ratios as a tuple (width_ratio, height_ratio).
        - The padding applied as a tuple (width_padding, height_padding).
    """

    # Convert PIL Image to Torch Tensor

    if isinstance(im, Image.Image):
        im = T.ToTensor()(im)

    # Original shape
    shape = im.shape[1:]  # shape = [height, width]

    # New shape
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) and compute padding
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = dw % stride, dh % stride
    elif scaleFill:
        dw, dh = 0, 0
        new_unpad = new_shape
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2
   
    # Resize image
    if shape[::-1] != new_unpad:
        resize_transform = T.Resize(new_unpad[::-1], interpolation=T.InterpolationMode.BILINEAR,
                                    antialias=False)
        im = resize_transform(im)

    # Pad image
    padding = (int(round(dw - 0.1)), int(round(dw + 0.1)), int(round(dh + 0.1)), int(round(dh - 0.1)))
    im = F.pad(im*255.0, padding, value=114)/255.0

    return im

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
            np_img (np.ndarray): Input image as a numpy array or PIL Image.

        Returns:
            torch.Tensor: Transformed image.
        """
        # Convert the image to a PyTorch tensor and normalize it
        if isinstance(np_img, np.ndarray):
            np_img = np_img.transpose((2, 0, 1))
            np_img = np.ascontiguousarray(np_img)
            np_img = torch.from_numpy(np_img).float()
            np_img /= 255.0

        # Resize and pad the image using a customized letterbox function. 
        img = letterbox(np_img, new_shape=self.target_size, stride=self.stride, auto=False)

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

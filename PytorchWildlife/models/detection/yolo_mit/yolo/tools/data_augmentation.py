from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class AugmentationComposer:
    """Composes several transforms together."""

    def __init__(self, transforms, image_size: int = [640, 640], base_size: int = 640):
        self.transforms = transforms
        # TODO: handle List of image_size [640, 640]
        self.pad_resize = PadAndResize(image_size)
        self.base_size = base_size

        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(self)

    def __call__(self, image, boxes=torch.zeros(0, 5)):
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        image, boxes, rev_tensor = self.pad_resize(image, boxes)
        image = TF.to_tensor(image)
        return image, boxes, rev_tensor


class PadAndResize:
    def __init__(self, image_size, background_color=(114, 114, 114)):
        """Initialize the object with the target image size."""
        self.target_width, self.target_height = image_size
        self.background_color = background_color

    def set_size(self, image_size: List[int]):
        self.target_width, self.target_height = image_size

    def __call__(self, image: Image, boxes):
        img_width, img_height = image.size
        scale = min(self.target_width / img_width, self.target_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pad_left = (self.target_width - new_width) // 2
        pad_top = (self.target_height - new_height) // 2
        padded_image = Image.new("RGB", (self.target_width, self.target_height), self.background_color)
        padded_image.paste(resized_image, (pad_left, pad_top))

        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * new_width + pad_left) / self.target_width
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * new_height + pad_top) / self.target_height

        transform_info = torch.tensor([scale, pad_left, pad_top, pad_left, pad_top])
        return padded_image, boxes, transform_info
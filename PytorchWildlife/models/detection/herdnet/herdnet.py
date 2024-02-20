# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from ..base_detector import HerdNetDetector

__all__ = [
    'HerdNetV1',
]

class HerdNetV1(HerdNetDetector):
    """
    HerdNet is a specialized class derived from the HerdNetBase class 
    that is specifically designed for detecting animals, persons, and vehicles.
    
    Attributes:
        CLASS_NAMES (dict): Maps class IDs to names for detected entities.
        DOWN_RATIO (int): Downscaling factor to manage image resolution.
        PATCHSIZE (tuple): Size of image patches for processing (width, height).
        OVERLAP (int): Pixel overlap between adjacent patches for seamless detection.
        IMAGE_SIZE (int): Standard image size (width/height) used during training.
        STRIDE (int): Stride of sliding window for object scanning.
        
    References:
        This class implements techniques described in the following research article:
        "From crowd to herd counting: How to precisely detect and count African mammals 
        using aerial imagery and deep learning". DOI: https://doi.org/10.1016/j.isprsjprs.2023.01.025

    """
    
    CLASS_NAMES = {
        0: "background",
        1: "Buffalo",
        2: "Elephant",
        3: "Kob",
        4: "Topi",
        5: "Warthog",
        6: "Waterbug"
    }
    DOWN_RATIO = 2
    PATCHSIZE = (512, 512)
    OVERLAP = 160
    IMAGE_SIZE = 1280  # image size used in training
    STRIDE = 64

    def __init__(self, weights=None, device="cpu", pretrained=True):
        """
        Initializes the HerdNet model with the option to load pretrained weights.
        
        Args:
            weights (str, optional): Path to the weights file.
            device (str, optional): Device to load the model on (e.g., "cpu" or "cuda"). Default is "cpu".
            pretrained (bool, optional): Whether to load the pretrained model. Default is True.
        """
        
        if pretrained:
            url = "https://zenodo.org/records/10456832/files/20220413_herd_net_v2_delplanque_2022.pth?download=1"
        else:
            url = None

        super(HerdNetV1, self).__init__(weights=weights, device=device, url=url)


# %%

"""
This is a Pytorch-Wildlife loader for the Deepfaune-New-England classifier.
The original model is available at: https://code.usgs.gov/vtcfwru/deepfaune-new-england/-/tree/main?ref_type=heads
Licence: CC0 1.0 Universal 
Copyright USGS 2024
laurence.clarfeld@uvm.edu
"""

# Import libraries

from .base_classifier import TIMM_BaseClassifierInference

__all__ = [
    "DFNE"
]

class DFNE(TIMM_BaseClassifierInference):
    """
    Base detector class for dinov2 classifier. This class provides utility methods
    for loading the model, performing single and batch image classifications, and 
    formatting results. Make sure the appropriate file for the model weights has been 
    downloaded to the "models" folder before running DFNE.
    """
    BACKBONE = "vit_large_patch14_dinov2.lvd142m"
    MODEL_NAME = "dfne_weights_v1_0.pth"
    IMAGE_SIZE = 182
    CLASS_NAMES = {
                0: "American Marten",
                1: "Bird sp.",
                2: "Black Bear",
                3: "Bobcat",
                4: "Coyote",
                5: "Domestic Cat",
                6: "Domestic Cow",
                7: "Domestic Dog",
                8: "Fisher",
                9: "Gray Fox",
                10: "Gray Squirrel",
                11: "Human",
                12: "Moose",
                13: "Mouse sp.",
                14: "Opossum",
                15: "Raccoon",
                16: "Red Fox",
                17: "Red Squirrel",
                18: "Skunk",
                19: "Snowshoe Hare",
                20: "White-tailed Deer",
                21: "Wild Boar",
                22: "Wild Turkey",
                23: "no-species"
            }

    def __init__(self, weights=None, device="cpu", transform=None):
        url = 'https://prod-is-usgs-sb-prod-publish.s3.amazonaws.com/67ae17fcd34e3f09c0e0f002/dfne_weights_v1_0.pth'
        super(DFNE, self).__init__(weights=weights, device=device, url=url, transform=transform, weights_key='model_state_dict')    
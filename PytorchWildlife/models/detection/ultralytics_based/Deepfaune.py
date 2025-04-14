"""
This is a Pytorch-Wildlife loader for the Deepfaune detector.
The original Deepfaune model is available at: https://www.deepfaune.cnrs.fr/en/
Licence: CC BY-SA 4.0
Copyright CNRS 2024
simon.chamaille@cefe.cnrs.fr; vincent.miele@univ-lyon1.fr
"""

from .yolov8_base import YOLOV8Base

__all__ = [
    'DeepfauneDetector',
]

class DeepfauneDetector(YOLOV8Base):
    """
    MegaDetectorV6 is a specialized class derived from the YOLOV8Base class 
    that is specifically designed for detecting animals, persons, and vehicles.
    
    Attributes:
        CLASS_NAMES (dict): Mapping of class IDs to their respective names.
    """
    
    CLASS_NAMES = {
        0: "animal",
        1: "person",
        2: "vehicle"
    }

    def __init__(self, weights=None, device="cpu"):
        """
        Initializes the MegaDetectorV5 model with the option to load pretrained weights.
        
        Args:
            weights (str, optional): Path to the weights file.
            device (str, optional): Device to load the model on (e.g., "cpu" or "cuda"). Default is "cpu".
            pretrained (bool, optional): Whether to load the pretrained model. Default is True.
            version (str, optional): Version of the model to load. Default is 'yolov9c'.
        """
        self.IMAGE_SIZE = 960

        url = "https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.3/deepfaune-yolov8s_960.pt" 
        self.MODEL_NAME = "deepfaune-yolov8s_960.pt"

        super(DeepfauneDetector, self).__init__(weights=weights, device=device, url=url)
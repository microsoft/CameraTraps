
from .yolo_mit_base import YOLOMITBase

__all__ = [
    'MegaDetectorV6MIT'
]

class MegaDetectorV6MIT(YOLOMITBase):
    """
    MegaDetectorV6 is a specialized class derived from the YOLOMITBase class 
    that is specifically designed for detecting animals, persons, and vehicles.
    
    Attributes:
        CLASS_NAMES (dict): Mapping of class IDs to their respective names.
    """
    
    CLASS_NAMES = {
        0: "animal",
        1: "person",
        2: "vehicle"
    }

    def __init__(self, weights=None, device="cpu", pretrained=True, version='MDV6-yolov9-c-mit'):
        """
        Initializes the MegaDetectorV6 model with the option to load pretrained weights.
        
        Args:
            weights (str, optional): Path to the weights file.
            device (str, optional): Device to load the model on (e.g., "cpu" or "cuda"). Default is "cpu".
            pretrained (bool, optional): Whether to load the pretrained model. Default is True.
            version (str, optional): Version of the model to load. Default is 'MDV6-yolov9-c-mit'.
        """
        self.IMAGE_SIZE = 640

        if version == 'MDV6-mit-yolov9-c':
            url = "https://zenodo.org/records/15398270/files/MDV6-mit-yolov9-c.ckpt?download=1"
            self.MODEL_NAME = "MDV6-mit-yolov9-c.ckpt"
        elif version == 'MDV6-mit-yolov9-e':
            url = "https://zenodo.org/records/15398270/files/MDV6-mit-yolov9-e.ckpt?download=1"
            self.MODEL_NAME = "MDV6-mit-yolov9-e.ckpt"
        else:
            raise ValueError('Select a valid model version: MDV6-mit-yolov9-c or MDV6-mit-yolov9-e')

        super(MegaDetectorV6MIT, self).__init__(weights=weights, device=device, url=url)
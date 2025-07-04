
from .yolov8_base import YOLOV8Base

__all__ = [
    'MegaDetectorV6'
]

class MegaDetectorV6(YOLOV8Base):
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

    def __init__(self, weights=None, device="cpu", pretrained=True, version='yolov9c'):
        """
        Initializes the MegaDetectorV5 model with the option to load pretrained weights.
        
        Args:
            weights (str, optional): Path to the weights file.
            device (str, optional): Device to load the model on (e.g., "cpu" or "cuda"). Default is "cpu".
            pretrained (bool, optional): Whether to load the pretrained model. Default is True.
            version (str, optional): Version of the model to load. Default is 'yolov9c'.
        """
        self.IMAGE_SIZE = 1280

        if version == 'MDV6-yolov9-c':            
            url = "https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1" 
            self.MODEL_NAME = "MDV6b-yolov9-c.pt"
        elif version == 'MDV6-yolov9-e':
            url = "https://zenodo.org/records/15398270/files/MDV6-yolov9-e-1280.pt?download=1"
            self.MODEL_NAME = "MDV6-yolov9-e-1280.pt"
        elif version == 'MDV6-yolov10-c':
            url = "https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1"
            self.MODEL_NAME = "MDV6-yolov10-c.pt"
        elif version == 'MDV6-yolov10-e':
            url = "https://zenodo.org/records/15398270/files/MDV6-yolov10-e-1280.pt?download=1"
            self.MODEL_NAME = "MDV6-yolov10-e-1280.pt"
        elif version == 'MDV6-rtdetr-c':
            url = "https://zenodo.org/records/15398270/files/MDV6-rtdetr-c.pt?download=1"
            self.MODEL_NAME = "MDV6b-rtdetr-c.pt"
        else:
            raise ValueError('Select a valid model version: MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e, MDV6-rtdetr-c')

        super(MegaDetectorV6, self).__init__(weights=weights, device=device, url=url)
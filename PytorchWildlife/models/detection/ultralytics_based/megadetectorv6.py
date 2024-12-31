
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
        self.IMAGE_SIZE = 640

        if version == 'yolov9c':            
            url = "https://zenodo.org/records/14567879/files/MDV6b-yolov9c.pt?download=1" 
            self.MODEL_NAME = "MDV6b-yolov9c.pt"
        elif version == 'yolov9e':
            url = "https://zenodo.org/records/14567879/files/MDV6-yolov9e.pt?download=1"
            self.MODEL_NAME = "MDV6-yolov9e.pt"
        elif version == 'yolov10n':
            url = "https://zenodo.org/records/14567879/files/MDV6-yolov10n.pt?download=1"
            self.MODEL_NAME = "MDV6-yolov10n.pt"
        elif version == 'yolov10x':
            url = "https://zenodo.org/records/14567879/files/MDV6-yolov10x.pt?download=1"
            self.MODEL_NAME = "MDV6-yolov10x.pt"
        elif version =='rtdetrl':
            url = "https://zenodo.org/records/14567879/files/MDV6b-rtdetrl.pt?download=1"
            self.MODEL_NAME = "MDV6b-rtdetrl.pt"
        else:
            print('Select a valid model version: yolov9c, yolov9e, yolov10n, yolov10x or rtdetrl')

        super(MegaDetectorV6, self).__init__(weights=weights, device=device, url=url)

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Base detector class. """

# Importing basic libraries
from torch import nn

class BaseDetector(nn.Module):
    """
    Base detector class. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """
    
    # Placeholder class-level attributes to be defined in derived classes
    IMAGE_SIZE = None
    STRIDE = None
    CLASS_NAMES = None
    TRANSFORM = None

    def __init__(self, weights=None, device="cpu", url=None):
        """
        Initialize the base detector.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        """
        super(BaseDetector, self).__init__()
        self.device = device


    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load model weights.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        Raises:
            Exception: If weights are not provided.
        """
        pass

    def results_generation(self, preds, img_id, id_strip=None):
        """
        Generate results for detection based on model predictions.
        
        Args:
            preds (numpy.ndarray): 
                Model predictions.
            img_id (str): 
                Image identifier.
            id_strip (str, optional): 
                Strip specific characters from img_id. Defaults to None.

        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        pass

    def single_image_detection(self, img, img_size=None, img_path=None, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a single image.
        
        Args:
            img (str or ndarray): 
                Image path or ndarray of images.
            img_size (tuple): 
                Original image size.
            img_path (str): 
                Image path or identifier.
            conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """
        pass

    def batch_image_detection(self, dataloader, conf_thres=0.2, id_strip=None):
        """
        Perform detection on a batch of images.
        
        Args:
            dataloader (DataLoader): 
                DataLoader containing image batches.
            conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            list: List of detection results for all images.
        """
        pass

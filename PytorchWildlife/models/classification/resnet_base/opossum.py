# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from .base_classifier import PlainResNetInference

__all__ = [
    "AI4GOpossum"
]


class AI4GOpossum(PlainResNetInference):
    """
    Opossum Classifier that inherits from PlainResNetInference.
    This classifier is specialized for distinguishing between Opossums and Non-opossums.
    """
    
    # Image size for the Opossum classifier
    IMAGE_SIZE = 224
    
    # Class names for prediction
    CLASS_NAMES = {
        0: "Non-opossum",
        1: "Opossum"
    }

    def __init__(self, weights=None, device="cpu", pretrained=True):
        """
        Initialize the Opossum Classifier.

        Args:
            weights (str, optional): Path to the model weights. Defaults to None.
            device (str, optional): Device for model inference. Defaults to "cpu".
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        """

        # If pretrained, use the provided URL to fetch the weights
        if pretrained:
            url = "https://zenodo.org/records/10023414/files/OpossumClassification_v0.0.0.ckpt?download=1"
        else:
            url = None

        super(AI4GOpossum, self).__init__(weights=weights, device=device,
                                          num_cls=1, num_layers=50, url=url)

    def results_generation(self, logits, img_ids, id_strip=None):
        """
        Generate results for classification.

        Args:
            logits (torch.Tensor): Output tensor from the model.
            img_id (list): List of image identifier.
            id_strip (str): stiping string for better image id saving.       

        Returns:
            dict: Dictionary containing image ID, prediction, and confidence score.
        """

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).squeeze(1).numpy().astype(int)

        results = []
        for pred, img_id, prob in zip(preds, img_ids, probs):
            r = {"img_id": str(img_id).strip(id_strip)}
            r["prediction"] = self.CLASS_NAMES[pred]
            r["class_id"] = pred
            r["confidence"] = prob.item() if pred == 1 else (1 - prob.item())
            results.append(r)
        
        return results

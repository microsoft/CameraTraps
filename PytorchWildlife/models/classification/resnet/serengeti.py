# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from .base_classifier import PlainResNetInference

__all__ = [
    "AI4GSnapshotSerengeti"
]


class AI4GSnapshotSerengeti(PlainResNetInference):
    """
    Snapshot Serengeti Animal Classifier that inherits from PlainResNetInference.
    This classifier is specialized for recognizing 9 different animals and has 1 'other' class.
    """
    
    # Image size for the Opossum classifier
    IMAGE_SIZE = 224
    
    # Class names for prediction
    CLASS_NAMES = {
        0: 'wildebeest',
        1: 'guineafowl',
        2: 'zebra',
        3: 'buffalo',
        4: 'gazellethomsons',
        5: 'gazellegrants',
        6: 'warthog',
        7: 'impala',
        8: 'hyenaspotted',
        9: 'other'
    }

    def __init__(self, weights=None, device="cpu", pretrained=True):
        """
        Initialize the Amazon animal Classifier.

        Args:
            weights (str, optional): Path to the model weights. Defaults to None.
            device (str, optional): Device for model inference. Defaults to "cpu".
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        """

        # If pretrained, use the provided URL to fetch the weights
        if pretrained:
            url = "https://zenodo.org/records/10456813/files/AI4GSnapshotSerengeti.ckpt?download=1"
        else:
            url = None

        super(AI4GSnapshotSerengeti, self).__init__(weights=weights, device=device,
                                                   num_cls=10, num_layers=18, url=url)

    def results_generation(self, logits, img_ids, id_strip=None):
        """
        Generate results for classification.

        Args:
            logits (torch.Tensor): Output tensor from the model.
            img_id (str): Image identifier.
            id_strip (str): stiping string for better image id saving.       

        Returns:
            dict: Dictionary containing image ID, prediction, and confidence score.
        """
        
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1)[0]
        confidences = probs[0].tolist()
        result = [[self.CLASS_NAMES[i], confidence] for i, confidence in enumerate(confidences)]

        results = []
        for pred, img_id, conf in zip(preds, img_ids, confs):
            r = {"img_id": str(img_id).strip(id_strip)}
            r["prediction"] = self.CLASS_NAMES[pred.item()]
            r["class_id"] = pred.item()
            r["confidence"] = conf.item()
            r["all_confidences"] = result
            results.append(r)
        
        return results

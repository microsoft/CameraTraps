# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from .base_classifier import PlainResNetInference

__all__ = [
    "CustomWeights"
]


class CustomWeights(PlainResNetInference):
    """
    Custom Weight Classifier that inherits from PlainResNetInference.
    This classifier can load any model that was based on the PytorchWildlife finetuning tool.
    """
    
    # Image size for the classifier
    IMAGE_SIZE = 224


    def __init__(self, weights=None, class_names=None, device="cpu"):
        """
        Initialize the Amazon animal Classifier.

        Args:
            weights (str, optional): Path to the model weights. Defaults to None.
            device (str, optional): Device for model inference. Defaults to "cpu".
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        """
        self.CLASS_NAMES = class_names
        self.num_cls = len(self.CLASS_NAMES)
        super(CustomWeights, self).__init__(weights=weights, device=device,
                                                   num_cls=self.num_cls, num_layers=50, url=None)

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

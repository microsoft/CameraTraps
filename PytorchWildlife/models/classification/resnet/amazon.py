# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from .base_classifier import PlainResNetInference

__all__ = [
    "AI4GAmazonRainforest"
]


class AI4GAmazonRainforest(PlainResNetInference):
    """
    Amazon Ranforest Animal Classifier that inherits from PlainResNetInference.
    This classifier is specialized for recognizing 36 different animals in the Amazon Rainforest.
    """
    
    # Image size for the Opossum classifier
    IMAGE_SIZE = 224
    
    # Class names for prediction
    CLASS_NAMES = {
        0: 'Dasyprocta',
        1: 'Bos',
        2: 'Pecari',
        3: 'Mazama',
        4: 'Cuniculus',
        5: 'Leptotila',
        6: 'Human',
        7: 'Aramides',
        8: 'Tinamus',
        9: 'Eira',
        10: 'Crax',
        11: 'Procyon',
        12: 'Capra',
        13: 'Dasypus',
        14: 'Sciurus',
        15: 'Crypturellus',
        16: 'Tamandua',
        17: 'Proechimys',
        18: 'Leopardus',
        19: 'Equus',
        20: 'Columbina',
        21: 'Nyctidromus',
        22: 'Ortalis',
        23: 'Emballonura',
        24: 'Odontophorus',
        25: 'Geotrygon',
        26: 'Metachirus',
        27: 'Catharus',
        28: 'Cerdocyon',
        29: 'Momotus',
        30: 'Tapirus',
        31: 'Canis',
        32: 'Furnarius',
        33: 'Didelphis',
        34: 'Sylvilagus',
        35: 'Unknown'
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
            url = "https://zenodo.org/records/10042023/files/AI4GAmazonClassification_v0.0.0.ckpt?download=1"
        else:
            url = None

        super(AI4GAmazonRainforest, self).__init__(weights=weights, device=device,
                                                   num_cls=36, num_layers=50, url=url)

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

        results = []
        for pred, img_id, conf in zip(preds, img_ids, confs):
            r = {"img_id": str(img_id).strip(id_strip)}
            r["prediction"] = self.CLASS_NAMES[pred.item()]
            r["class_id"] = pred.item()
            r["confidence"] = conf.item()
            results.append(r)
        
        return results

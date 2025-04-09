
""" model class for loading the DFNE classifier. """

# Import libraries

import os
import wget
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import timm

from ..base_classifier import BaseClassifierInference
from ....data import transforms as pw_trans
from ....data import datasets as pw_data 


class TIMM_BaseClassifierInference(BaseClassifierInference):
    """
    Base detector class for dinov2 classifier. This class provides utility methods
    for loading the model, performing single and batch image classifications, and 
    formatting results. Make sure the appropriate file for the model weights has been 
    downloaded to the "models" folder before running DFNE.
    """

    BACKBONE = None
    MODEL_NAME = None
    IMAGE_SIZE = None

    def __init__(self, weights=None, device="cpu", url=None, transform=None,
                 weights_key='model_state_dict', weights_prefix=''):
        """
        Initialize the model.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
            weights_key (str, optional): 
                Key to fetch the model weights. Defaults to None.
            weights_prefix (str, optional): 
                prefix of model weight keys. Defaults to None.
        """
        super(TIMM_BaseClassifierInference, self).__init__()
        self.device = device

        if transform:
            self.transform = transform
        else:
            self.transform = pw_trans.Classification_Inference_Transform(target_size=self.IMAGE_SIZE)

        self._load_model(weights, url, weights_key, weights_prefix)

    def _load_model(self, weights=None, url=None, weights_key='model_state_dict', weights_prefix=''):
        """
        Load TIMM based model weights
        
        Args:
        weights (str, optional): 
            Path to the model weights. (defaults to None)
        url (str, optional): 
            url to the model weights. (defaults to None)
        """

        self.predictor = timm.create_model(
            self.BACKBONE, 
            pretrained = False, 
            num_classes = len(self.CLASS_NAMES),
            dynamic_img_size = True
        )

        if url:
            if not os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", self.MODEL_NAME)):
                os.makedirs(os.path.join(torch.hub.get_dir(), "checkpoints"), exist_ok=True)
                weights = wget.download(url, out=os.path.join(torch.hub.get_dir(), "checkpoints"))
            else:
                weights = os.path.join(torch.hub.get_dir(), "checkpoints", self.MODEL_NAME)
        elif weights is None:
            raise Exception("Need weights for inference.")

        checkpoint = torch.load(
            f = weights,
            map_location = self.device,
            weights_only = False
        )[weights_key]

        checkpoint = OrderedDict({k.replace("{}".format(weights_prefix), ""): checkpoint[k]
                                    for k in checkpoint})

        self.predictor.load_state_dict(checkpoint)
        print("Model loaded from {}".format(os.path.join(torch.hub.get_dir(), "checkpoints", self.MODEL_NAME)))

        self.predictor.to(self.device)
        self.eval()

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

    def single_image_classification(self, img, img_id=None, id_strip=None):
        """
        Perform classification on a single image.
        
        Args:
            img (str or ndarray): 
                Image path or ndarray of images.
            img_id (str, optional): 
                Image path or identifier.
            id_strip (str, optional):
                Whether to strip stings in id. Defaults to None.

        Returns:
            (dict): Classification results.
        """
        if type(img) == str:
            img = Image.open(img).convert("RGB")
        else:
            img = Image.fromarray(img)
        img = self.transform(img)

        logits = self.predictor(img.unsqueeze(0).to(self.device))
        return self.results_generation(logits.cpu(), [img_id], id_strip=id_strip)[0]

    def batch_image_classification(self, data_path=None, det_results=None, id_strip=None,
                                   batch_size=32, num_workers=0, **kwargs):
        """
        Perform classification on a batch of images.
        
        Args:
            data_path (str): 
                Path containing all images for inference. Defaults to None. 
            det_results (dict):
                Dirct outputs from detectors. Defaults to None.
            id_strip (str, optional):
                Whether to strip stings in id. Defaults to None.
            batch_size (int, optional):
                Batch size for inference. Defaults to 32.
            num_workers (int, optional):
                Number of workers for dataloader. Defaults to 0.

        Returns:
            (dict): Classification results.
        """

        if data_path:
            dataset = pw_data.ImageFolder(
                data_path,
                transform=self.transform,
                path_head='.'
            )
        elif det_results:
            dataset = pw_data.DetectionCrops(
                det_results,
                transform=self.transform,
                path_head='.'
            )
        else:
            raise Exception("Need data for inference.")

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=False, pin_memory=True, drop_last=False, **kwargs)
        
        total_logits = []
        total_paths = []

        with tqdm(total=len(dataloader)) as pbar: 
            for batch in dataloader:
                imgs, paths = batch
                imgs = imgs.to(self.device)
                total_logits.append(self.predictor(imgs))
                total_paths.append(paths)
                pbar.update(1)

        total_logits = torch.cat(total_logits, dim=0).cpu()
        total_paths = np.concatenate(total_paths, axis=0)

        return self.results_generation(total_logits, total_paths, id_strip=id_strip)

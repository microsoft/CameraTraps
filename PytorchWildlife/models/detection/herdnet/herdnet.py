from ..base_detector import BaseDetector
from ..herdnet.animaloc.eval import HerdNetStitcher, HerdNetLMDS
from ....data import datasets as pw_data
from .model import HerdNet as HerdNetArch

import torch
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
import torchvision.transforms as transforms  

import numpy as np
from PIL import Image
from tqdm import tqdm
import supervision as sv
import os
import wget

class HerdNet(BaseDetector):
    """
    HerdNet detector class. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """
    
    def __init__(self, weights=None, device="cpu", dataset='general' ,url="https://zenodo.org/records/13899852/files/20220413_HerdNet_General_dataset_2022.pth?download=1", transform=None):
        """
        Initialize the HerdNet detector.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            dataset (str, optional):
                Dataset for which the model is trained. It should be either 'general' or 'ennedi'. Defaults to 'general'.
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
            transform (torchvision.transforms.Compose, optional):
                Image transformation for inference. Defaults to None.
        """
        super(HerdNet, self).__init__(weights=weights, device=device, url=url)
        # Assert that the dataset is either 'general' or 'ennedi'
        dataset = dataset.lower()
        assert dataset in ['general', 'ennedi'], "Dataset should be either 'general' or 'ennedi'"
        if dataset == 'ennedi':
            url = "https://zenodo.org/records/13914287/files/20220329_HerdNet_Ennedi_dataset_2023.pth?download=1"
        self._load_model(weights, device, url)

        self.stitcher = HerdNetStitcher( # This module enables patch-based inference
            model = self.model,
            size = (512,512),
            overlap = 160,
            down_ratio = 2,
            up = True, 
            reduction = 'mean',
            device_name = device
            )
        
        self.lmds_kwargs: dict = {'kernel_size': (3, 3), 'adapt_ts': 0.2, 'neg_ts': 0.1}
        self.lmds = HerdNetLMDS(up=False, **self.lmds_kwargs) # Local Maxima Detection Strategy

        if not transform:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.img_mean, std=self.img_std)  
                ]) 
        else:
            self.transforms = transform

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load the HerdNet model weights.
        
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
        if weights:
            checkpoint = torch.load(weights, map_location=torch.device(device))
        elif url:
            filename = url.split('/')[-1][:-11] # Splitting the URL to get the filename and removing the '?download=1' part
            if not os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", filename)):
                os.makedirs(os.path.join(torch.hub.get_dir(), "checkpoints"), exist_ok=True)
                weights = wget.download(url, out=os.path.join(torch.hub.get_dir(), "checkpoints"))
            else:
                weights = os.path.join(torch.hub.get_dir(), "checkpoints", filename)
            checkpoint = torch.load(weights, map_location=torch.device(device))
            #checkpoint = load_state_dict_from_url(url, map_location=torch.device(self.device)) # NOTE: This function is not used in the current implementation
        else:
            raise Exception("Need weights for inference.")
        
        # Load the class names and other metadata from the checkpoint
        self.CLASS_NAMES = checkpoint["classes"]
        self.num_classes = len(self.CLASS_NAMES) + 1
        self.img_mean = checkpoint['mean']
        self.img_std = checkpoint['std']

        # Load the model architecture
        self.model = HerdNetArch(num_classes=self.num_classes, pretrained=False)

        # Load checkpoint into model
        state_dict = checkpoint['model_state_dict']  
        # Remove 'model.' prefix from the state_dict keys if the key starts with 'model.'
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        # Load the new state_dict 
        self.model.load_state_dict(new_state_dict, strict=True)

        print(f"Model loaded from {weights}")

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
        results = {"img_id": str(img_id).strip(id_strip) if id_strip else str(img_id)}
        results["detections"] = sv.Detections(
            xyxy=preds[:, :4],
            confidence=preds[:, 4],
            class_id=preds[:, 5].astype(int)
        )
        results["labels"] = [
            f"{self.CLASS_NAMES[class_id]} {confidence:0.2f}"
            for confidence, class_id in zip(results["detections"].confidence, results["detections"].class_id)
        ]
        return results
    
    def single_image_detection(self, img, img_path=None, det_conf_thres=0.2, clf_conf_thres=0.2, id_strip=None):
        """
        Perform detection on a single image.

        Args:
            img (str or np.ndarray): 
                Image for inference.
            img_path (str, optional): 
                Path to the image. Defaults to None.
            det_conf_thres (float, optional):
                Confidence threshold for detections. Defaults to 0.2.
            clf_conf_thres (float, optional):
                Confidence threshold for classification. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results for the image.
        """  
        if isinstance(img, str):  
            img_path = img_path or img  
            img = np.array(Image.open(img_path).convert("RGB"))  
        if self.transforms:  
            img_tensor = self.transforms(img)

        preds = self.stitcher(img_tensor)  
        heatmap, clsmap = preds[:,:1,:,:], preds[:,1:,:,:]  
        counts, locs, labels, scores, dscores = self.lmds((heatmap, clsmap))
        preds_array = self.process_lmds_results(counts, locs, labels, scores, dscores, det_conf_thres, clf_conf_thres)
        return self.results_generation(preds_array, img_path, id_strip=id_strip)  


    def batch_image_detection(self, data_path, det_conf_thres=0.2, clf_conf_thres=0.2, batch_size=1, id_strip=None):
        """
        Perform detection on a batch of images.
        
        Args:
            data_path (str): 
                Path containing all images for inference.
            det_conf_thres (float, optional):
                Confidence threshold for detections. Defaults to 0.2.
            clf_conf_thres (float, optional):
                Confidence threshold for classification. Defaults to 0.2.
            batch_size (int, optional):
                Batch size for inference. Defaults to 1.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            list: List of detection results for all images.
        """
        dataset = pw_data.DetectionImageFolder(
            data_path,
            transform=self.transforms
        )
        # Creating a Dataloader for batching and parallel processing of the images
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=True, num_workers=0, drop_last=False) # TODO: discuss. why is num_workers 0?
        
        results = []

        with tqdm(total=len(loader)) as pbar:
            for batch_index, (imgs, paths, sizes) in enumerate(loader):
                imgs = imgs.to(self.device)
                predictions = self.stitcher(imgs[0]).detach().cpu()
                heatmap, clsmap = predictions[:,:1,:,:], predictions[:,1:,:,:]
                counts, locs, labels, scores, dscores = self.lmds((heatmap, clsmap))
                preds_array = self.process_lmds_results(counts, locs, labels, scores, dscores, det_conf_thres, clf_conf_thres) 
                results_dict = self.results_generation(preds_array, paths[0], id_strip=id_strip)
                pbar.update(1)
                sizes = sizes.numpy()
                normalized_coords = [[x1 / sizes[0][0], y1 / sizes[0][1], x2 / sizes[0][0], y2 / sizes[0][1]] for x1, y1, x2, y2 in preds_array[:, :4]] # TODO: Check if this is correct due to xy swapping 
                results_dict['normalized_coords'] = normalized_coords
                results.append(results_dict)
        return results

    def process_lmds_results(self, counts, locs, labels, scores, dscores, det_conf_thres=0.2, clf_conf_thres=0.2):
        """
        Process the results from the Local Maxima Detection Strategy.

        Args:
            counts (list): 
                Number of detections for each species.
            locs (list): 
                Locations of the detections.
            labels (list): 
                Labels of the detections.
            scores (list): 
                Scores of the detections.
            dscores (list): 
                Detection scores.
            det_conf_thres (float, optional):
                Confidence threshold for detections. Defaults to 0.2.
            clf_conf_thres (float, optional):
                Confidence threshold for classification. Defaults to 0.2.

        Returns:
            numpy.ndarray: Processed detection results.
        """
        # Flatten the lists since we know its a single image 
        counts = counts[0]  
        locs = locs[0]  
        labels = labels[0]  
        scores = scores[0]
        dscores = dscores[0]  
    
        # Calculate the total number of detections  
        total_detections = sum(counts)  
        
        # Pre-allocate based on total possible detections  
        preds_array = np.empty((total_detections, 6)) #xyxy, confidence, class_id format
        detection_idx = 0
        valid_detections_idx = 0 # Index for valid detections after applying the confidence threshold
        # Loop through each species  
        for specie_idx in range(len(counts)):  
            count = counts[specie_idx]  
            if count == 0:  
                continue  
            
            # Get the detections for this species  
            species_locs = np.array(locs[detection_idx : detection_idx + count])
            species_locs[:, [0, 1]] = species_locs[:, [1, 0]] # Swap x and y in species_locs
            species_scores = np.array(scores[detection_idx : detection_idx + count])
            species_dscores = np.array(dscores[detection_idx : detection_idx + count])
            species_labels = np.array(labels[detection_idx : detection_idx + count])

            # Apply the confidence threshold
            valid_detections_by_clf_score = species_scores > clf_conf_thres
            valid_detections_by_det_score = species_dscores > det_conf_thres
            valid_detections = np.logical_and(valid_detections_by_clf_score, valid_detections_by_det_score)
            valid_detections_count = np.sum(valid_detections)
            valid_detections_idx += valid_detections_count
            # Fill the preds_array with the valid detections
            if valid_detections_count > 0:
                preds_array[valid_detections_idx - valid_detections_count : valid_detections_idx, :2] = species_locs[valid_detections] - 1
                preds_array[valid_detections_idx - valid_detections_count : valid_detections_idx, 2:4] = species_locs[valid_detections] + 1
                preds_array[valid_detections_idx - valid_detections_count : valid_detections_idx, 4] = species_scores[valid_detections]
                preds_array[valid_detections_idx - valid_detections_count : valid_detections_idx, 5] = species_labels[valid_detections]
            
            detection_idx += count # Move to the next species 
        
        preds_array = preds_array[:valid_detections_idx] # Remove the empty rows
        
        return preds_array

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the model.
        
        Args:
            input (torch.Tensor): 
                Input tensor for the model.
        
        Returns:
            torch.Tensor: Model output.
        """
        # Call the forward method of the model in evaluation mode
        self.model.eval()
        return self.model(input)

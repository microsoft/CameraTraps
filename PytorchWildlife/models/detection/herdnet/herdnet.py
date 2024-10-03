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

class HerdNet(BaseDetector):
    """
    HerdNet detector class. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """
    
    def __init__(self, weights=None, device="cpu", url=None, transform=None):
        """
        Initialize the HerdNet detector.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        """
        super(HerdNet, self).__init__(weights=weights, device=device, url=url)
        self.model = HerdNetArch(num_classes=7, pretrained=False) # TODO: Do we want to keep the number of classes hardcoded or with an argument?
        # TODO: We can also define self.model before super().__init__ and once again call the _load_model meethod in the BaseDetector class
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
            checkpoint = load_state_dict_from_url(url, map_location=torch.device(self.device))
        else:
            raise Exception("Need weights for inference.")
        
        # Load checkpoint into model
        state_dict = checkpoint['model_state_dict']  
  
        # Remove 'model.' prefix from the state_dict keys  
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}  
  
        # Load the new state_dict 
        self.model.load_state_dict(new_state_dict, strict=True)

        print(f"Model loaded from {weights}")

        self.CLASS_NAMES = checkpoint["classes"]
        self.num_classes = len(self.CLASS_NAMES) + 1
        self.img_mean = checkpoint['mean']
        self.img_std = checkpoint['std']

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
    
    # TODO: see if it works in an image with multiple species
    def single_image_detection(self, img, img_path=None, id_strip=None):
        """
        Perform detection on a single image.

        Args:
            img (str or np.ndarray): 
                Image for inference.
            img_path (str, optional): 
                Path to the image. Defaults to None.
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
        preds_array = self.process_lmds_results(counts, locs, labels, scores, dscores)
        return self.results_generation(preds_array, img_path, id_strip=id_strip)  


    def batch_image_detection(self, data_path, batch_size=1, id_strip=None, extension='JPG'):
        """
        Perform detection on a batch of images.
        
        Args:
            data_path (str): 
                Path containing all images for inference.
            batch_size (int, optional):
                Batch size for inference. Defaults to 1.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.
            extension (str, optional):
                Image extension to search for. Defaults to "JPG"

        Returns:
            list: List of detection results for all images.
        """
        dataset = pw_data.DetectionImageFolder(
            data_path,
            extension=extension,
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
                preds_array = self.process_lmds_results(counts, locs, labels, scores, dscores)
                results_dict = self.results_generation(preds_array, paths[0], id_strip=id_strip)
                pbar.update(1)
                sizes = sizes.numpy()
                normalized_coords = [[x1 / sizes[0][0], y1 / sizes[0][1], x2 / sizes[0][0], y2 / sizes[0][1]] for x1, y1, x2, y2 in preds_array[:, :4]] # TODO: Check if this is correct due to xy swapping 
                results_dict['normalized_coords'] = normalized_coords
                results.append(results_dict)
        return results

    def process_lmds_results(self, counts, locs, labels, scores, dscores):
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

        Returns:
            numpy.ndarray: Processed detection results.
        """
        # Flatten the lists since we know its a single image 
        counts = counts[0]  
        locs = locs[0]  
        labels = labels[0]  
        scores = scores[0]  
    
        # Calculate the total number of detections  
        total_detections = sum(counts)  
        
        # Pre-allocate based on total possible detections  
        preds_array = np.empty((total_detections, 6)) #xyxy, confidence, class_id format
        detection_idx = 0  
        # Loop through each species  
        for specie_idx in range(len(counts)):  
            count = counts[specie_idx]  
            if count == 0:  
                continue  
    
            # Get the detections for this species  
            species_locs = np.array(locs[detection_idx : detection_idx + count])
            species_locs[:, [0, 1]] = species_locs[:, [1, 0]] # Swap x and y in species_locs
            species_scores = np.array(scores[detection_idx : detection_idx + count])
            species_labels = np.array(labels[detection_idx : detection_idx + count])
    
            # Populate the pre-allocated array (xyxy, confidence, class_id format)
            preds_array[detection_idx : detection_idx + count, :2] = species_locs - 1  
            preds_array[detection_idx : detection_idx + count, 2:4] = species_locs + 1  
            preds_array[detection_idx : detection_idx + count, 4] = species_scores  
            preds_array[detection_idx : detection_idx + count, 5] = species_labels  
            
            detection_idx += count
        
        preds_array = preds_array[:detection_idx]  # Trim to the actual number of detections 
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
        return self.model(input)

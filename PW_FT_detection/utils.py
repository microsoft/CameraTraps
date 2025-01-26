import os
import wget
import torch

def get_model_path(model):

    if model == "MDV6-yolov9-c":           
        url = "https://zenodo.org/records/14567879/files/MDV6b-yolov9c.pt?download=1" 
        model_name = "MDV6b-yolov9c.pt"
    elif model == "MDV6-yolov9-e":
        url = "https://zenodo.org/records/14567879/files/MDV6-yolov9e.pt?download=1"
        model_name = "MDV6-yolov9e.pt"
    elif model == "MDV6-yolov10-c":
        url = "https://zenodo.org/records/14567879/files/MDV6-yolov10n.pt?download=1"
        model_name = "MDV6-yolov10n.pt"
    elif model == "MDV6-yolov10-e":
        url = "https://zenodo.org/records/14567879/files/MDV6-yolov10x.pt?download=1"
        model_name = "MDV6-yolov10x.pt"
    elif model == "MDV6-rtdetr-c":
        url = "https://zenodo.org/records/14567879/files/MDV6b-rtdetrl.pt?download=1"
        model_name = "MDV6b-rtdetrl.pt"
    else:
        raise ValueError('Select a valid model version: MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e or MDV6-rtdetr-c')

    if not os.path.exists(os.path.join(torch.hub.get_dir(), "checkpoints", model_name)):
        os.makedirs(os.path.join(torch.hub.get_dir(), "checkpoints"), exist_ok=True)
        model_path = wget.download(url, out=os.path.join(torch.hub.get_dir(), "checkpoints"))
    else:
        model_path = os.path.join(torch.hub.get_dir(), "checkpoints", model_name)
    
    return model_path
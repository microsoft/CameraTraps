#Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation

## Welcome to Version 1.0

The **Pytorch-Wildlife** library allows users to directly load the MegadetectorV5 model weights for animal detection. We've fully refactored our codebase, prioritizing ease of use in model deployment and expansion. In addition to `MegadetectorV5`, **Pytorch-Wildlife** also accommodates a range of classification weights, such as those derived from the Amazon Rainforest dataset and the Opossum classification dataset. Explore the codebase and functionalities of **Pytorch-Wildlife** through our interactive `Gradio` web app and detailed Jupyter notebooks, designed to showcase the practical applications of our enhancements. You can find more information in our [documentation](https://cameratraps.readthedocs.io/en/latest/).

## Table of Contents
- [Welcome to Version 1.0](#welcome-to-version-10)
- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Install through pip:](#install-through-pip)
- [Running the Demo](#running-the-demo)
- [License](#license)
- [Copyright](#copyright)
 
## Prerequisites
 
1. Python 3.8 
2. NVIDIA GPU (for CUDA support, although the demo can run on CPU)

If you have conda/mamba installed, you can create a new environment with the following command:
```bash
conda create -n pytorch-wildlife python=3.8 -y
conda activate pytorch-wildlife
```

## Installation
### Install through pip:
```bash
pip install PytorchWildlife
```
## Running the Demo
Here is a brief example on how to perform detection and classification on a single image using `PyTorch-wildlife`:

```python
import torch
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

img = torch.randn((3, 1280, 1280))

# Detection
detection_model = pw_detection.MegaDetectorV5() # Model weights are automatically downloaded.
detection_result = detection_model.single_image_detection(img)

#Classification
classification_model = pw_classification.AI4GAmazonRainforest() # Model weights are automatically downloaded.
classification_results = classification_model.single_image_classification(img)
```

If you want to use our Gradio demo for a user-friendly interface. Please run the following code inside the current repo. You can also find Jupyter Notebooks with an image and video tutorial:

```bash
git clone -b PytorchWildlife_Dev --single-branch https://github.com/microsoft/CameraTraps.git
cd CameraTraps
cd demo
# For the image demo
python image_demo.py
# For the video demo
python video_demo.py
# For the gradio app
python demo_gradio.py
```
The `demo_gradio.py` will launch a Gradio interface where you can:
- Perform Single Image Detection: Upload an image and set a confidence threshold to get detections.
- Perform Batch Image Detection: Upload a zip file containing multiple images to get detections in a JSON format.
- Perform Video Detection: Upload a video and get a processed video with detected animals. *The package installed via pip might encounter video loading issues with browsers for this task (i.e., processed videos don't show in the webapp), as the opencv installed through pip lacks a browser-supported codec. This issue will be resolved once we have a conda package available.*

<img src="images/gradio_UI.png">
  
## License
This project is licensed under the MIT License. Refer to the LICENSE file for more details.
## Copyright
Copyright (c) Microsoft Corporation. All rights reserved.

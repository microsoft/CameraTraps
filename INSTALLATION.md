# Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation

## Welcome to Version 1.0

The **Pytorch-Wildlife** library allows users to directly load the MegadetectorV5 model weights for animal detection. We've fully refactored our codebase, prioritizing ease of use in model deployment and expansion. In addition to `MegadetectorV5`, **Pytorch-Wildlife** also accommodates a range of classification weights, such as those derived from the Amazon Rainforest dataset and the Opossum classification dataset. Explore the codebase and functionalities of **Pytorch-Wildlife** through our interactive `Gradio` web app and detailed Jupyter notebooks, designed to showcase the practical applications of our enhancements. You can find more information in our [documentation](https://cameratraps.readthedocs.io/en/latest/).

## Table of Contents
- [Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation](#pytorch-wildlife-a-collaborative-deep-learning-framework-for-conservation)
  - [Welcome to Version 1.0](#welcome-to-version-10)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
    - [Create environment](#create-environment)
    - [Ubuntu](#ubuntu)
    - [MacOS](#macos)
  - [Installation](#installation)
    - [Install through pip:](#install-through-pip)
  - [Running the Demo](#running-the-demo)
  - [License](#license)
  - [Copyright](#copyright)
 
## Prerequisites
 
1. Python 3.8 
2. NVIDIA GPU for CUDA support (Optional, the code and demo also supports cpu calculation).
3. `conda` or `mamba` for python environment management and specific version of `opencv`.
4. If you are using CUDA. [CudaToolkit 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive) is required.

### Create environment
If you have `conda` or `mamba` installed, you can create a new environment with the following commands (switch `conda` to `mamba` for `mamba` users):
```bash
conda create -n pytorch-wildlife python=3.8 -y
conda activate pytorch-wildlife
```
NOTE: For Windows users, please use the Anaconda Prompt if you are using Anaconda. Otherwise, please use PowerShell for the conda environment and the rest of the set up.

### Ubuntu
If you are using a clean install of Ubuntu, additional libraries of OpenCV may need to be installed, please run the following command:
```bash
sudo apt-get update
sudo apt-get install -y python3-opencv
```

### MacOS
If you are using MacOS, please run the following command to install ffmpeg for video decoding:
```bash
brew install ffmpeg
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
git clone https://github.com/microsoft/CameraTraps.git
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
- Perform Video Detection: Upload a video and get a processed video with detected animals. 

As a showcase platform, the gradio demo offers a hands-on experience with all the available features. However, it's important to note that this interface is primarily for demonstration purposes. While it is fully equipped to run all the features effectively, it may not be optimized for scenarios involving excessive data loads. We advise users to be mindful of this limitation when experimenting with large datasets.

Some browsers may not render processed videos due to unsupported codec. If that happens, please either use a newer version of browser or run the following for a `conda` version of `opencv` and choose `avc1` in the Video encoder drop down menu in the webapp (this might not work for MacOS):

```bash
pip uninstall opencv-python
conda install -c conda-forge opencv
```

<img src="images/gradio_UI.png">

**NOTE: Windows may encounter some errors with large file uploads making Batch Image Detection and Video Detection unable to process. It is a Gradio issue. Newer versions of Gradio in the future may fix this problem.**
  
## License
This project is licensed under the MIT License. Refer to the LICENSE file for more details.
## Copyright
Copyright (c) Microsoft Corporation. All rights reserved.

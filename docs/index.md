![image](https://zenodo.org/records/15376499/files/Pytorch_Banner_transparentbk.png)

<div align="center"> 
<font size="6"> A Collaborative Deep Learning Framework for Conservation </font>
<br>
<hr>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/v/PytorchWildlife?color=limegreen" /></a> 
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://static.pepy.tech/badge/pytorchwildlife" /></a> 
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/pyversions/PytorchWildlife" /></a> 
<a href="https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" /></a>
<a href="https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing"><img src="https://img.shields.io/badge/Colab-Demo-blue?logo=GoogleColab" /></a>
<!-- <a href="https://colab.research.google.com/drive/16-OjFVQ6nopuP-gfqofYBBY00oIgbcr1?usp=sharing"><img src="https://img.shields.io/badge/Colab-Video detection-blue?logo=GoogleColab" /></a> -->
<a href="https://github.com/microsoft/CameraTraps/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/PytorchWildlife" /></a>
<a href="https://discord.gg/TeEVxzaYtm"><img src="https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=Discord" /></a>
<a href="https://microsoft.github.io/CameraTraps/"><img src="https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white" /></a>
<br><br>
</div>


## 👋 Welcome to Pytorch-Wildlife
**PyTorch-Wildlife** is an AI platform designed for the AI for Conservation community to create, modify, and share powerful AI conservation models. It allows users to directly load a variety of models including [MegaDetector](https://microsoft.github.io/CameraTraps/megadetector/), [DeepFaune](https://microsoft.github.io/CameraTraps/megadetector/), and [HerdNet](https://github.com/Alexandre-Delplanque/HerdNet) from our ever expanding [model zoo](model_zoo/megadetector.md) for both animal detection and classification. In the future, we will also include models that can be used for applications, including underwater images and bioacoustics. We want to provide a unified and straightforward experience for both practitioners and developers in the AI for conservation field. Your engagement with our work is greatly appreciated, and we eagerly await any feedback you may have.


## 🚀 Quick Start

👇 Here is a brief example of how to perform detection and classification on a single image using `PyTorch-wildlife`
```python
import numpy as np
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

img = np.random.randn(3, 1280, 1280)

# Detection
detection_model = pw_detection.MegaDetectorV6() # Model weights are automatically downloaded.
detection_result = detection_model.single_image_detection(img)

#Classification
classification_model = pw_classification.AI4GAmazonRainforest() # Model weights are automatically downloaded.
classification_results = classification_model.single_image_classification(img)
```

## ⚙️ Install Pytorch-Wildlife
```
pip install PytorchWildlife
```
Please refer to our [installation guide](installation.md) for more installation information.


## 🖼️ Examples

### Image detection using `MegaDetector`
<img src="https://zenodo.org/records/15376499/files/animal_det_1.JPG" alt="animal_det_1" width="300"/><br>
*Credits to Universidad de los Andes, Colombia.*

### Image classification with `MegaDetector` and `AI4GAmazonRainforest`
<img src="https://zenodo.org/records/15376499/files/animal_clas_1.png" alt="animal_clas_1" width="300"/><br>
*Credits to Universidad de los Andes, Colombia.*

### Opossum ID with `MegaDetector` and `AI4GOpossum`
<img src="https://zenodo.org/records/15376499/files/opossum_det.png" alt="opossum_det" width="300"/><br>
*Credits to the Agency for Regulation and Control of Biosecurity and Quarantine for Galápagos (ABG), Ecuador.*


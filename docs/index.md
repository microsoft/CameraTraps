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
<a href="https://cameratraps.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/read-docs-yellow?logo=ReadtheDocs" /></a>
<a href="https://github.com/microsoft/CameraTraps/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/PytorchWildlife" /></a>
<a href="https://discord.gg/TeEVxzaYtm"><img src="https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=Discord" /></a>
<br><br>
</div>


## 👋 Welcome to Pytorch-Wildlife

**PyTorch-Wildlife** is a platform to create, modify, and share powerful AI conservation models. These models can be used for a variety of applications, including camera trap images, overhead images, underwater images, or bioacoustics. Your engagement with our work is greatly appreciated, and we eagerly await any feedback you may have.


The **Pytorch-Wildlife** library allows users to directly load the `MegaDetector` model weights for animal detection. We've fully refactored our codebase, prioritizing ease of use in model deployment and expansion. In addition to `MegaDetector`, **Pytorch-Wildlife** also accommodates a range of classification weights, such as those derived from the Amazon Rainforest dataset and the Opossum classification dataset. Explore the codebase and functionalities of **Pytorch-Wildlife** through our interactive [HuggingFace web app](https://huggingface.co/spaces/AndresHdzC/pytorch-wildlife) or local [demos and notebooks](./demo), designed to showcase the practical applications of our enhancements at [PyTorchWildlife](docs/installation.md). You can find more information in our [documentation](https://cameratraps.readthedocs.io/en/latest/).

👇 Here is a brief example on how to perform detection and classification on a single image using `PyTorch-wildlife`
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


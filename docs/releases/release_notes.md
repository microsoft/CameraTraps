# Main changes and additions

### Pytorch-Wildlife Version 1.2.1

#### SpeciesNet is available in Pytorch-Wildlife for testing! 
- We have added SpeciesNet into our model zoo, which is compatible with all detection models provided by Pytorch-Wildlife. Please refer to [this document](https://github.com/microsoft/CameraTraps/blob/SppNet_TF/PytorchWildlife/models/classification/speciesnet_base/sppnet_readme.md) for more details!

#### Deepfaune in Our Model Zoo!! 
- We are excited to announce the release of the Deepfaune models—both the detector and classifier—in PyTorch-Wildlife, adding to our growing model zoo. A huge thank you to the Deepfaune team for your support! Deepfaune is one of the most comprehensive models focused on the European ecosystem for both detection and classification. It serves as a great complement to MegaDetector, which has primarily been trained on datasets from North America, South America, and Africa. The Deepfaune detector is also our first third-party camera trap detection model integrated into PyTorch-Wildlife!
- To use the model, you just need to load them as any other Pytorch-Wildife models: 
```
detection_model = pw_detection.DeepfauneDetector(device=DEVICE)
classification_model = pw_classification.DeepfauneClassifier(device=DEVICE)
```
- You can also use the `detection_classification_pipeline_demo.py` script in the demo folder to test the whole detection + classification pipeline. 
- Please also take a look at the original [Deepfaune website](https://www.deepfaune.cnrs.fr/en/) and give them a star! 

#### Deepfaune-New-England in Our Model Zoo Too!!
- Besides the original Deepfaune mode, there is another fine-tuned Deepfaune model developed by USGS for the Northeastern NA area called Deepfaune-New-England (DFNE). It can also be loaded with `classification_model = pw_classification.DFNE(device=DEVICE)`
- Please take a look at the orignal [DFNE repo](https://code.usgs.gov/vtcfwru/deepfaune-new-england/-/tree/main?ref_type=heads) and give them a star! 


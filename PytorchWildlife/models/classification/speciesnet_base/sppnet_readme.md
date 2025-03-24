## SpeciesNet Available in Pytorch-Wildlife for testing! 

- Wildlife Insights and Google released an amazing animal classification model a couple weeks ago, ***SpecieNet***. We think it might also make sense to host the model on Pytorch-Wildlife and see how it functions with our framework and pipeline, and the interactions with other models and data. 
- Currently, it is not in our official branch for two reasons: 
    1) The current version of SpeciesNet is in Tensorflow, which can be a little tricky for a pure Pytorch integration. But the developers mentioned in their repo that they want to ultimately release a Pytorch version. We would like to wait for that version for our main branch release.
    2) We want to first have our users to test SpeciesNet in Pytorch-Wildlife and see if it makes sense to everybody.
- Since the nature of Pytorch-Wildlife is being an AI platform instead of a single model, we would like to use this opportunity to show how easy it can be to integrate third-party models into Pytorch-Wildlife, standardize the input and output formats and pipeline, and make it compatible with existing models in the model zoo. 

### How to use SpeciesNet in Pytorch-Wildlife?
- To start using SpeciesNet, please use our [`SppNet_TF branch`](https://github.com/microsoft/CameraTraps/tree/SppNet_TF)
- Then, before everything, please install this branch using `pip install -e .` and SpeciesNet python package with `pip install speciesnet`
- Everything should be ready at this point. You can load SpeciesNet the way we load all the other PW models:
```python
from PytorchWildlife.models import classification as pw_classification
classification_model = pw_classification.SpeciesNetTFInference(version='v4.0.0a', run_mode='multi_thread')
```
- This [demo file](https://github.com/microsoft/CameraTraps/blob/SppNet_TF/demo/detection_classification_pipeline_sppnet_demo.py) also shows how the single and batch detection + classification work under Pytorch-Wildife and can be a starting point for your reference. 

Please let us know what you think and if you have any questions! Also, remember to go to [SpeciesNet Repo](https://github.com/google/cameratrapai) and give them a star! 

Thank you as always!

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch.nn as nn

# Making the PlainResNetInference class available for import from this module
__all__ = ["BaseClassifierInference"]

class BaseClassifierInference(nn.Module):
    """
    Inference module for the PlainResNet Classifier.
    """
    def __init__(self):
        super(BaseClassifierInference, self).__init__()
        pass

    def results_generation(self):
        pass

    def forward(self):
        pass

    def single_image_classification(self):
        pass

    def batch_image_classification(self):
        pass

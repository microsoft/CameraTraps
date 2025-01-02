import os
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import *


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_state_dict_from_url
# Exportable class names for external use
__all__ = [
    'PlainResNetClassifier'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
}

class ResNetBackbone(ResNet):
    """
    Custom ResNet backbone class for feature extraction.

    Inherits from the torchvision ResNet class and allows customization of the architecture.
    """

    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        """
        Initialize the ResNet backbone.

        Args:
            block (nn.Module): Type of block to use (BasicBlock or Bottleneck).
            layers (list of int): Number of layers in each block.
            zero_init_residual (bool): Zero-initialize the last BN in each residual branch.
            groups (int): Number of groups for group normalization.
            width_per_group (int): Width per group.
            replace_stride_with_dilation (list of bool or None): Use dilation instead of stride.
            norm_layer (callable or None): Norm layer to use.
        """
        super(ResNetBackbone, self).__init__(
            block=block,
            layers=layers,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )

    def _forward_impl(self, x):
        """
        Forward pass implementation for the ResNet backbone.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the ResNet backbone.
        """
        # Applying the ResNet layers and operations
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class PlainResNetClassifier(nn.Module):
    """
    Custom ResNet classifier class.

    Extends nn.Module and provides a complete ResNet-based classifier, including feature extraction and classification layers.
    """

    name = 'PlainResNetClassifier'

    def __init__(self, num_cls=10, num_layers=18):
        """
        Initialize the PlainResNetClassifier.

        Args:
            num_cls (int): Number of classes for the classifier.
            num_layers (int): Number of layers in the ResNet model (e.g., 18, 50).
        """
        super(PlainResNetClassifier, self).__init__()
        self.num_cls = num_cls
        self.num_layers = num_layers
        self.feature = None
        self.classifier = None
        self.criterion_cls = None

        # Initialize the network with the specified settings
        self.setup_net()

    def setup_net(self):
        """
        Set up the ResNet network and initialize its weights.
        """
        kwargs = {}

        # Selecting the appropriate ResNet architecture and pre-trained weights
        if self.num_layers == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
            #self.pretrained_weights = ResNet18_Weights.IMAGENET1K_V1
            self.pretrained_weights = state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                              progress=True)
        elif self.num_layers == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
            #self.pretrained_weights = ResNet50_Weights.IMAGENET1K_V1
            self.pretrained_weights = state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
        else:
            raise Exception('ResNet Type not supported.')

        # Constructing the feature extractor and classifier
        self.feature = ResNetBackbone(block, layers, **kwargs)
        self.classifier = nn.Linear(512 * block.expansion, self.num_cls)

    def setup_criteria(self):
        """
        Set up the criterion for the classifier.
        """
        # Criterion for binary classification
        self.criterion_cls = nn.CrossEntropyLoss()

    def feat_init(self):
        """
        Initialize the feature extractor with pre-trained weights.
        """
        # Load pre-trained weights and adjust for the current model
        #init_weights = self.pretrained_weights.get_state_dict(progress=True)
        init_weights = self.pretrained_weights
        init_weights = OrderedDict({k.replace('module.', '').replace('feature.', ''): init_weights[k]
                                    for k in init_weights})

        # Load the weights into the feature extractor
        self.feature.load_state_dict(init_weights, strict=False)

        # Identify missing and unused keys in the loaded weights
        load_keys = set(init_weights.keys())
        self_keys = set(self.feature.state_dict().keys())

        missing_keys = self_keys - load_keys
        unused_keys = load_keys - self_keys
        print('missing keys: {}'.format(sorted(list(missing_keys))))
        print('unused_keys: {}'.format(sorted(list(unused_keys))))

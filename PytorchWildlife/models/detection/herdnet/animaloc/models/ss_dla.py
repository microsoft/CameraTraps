__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import torch

import torch.nn as nn
import numpy as np

from typing import Optional

from .register import MODELS
from . import dla as dla_modules


@MODELS.register()
class SemSegDLA(nn.Module):
    ''' Semantic segmentation version of DLA with multi channels output mode '''

    def __init__(
        self,
        num_layers: int = 34,
        num_classes: int = 2,
        pretrained: bool = True, 
        down_ratio: Optional[int] = 2, 
        head_conv: int = 64,
        ):
        '''
        Args:
            num_layers (int, optional): number of layers of DLA. Defaults to 34.
            num_classes (int, optional): number of output classes, background included. 
                Defaults to 2.
            pretrained (bool, optional): set False to disable pretrained DLA encoder parameters
                from ImageNet. Defaults to True.
            down_ratio (int, optional): downsample ratio. Possible values are 1, 2, 4, 8, or 16. 
                Set to 1 to get output of the same size as input (i.e. no downsample).
                Defaults to 2.
            head_conv (int, optional): number of supplementary convolutional layers at the end 
                of decoder. Defaults to 64.
        '''

        super(SemSegDLA, self).__init__()

        assert down_ratio in [1, 2, 4, 8, 16], \
            f'Downsample ratio possible values are 1, 2, 4, 8 or 16, got {down_ratio}'
        
        base_name = 'dla{}'.format(num_layers)

        self.down_ratio = down_ratio
        self.num_classes = num_classes

        self.first_level = int(np.log2(down_ratio))
        self.base = dla_modules.__dict__[base_name](pretrained=pretrained, return_levels=True)
        channels = self.base.channels

        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = dla_modules.DLAUp(channels[self.first_level:], scales=scales)

        # heatmap head
        self.hm_head = nn.Sequential(
            nn.Conv2d(channels[self.first_level], head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d( 
                head_conv, self.num_classes - 1,
                kernel_size=1, stride=1, 
                padding=0, bias=True
                ),
            nn.Sigmoid()
            )

        # self.hm_head[-2].bias.data.fill_(0.00)
        
    def forward(self, input: torch.Tensor):
        encode = self.base(input)
        decode_hm = self.dla_up(encode[self.first_level:])
        heatmap = self.hm_head(decode_hm)

        return heatmap
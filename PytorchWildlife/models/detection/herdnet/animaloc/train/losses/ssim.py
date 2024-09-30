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


''' 
Following code adapted from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py 
Free to use under the MIT license.
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    return ssim_map

''' Personal code '''
from typing import Optional
from .register import LOSSES

def _ssim_loss(
    output: torch.Tensor,
    target: torch.Tensor, 
    window: torch.autograd.Variable,
    window_size: int, 
    channel: int,
    reduction: str,
    weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

    assert reduction in ['mean', 'sum'], \
        f'Reduction must be either \'mean\' or \'sum\', got {reduction}'

    if weights is not None:
        assert weights.shape[0] == channel, \
            'Number of weights must match the number of channels, ' \
                f'got {channel} channels and {weights.shape[0]} weights'

    ssim_map = _ssim(target, output, window, window_size, channel)

    if weights is not None:
        weights = weights.to(output.device)
        ssim_list = 1. - ssim_map.mean(3).mean(2)
        loss = weights * ssim_list
    else:
        loss =  1. - ssim_map
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()

def ssim_loss(
    output: torch.Tensor,
    target: torch.Tensor, 
    window_size: int = 11, 
    reduction: str = 'mean',
    weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

    (_, channel, _, _) = target.size()
    window = create_window(window_size, channel)
    
    if target.is_cuda:
        window = window.cuda(target.get_device())
    window = window.type_as(target)

    return _ssim_loss(target, output, window, window_size, channel, reduction, weights)

@LOSSES.register()
class SSIMLoss(torch.nn.Module):

    def __init__(
        self, 
        window_size: int = 11, 
        reduction: str = 'mean', 
        weights: Optional[torch.Tensor] = None
        ) -> None:

        super(SSIMLoss, self).__init__()

        self.window_size = window_size
        self.reduction = reduction
        self.weights = weights
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        (_, channel, _, _) = target.size()

        if channel == self.channel and self.window.data.type() == target.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if target.is_cuda:
                window = window.cuda(target.get_device())
            window = window.type_as(target)
            
            self.window = window
            self.channel = channel

        return _ssim_loss(target, output, window, self.window_size, channel, self.reduction, self.weights)

@LOSSES.register()
class ISSIMLoss(torch.nn.Module):
    ''' Independent SSIM loss, as presented in https://arxiv.org/abs/2102.07925 '''

    def __init__(
        self, 
        local_size: int, 
        n_backpts: int = 30, 
        reduction: str = 'mean',
        weights: Optional[torch.Tensor] = None
        ) -> None:

        super(ISSIMLoss, self).__init__()

        self.local_size = local_size
        self.n_backpts = n_backpts
        self.reduction = reduction
        self.weights = weights

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        batch_size = target.shape[0]

        loss = torch.zeros((batch_size, 1), dtype=torch.float32, device=target.device)
        for b in range(batch_size):

            points = torch.nonzero(target[b]==1.)
            n_points = torch.count_nonzero(target[b]==1.)

            if n_points == 0:
                h = torch.randint(target.shape[2], (self.n_backpts, 1))
                w = torch.randint(target.shape[3], (self.n_backpts, 1))
                z = torch.zeros(self.n_backpts, 1, dtype=torch.int)
                points = torch.cat((z,h,w), 1)
                n_points = self.n_backpts
            
            for pt in points:
                y, x = pt[1], pt[2]
                local_target = self._local_crop(target[b], y, x).unsqueeze(0)
                local_est = self._local_crop(output[b], y, x).unsqueeze(0)
                loss[b] += ssim_loss(local_target, local_est, 
                    window_size=self.local_size, reduction='sum', weights=self.weights)
            
            loss[b] = torch.div(loss[b], n_points)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
    
    def _local_crop(self, input: torch.Tensor, y: int, x: int) -> torch.Tensor:

        off = self.local_size // 2

        y_min, x_min = max(0, y-off), max(0, x-off)
        y_max, x_max = min(input.shape[1]-1, y+off), min(input.shape[2]-1, x+off)
        
        return input[:,y_min:y_max,x_min:x_max]

@LOSSES.register()
class LocalSSIMLoss(torch.nn.Module):
    ''' Compute SSIM loss only in the areas where the target pixels 
    values are greater than 0. '''

    def __init__(self, window_size: int = 11, reduction: str = 'mean') -> None:
        '''
        Args:
            window_size (int, optional): window size, in pixels. Defaults to 11.
            reduction (str, optional): reduction mode, possible mode are 'mean',
                'sum', or 'none'. Defaults to 'mean'.
        '''

        assert reduction in ['none', 'mean', 'sum'], \
            f'Reduction must be either \'none\', \'mean\' or \'sum\', got {reduction}'
        
        super().__init__()

        self.window_size = window_size
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        mask = self._mask(target)
        window = self._create_window(target)
        _, channel, _, _ = target.shape

        output = 1. - _ssim(target, input, window, self.window_size, channel)
        output = mask * output

        return self._reduce(output.mean(1))

    def _mask(self, target: torch.Tensor) -> torch.Tensor:
        return (target > 0)
    
    def _create_window(self, target: torch.Tensor) -> torch.autograd.Variable:

        (_, channel, _, _) = target.size()
        window = create_window(self.window_size, channel)
        
        if target.is_cuda:
            window = window.cuda(target.get_device())

        window = window.type_as(target)

        return window

    def _reduce(self, output: torch.Tensor) -> torch.Tensor:

        if self.reduction == 'mean':
            return output.mean()
        
        elif self.reduction == 'sum':
            return output.sum()

        else:
            return output
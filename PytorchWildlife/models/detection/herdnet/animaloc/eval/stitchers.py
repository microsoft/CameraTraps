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
import torchvision

import torch.nn.functional as F
import numpy as np

from typing import List, Tuple
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from ..data import ImageToPatches

class Stitcher(ImageToPatches):
    ''' Class to stitch detections of patches into original image
    coordinates system 

    This algorithm works as follow:
        1) Cut original image into patches
        2) Make inference on each patches and harvest the detections
        3) Patch the detections maps into the coordinate system of the original image
        Optional:
        4) Upsample the patched detection map
    '''

    def __init__(
        self,
        model: torch.nn.Module, 
        size: Tuple[int,int], 
        overlap: int = 100,
        batch_size: int = 1,
        down_ratio: int = 1,
        up: bool = False,
        reduction: str = 'sum',
        device_name: str = 'cuda',
        ) -> None:
        '''
        Args:
            model (torch.nn.Module): CNN detection model, that takes as inputs image and returns
                output and dict (i.e. wrapped by LossWrapper)
            size (tuple): patches size (height, width), in pixels
            overlap (int, optional): overlap between patches, in pixels. 
                Defaults to 100. 
            batch_size (int, optional): batch size used for inference over patches. 
                Defaults to 1.
            down_ratio (int, optional): downsample ratio. Set to 1 to get output of the same 
                size as input (i.e. no downsample). Defaults to 1.
            up (bool, optional): set to True to upsample the patched map. Defaults to False.
            reduction (str, optional): specifies the reduction to apply on overlapping areas.
                Possible values are 'sum', 'mean', 'max'. Defaults to 'sum'.
            device_name (str, optional): the device name on which tensors will be allocated 
                ('cpu' or 'cuda'). Defaults to 'cuda'.
        '''

        assert isinstance(model, torch.nn.Module), \
            'model argument must be an instance of nn.Module()'
        
        assert reduction in ['sum', 'mean', 'max'], \
            'reduction argument possible values are \'sum\', \'mean\' and \'max\' ' \
                f'got \'{reduction}\''

        self.model = model
        self.size = size
        self.overlap = overlap
        self.batch_size = batch_size
        self.down_ratio = down_ratio
        self.up = up
        self.reduction = reduction
        self.device = torch.device(device_name)

        self.model.to(self.device)

    def __call__(
        self, 
        image: torch.Tensor
        ) -> torch.Tensor:
        ''' Apply the stitching algorithm to the image

        Args:
            image (torch.Tensor): image of shape [C,H,W]
        
        Returns:
            torch.Tensor
                the detections into the coordinate system of the original image
        '''

        super(Stitcher, self).__init__(image, self.size, self.overlap)
        
        self.image = image.to(torch.device('cpu')) 

        # step 1 - get patches and limits
        patches = self.make_patches()

        # step 2 - inference to get maps
        det_maps = self._inference(patches)

        # step 3 - patch the maps into initial coordinates system
        patched_map = self._patch_maps(det_maps)
        patched_map = self._reduce(patched_map)

        # (step 4 - upsample)
        if self.up:
            patched_map = F.interpolate(patched_map, scale_factor=self.down_ratio, 
                mode='bilinear', align_corners=True)

        return patched_map

    
    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> List[torch.Tensor]:
        
        self.model.eval()

        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset,   
            batch_size=self.batch_size,
            sampler=SequentialSampler(dataset)
            )

        maps = []
        for patch in dataloader:
            patch = patch[0].to(self.device)
            outputs, _ = self.model(patch)
            maps = [*maps, *outputs.unsqueeze(0)]

        return maps

    def _patch_maps(self, maps: List[torch.Tensor]) -> torch.Tensor:

        _, h, w = self.image.shape
        dh, dw = h // self.down_ratio, w // self.down_ratio
        kernel_size = np.array(self.size) // self.down_ratio
        stride = kernel_size - self.overlap // self.down_ratio
        output_size = (
            self._ncol * kernel_size[0] - ((self._ncol-1) * self.overlap // self.down_ratio), 
            self._nrow * kernel_size[1] - ((self._nrow-1) * self.overlap // self.down_ratio)
            )

        maps = torch.cat(maps, dim=0)

        if self.reduction == 'max':
            out_map = self._max_fold(maps, output_size=output_size,
                kernel_size=tuple(kernel_size), stride=tuple(stride))
        else:
            n_patches = maps.shape[0]
            maps = maps.permute(1,2,3,0).contiguous().view(1, -1, n_patches)
            out_map = F.fold(maps, output_size=output_size, 
                kernel_size=tuple(kernel_size), stride=tuple(stride))

        out_map = out_map[:,:, 0:dh, 0:dw]

        return out_map
    
    def _reduce(self, map: torch.Tensor) -> torch.Tensor:

        dh = self.image.shape[1] // self.down_ratio
        dw = self.image.shape[2] // self.down_ratio
        ones = torch.ones(self.image.shape[0],dh,dw)

        if self.reduction == 'mean':
            ones_patches = ImageToPatches(ones, 
                np.array(self.size)//self.down_ratio, 
                self.overlap//self.down_ratio
                ).make_patches()

            ones_patches = [p.unsqueeze(0).unsqueeze(0) for p in ones_patches[:,1,:,:]]
            norm_map = self._patch_maps(ones_patches)
        
        else:
            norm_map = ones[1,:,:]
        
        return torch.div(map.to(self.device), norm_map.to(self.device))
    
    def _max_fold(self, maps: torch.Tensor, output_size: tuple, 
        kernel_size: tuple, stride: tuple
        ) -> torch.Tensor:
        
        output = torch.zeros((1, maps.shape[1], *output_size))

        fn = lambda x: [[i, i+kernel_size[x]] for i in range(0, output_size[x], stride[x])][:-1]
        locs = [[*h, *w] for h in fn(0) for w in fn(1)]

        for loc, m in zip(locs, maps):
            patch = torch.zeros(output.shape)
            patch[:,:, loc[0]:loc[1], loc[2]:loc[3]] = m
            output = torch.max(output, patch)

        return output

class HerdNetStitcher(Stitcher):

    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> List[torch.Tensor]:
        
        self.model.eval()

        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset,   
            batch_size=self.batch_size,
            sampler=SequentialSampler(dataset)
            )

        maps = []
        for patch in dataloader:
            patch = patch[0].to(self.device)
            #outputs = self.model(patch)[0]
            outputs = self.model(patch) # LossWrapper is not used
            heatmap = outputs[0]
            scale_factor = 16
            clsmap = F.interpolate(outputs[1], scale_factor=scale_factor, mode='nearest')
            # cat
            outmaps = torch.cat([heatmap, clsmap], dim=1)
            maps = [*maps, *outmaps.unsqueeze(0)]

        return maps

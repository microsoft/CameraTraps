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

from .utils import HannWindow2D

from ..data import ImageToPatches

from ..utils.registry import Registry

STITCHERS = Registry('stitchers', module_key='animaloc.eval.stitchers')

__all__ = ['STITCHERS', *STITCHERS.registry_names]

@STITCHERS.register()
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

@STITCHERS.register()
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
            outputs = self.model(patch)[0]
            heatmap = outputs[0]
            scale_factor = 16
            clsmap = F.interpolate(outputs[1], scale_factor=scale_factor, mode='nearest')
            # cat
            outmaps = torch.cat([heatmap, clsmap], dim=1)
            maps = [*maps, *outmaps.unsqueeze(0)]

        return maps

@STITCHERS.register()
class FasterRCNNStitcher(Stitcher):

    def __init__(
        self,
        model: torch.nn.Module, 
        size: Tuple[int,int], 
        overlap: int = 100,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.0,
        batch_size: int = 1,
        device_name: str = 'cuda',
        ) -> None:
        super().__init__(model, size, overlap=overlap, batch_size=batch_size, device_name=device_name)
        
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.up = False

    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> List[dict]:
        
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
            maps.append(*outputs)

        return maps
    
    def _patch_maps(self, maps: List[dict]) -> dict:
        boxes, labels, scores = [], [], []
        for map, limit in zip(maps, self.get_limits().values()):
            for box in map['boxes'].tolist():
                x1, y1, x2, y2 = box
                new_box = [x1 + limit.x_min, y1 + limit.y_min, x2 + limit.x_min, y2 + limit.y_min]
                boxes = [*boxes, new_box]

            labels = [*labels, *map['labels'].tolist()]
            scores = [*scores, *map['scores'].tolist()]

        return dict(boxes=torch.Tensor(boxes), labels=torch.Tensor(labels), scores=torch.Tensor(scores)) 
    
    def _reduce(self, map: dict) -> dict:
        if map['boxes'].nelement() == 0:
            return map
        else:
            indices = torchvision.ops.nms(map['boxes'], map['scores'], self.nms_threshold)
            reduced = dict(boxes=map['boxes'][indices], labels=map['labels'][indices], scores=map['scores'][indices]) 
            # score thresholding
            indices = torch.nonzero((reduced['scores'] > self.score_threshold), as_tuple=True)[0]
            reduced = dict(
                boxes=reduced['boxes'][indices], 
                labels=reduced['labels'][indices], 
                scores=reduced['scores'][indices]
                )
            
            return reduced

@STITCHERS.register()
class DensityMapStitcher(Stitcher):

    def __init__(
        self,
        model: torch.nn.Module, 
        size: Tuple[int,int], 
        overlap: int = 100,
        batch_size: int = 1,
        down_ratio: int = 2,
        adapt_ts: float = 0.0,
        reduction: str = 'mean',
        device_name: str = 'cuda',
        ) -> None:
        super().__init__(model, size, overlap=overlap, batch_size=batch_size, 
            down_ratio=down_ratio, reduction=reduction, device_name=device_name)
        
        self.adapt_ts = adapt_ts
        self.up = False
    
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

        patched_map = super(DensityMapStitcher, self).__call__(image)

        B, C, H, W = patched_map.shape
        
        # thresholding
        max_values = patched_map.max(3)[0].max(2)[0]
        thresholds = (max_values * self.adapt_ts).repeat(B, H, W, 1).permute(0,3,1,2)
        patched_map = patched_map * (patched_map > thresholds).float()
        # outputs = F.threshold(outputs, self.adapt_ts * max_value, 0.0)

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

        # 2D Hann windows matrix
        self.hann_matrix = self._make_hann_matrix()
        if len(patches) == 1:
            hann = HannWindow2D(size = self.size[0] // self.down_ratio)
            self.hann_matrix = [hann.get_window('original','up')]
        
        maps = []
        for patch, hann_2D in zip(dataloader, self.hann_matrix):
            patch = patch[0].to(self.device)
            outputs, _ = self.model(patch)

            # hann filter
            outputs = outputs * hann_2D.to(outputs.device)

            maps = [*maps, *outputs.unsqueeze(0)]

        return maps
    
    def _make_hann_matrix(self) -> list:

        hann = HannWindow2D(size = self.size[0] // self.down_ratio)

        first_row = [hann.get_window('edge', 'up')] * self._nrow
        first_row[0] = hann.get_window('corner', 'up_left')
        first_row[-1] = hann.get_window('corner', 'up_right')

        middle_row = [hann.get_window('original', 'up')] * self._nrow
        middle_row[0] = hann.get_window('edge', 'left')
        middle_row[-1] = hann.get_window('edge', 'right')

        last_row = [hann.get_window('edge', 'down')] * self._nrow
        last_row[0] = hann.get_window('corner', 'down_left')
        last_row[-1] = hann.get_window('corner', 'down_right')

        matrix = [*first_row, *middle_row * (self._ncol - 2), *last_row]

        return matrix
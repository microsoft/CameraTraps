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


import os
import PIL
import torch
import pandas
import numpy
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid, save_image

from typing import Union, Tuple

from tqdm import tqdm

from .types import BoundingBox

__all__ = ['ImageToPatches']

class ImageToPatches:
    ''' Class to make patches from a tensor image '''

    def __init__(
            self, 
            image: Union[PIL.Image.Image, torch.Tensor], 
            size: Tuple[int,int], 
            overlap: int = 0
        ) -> None:
        '''
        Args:
          image (PIL.Image.Image or torch.Tensor): image, if tensor: (C,H,W)
          size (tuple): patches size (height, width), in pixels
          overlap (int, optional): overlap between patches, in pixels. 
              Defaults to 0.
        '''

        assert isinstance(image, (PIL.Image.Image, torch.Tensor)), \
            'image must be a PIL.Image.Image or a torch.Tensor instance'

        self.image = image
        if isinstance(self.image, PIL.Image.Image):
            self.image = torchvision.transforms.ToTensor()(self.image)

        self.size = size
        self.overlap = overlap
    
    def make_patches(self) -> torch.Tensor:
        ''' Make patches from the image

        When the image division is not perfect, a zero-padding is performed 
        so that the patches have the same size.

        Returns:
            torch.Tensor:
                patches of shape (B,C,H,W)
        '''
        # patches' height & width
        height = min(self.image.size(1),self.size[0])
        width = min(self.image.size(2),self.size[1])

        # unfold on height 
        height_fold = self.image.unfold(1, height, height - self.overlap)

        # if non-perfect division on height
        residual = self._img_residual(self.image.size(1), height, self.overlap)
        if residual != 0:
            # get the residual patch and add it to the fold
            remaining_height = torch.zeros(3, 1, self.image.size(2), height) # padding
            remaining_height[:,:,:,:residual] = self.image[:,-residual:,:].permute(0,2,1).unsqueeze(1)

            height_fold = torch.cat((height_fold,remaining_height),dim=1)

        # unfold on width
        fold = height_fold.unfold(2, width, width - self.overlap)

        # if non-perfect division on width, the same
        residual = self._img_residual(self.image.size(2), width, self.overlap)
        if residual != 0:
            remaining_width = torch.zeros(3, fold.shape[1], 1, height, width) # padding
            remaining_width[:,:,:,:,:residual] = height_fold[:,:,-residual:,:].permute(0,1,3,2).unsqueeze(2)

            fold = torch.cat((fold,remaining_width),dim=2)

        self._nrow , self._ncol = fold.shape[2] , fold.shape[1]

        # reshaping
        patches = fold.permute(1,2,0,3,4).reshape(-1,self.image.size(0),height,width)

        return patches
    
    def get_limits(self) -> dict:
        ''' Get patches limits within the image frame

        When the image division is not perfect, the zero-padding is not
        considered here. Hence, the limits are the true limits of patches
        within the initial image.

        Returns:
            dict:
                a dict containing int as key and BoundingBox as value
        '''

        # patches' height & width
        height = min(self.image.size(1),self.size[0])
        width = min(self.image.size(2),self.size[1])

        # lists of pixels numbers
        y_pixels = torch.tensor(list(range(0,self.image.size(1)+1)))
        x_pixels = torch.tensor(list(range(0,self.image.size(2)+1)))

        # cut into patches to get limits
        y_pixels_fold = y_pixels.unfold(0, height+1, height-self.overlap)
        y_mina = [int(patch[0]) for patch in y_pixels_fold]
        y_maxa = [int(patch[-1]) for patch in y_pixels_fold]

        x_pixels_fold = x_pixels.unfold(0, width+1, width-self.overlap)
        x_mina = [int(patch[0]) for patch in x_pixels_fold]
        x_maxa = [int(patch[-1]) for patch in x_pixels_fold]

        # if non-perfect division on height
        residual = self._img_residual(self.image.size(1), height, self.overlap)
        if residual != 0:
            remaining_y = y_pixels[-residual-1:].unsqueeze(0)[0]
            y_mina.append(int(remaining_y[0]))
            y_maxa.append(int(remaining_y[-1]))

        # if non-perfect division on width  
        residual = self._img_residual(self.image.size(2), width, self.overlap)
        if residual != 0:
            remaining_x = x_pixels[-residual-1:].unsqueeze(0)[0]
            x_mina.append(int(remaining_x[0]))
            x_maxa.append(int(remaining_x[-1]))
        
        i = 0
        patches_limits = {}
        for y_min , y_max in zip(y_mina,y_maxa):
            for x_min , x_max in zip(x_mina,x_maxa):
                patches_limits[i] = BoundingBox(x_min,y_min,x_max,y_max)
                i += 1
         
        return patches_limits
    
    def show(self) -> None:
        ''' Show the grid of patches '''

        grid = make_grid(
            self.make_patches(),
            padding=50,
            nrow=self._nrow
            ).permute(1,2,0).numpy()

        plt.imshow(grid)

        plt.show()

        return grid
    
    def _img_residual(self, ims: int, ks: int, overlap: int) -> int:

        ims, stride = int(ims), int(ks - overlap)
        n = ims // stride
        end = n * stride + overlap
        
        residual = ims % stride

        if end > ims:
            n -= 1
            residual = ims - (n * stride)

        return residual
    
    def __len__(self) -> int:
        return len(self.get_limits())

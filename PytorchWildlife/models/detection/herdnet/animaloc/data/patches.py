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
from .annotations import Annotations, AnnotationsFromCSV, dict_from_objects
from .processors import object_processor

__all__ = ['save_batch_images', 'ImageToPatches', 'AnnotatedImageToPatches',
    'PatchesBuffer']


def save_batch_images(
    batch: torch.Tensor,
    basename: str,
    dest_folder: str
    ) -> None:
    ''' Save mini-batch tensors into image files

    Use torchvision save_image function,
    see https://pytorch.org/vision/stable/utils.html#torchvision.utils.save_image

    Args:
        batch (torch.Tensor): mini-batch tensor
        basename (str) : parent image name, with extension
        dest_folder (str): destination folder path
    '''

    base_wo_extension, extension = basename.split('.')[0], basename.split('.')[1]
    for i, b in enumerate(range(batch.shape[0])):
        full_path = '_'.join([base_wo_extension, str(i) + '.']) + extension
        save_path = os.path.join(dest_folder, full_path)
        save_image(batch[b], fp=save_path)


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


class AnnotatedImageToPatches(ImageToPatches):
    ''' Class to make annotated patches from an annotated image '''

    def __init__(
        self,  
        image: Union[PIL.Image.Image, torch.Tensor], 
        annotations: Annotations,
        size: Tuple[int,int], 
        overlap: int = 0
        ) -> None:
        '''
        Args:
            image (PIL.Image.Image or torch.Tensor): image, if tensor: (C,H,W)
            annotations (Annotations): Annotations class instance containing the
                image annotations
            size (tuple): patches size (height, width), in pixels
            overlap (int, optional): overlap between patches, in pixels. 
                Defaults to 0. 
        '''

        assert isinstance(annotations, Annotations), \
            'annotations argument must be an Annotations class instance'

        super(AnnotatedImageToPatches, self).__init__(image, size, overlap=overlap)
        self.annotations = annotations
    
    def make_annotated_patches(
        self, 
        min_visibility: float = 0.1
        ) -> Tuple[torch.Tensor, Annotations]:
        ''' Save annotated patches from an annotated image

        This method return the patches (torch.Tensor) and the new annotations 
        (Annotations object) with images attribute containing a string  
        representing the initial image's name, the patch index and the initial 
        image extension (for correspondence with potential patches saving using 
        save_batch_images function).

        Args:
            min_visibility (float, optional): minimum fraction of area for 
                an annotation to be kept. Defaults to 0.1.
        
        Returns:
            Tuple[torch.Tensor, Annotations]:
                patches (Tensor) and Annotations class instance
        '''

        patches = self.make_patches()
        annos = self.get_annotated_limits(min_visibility)
        delattr(annos, 'limits')
        
        return patches, annos
    
    def get_annotated_limits(self, min_visibility: float = 0.1) -> Annotations:
        ''' Get only annotated patches limits within the annotated image frame

        Args:
            min_visibility (float, optional): minimum fraction of area for 
                an annotation to be kept. Must be in [0, 0.99] Defaults to 0.1.

        Returns:
            Annotations:
                Annotations class instance representing the annotated patches
                limits only
        '''

        assert min_visibility >= 0.0 and min_visibility < 1.0, \
            'min_visibility argument must be between 0.0 and 0.99 included, ' \
                f'got {min_visibility}'
                
        all_new_annos = []
        # loop through patches limits
        for idx , limit in self.get_limits().items():

            # loop through annotations
            for anno_dict in self.annotations:

                old_anno = object_processor(anno_dict['annos'])
                new_anno = old_anno.intersect(limit)

                frac_area = new_anno.area / old_anno.area

                # check fraction of area to keep the annotation
                if frac_area > min_visibility:
                    # shift annotation coordinates to limits frame
                    new_anno = object_processor(new_anno)
                    new_anno_dict = dict_from_objects([new_anno.shift(limit)])[0]

                    # link all the remaining information
                    img_name , img_ext = tuple(anno_dict['images'].split('.'))
                    for key in ['annos','images']:
                        anno_dict.pop(key)

                    all_new_annos.append(dict(
                        images= f'{img_name}_{idx}.{img_ext}',
                        base_images=f'{img_name}.{img_ext}',
                        limits=limit,
                        **new_anno_dict, 
                        **anno_dict)
                        )

        all_new_annos_df = pandas.DataFrame(all_new_annos)

        if all_new_annos_df.empty is True:
            return []
        else:
            return AnnotationsFromCSV(all_new_annos_df)


class PatchesBuffer:
    ''' Create a data buffer with limits of images' patches and annotations 
    
    Allows to further create a torchvision.DataLoader without the need to save
    the patches.
    '''
    
    def __init__(
        self, 
        csv_file: str, 
        root_dir: str, 
        patch_size: Tuple[int, int], 
        overlap: int = 0, 
        min_visibility: float = 0.1
        ) -> None:
        '''
        Args:
            csv_file (str): absolute path to the CSV file containing images
                annotations
            root_dit (str): path to the folder containing the images
            patch_size (tuple): patches size (height, width), in pixels
            overlap (int, optional): overlap between patches, in pixels. 
                Defaults to 0.
            min_visibility (float, optional): minimum fraction of area for 
                an annotation to be kept. Defaults to 0.1.
        '''

        self.csv_file = csv_file 
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.min_visibility = min_visibility

        self.data = AnnotationsFromCSV(self.csv_file)

        self.buffer = self._create_buffer()
    
    def _create_buffer(self) -> pandas.DataFrame:
        ''' Create a pandas.DataFrame containing the patches data buffer,
        i.e. the annotations and limits of each patches (for memory saving).

        Allows to further create a torchvision.DataLoader without the need to save
        the patches.

        Returns:
            pandas.DataFrame:
                a DataFrame ordered by image name and containing annotations 
                in coordinates format, and patches limits
        '''

        buffer = []

        # loop through images name
        for img_name in tqdm(numpy.unique(self.data.images), desc='Creating the buffer'):
            
            img_path = os.path.join(self.root_dir, img_name)
            pil_image = PIL.Image.open(img_path).convert('RGB')

            # get corresponding annotations
            annos_obj = self.data.sub(img_name)

            # get patches annotations
            patches_annos = AnnotatedImageToPatches(
                image = pil_image, 
                annotations = annos_obj, 
                size = self.patch_size,
                overlap = self.overlap).get_annotated_limits(self.min_visibility)

            # iter through to store annotations in a list
            for patch_anno in patches_annos:
                new_patch_anno = patch_anno.copy()
                new_patch_anno.pop('annos')
                new_patch_anno.update(dict_from_objects([patch_anno['annos']])[0])
                buffer.append(new_patch_anno)

        return pandas.DataFrame(buffer)
    
    def __len__(self):
        return len(self.buffer)
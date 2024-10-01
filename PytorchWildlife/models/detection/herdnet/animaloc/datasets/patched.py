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

from typing import Optional, Tuple

from .register import DATASETS

from ..data.patches import PatchesBuffer
from .csv import CSVDataset

@DATASETS.register()
class PatchedDataset(CSVDataset):
    ''' Class to create a virtual patched CSVDataset from a CSV file 
    
    This dataset allows you to send cropped images (i.e. patches) from 
    large original images, without saving theses patches to a local hard 
    drive.

    These patches are generated in an orderly fashion using the 
    AnnotatedImageToPatches class internally.
    '''

    def __init__(
        self, 
        csv_file: str, 
        root_dir: str, 
        patch_size: Tuple[int, int], 
        overlap: Optional[int] = 0, 
        min_visibility: Optional[float] = 0.1,
        albu_transforms: Optional[list] = None,
        end_transforms: Optional[list] = None
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
            albu_transforms (list, optional): an albumentations' transformations 
                list that takes input sample as entry and returns a transformed 
                version. Defaults to None.
            end_transforms (list, optional): list of transformations that takes
                tensor and expected target as input and returns a transformed
                version. These will be applied after albu_transforms. Defaults
                to None.
        '''

        # create a buffer containing patches (dataframe)
        patches = PatchesBuffer(
            csv_file, 
            root_dir, 
            patch_size, 
            overlap, 
            min_visibility).buffer

        super(PatchedDataset, self).__init__(patches, root_dir, albu_transforms, end_transforms)

        self.patch_size = patch_size
        self.overlap = overlap
        self.min_visibility = min_visibility
    
    def _load_image(self, index: int) -> PIL.Image.Image:
        ''' Overwrite the original base class method to return patch 
        (i.e. cropped image) 
        '''

        # get the large image
        img_name = list(set(self.annotations.images))[index]
        base_image = list(self.data[self.data['images'] == img_name]['base_images'])[0]
        img_path = os.path.join(self.root_dir, base_image)
        img = PIL.Image.open(img_path).convert('RGB')

        # get the patch limits (BoundingBox object)
        patch_limits = list(self.data[self.data['images'] == img_name]['limits'])[0]

        return img.crop(patch_limits.get_tuple)
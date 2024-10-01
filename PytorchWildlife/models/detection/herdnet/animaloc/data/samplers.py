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

from torch.utils.data import Sampler
from typing import Iterable, Iterator

from ..datasets import CSVDataset
from ..data.utils import group_by_image

from ..utils.registry import Registry

SAMPLERS = Registry('samplers', module_key='animaloc.data.samplers')

__all__ = ['SAMPLERS', *SAMPLERS.registry_names]

@SAMPLERS.register()
class BinaryBatchSampler(Sampler):
    ''' Samples elements from two image-level categories (C0 and C1) and returns batches
    consisting of the same number of elements for each domain.
    
    The batch size must be even and the csv file on which the dataset has been 
    built must contain a column defining the two categories, i.e. C0 (0) and C1 (1).
    '''

    def __init__(
        self, 
        dataset: CSVDataset, 
        col: str, 
        batch_size: int = 2, 
        shuffle: bool = False,
        *args, **kwargs
        ) -> None:
        '''
        Args:
            dataset (CSVDataset): dataset from which to sample data. Must be a CSVDataset.
            col (str): dataset's DataFrame column defining categories C0 and C1.
            batch_size (int, optional): how many samples per batch to load. Defaults to 2.
            shuffle (bool, optional): set to True to have the data reshuffled at every epoch.
                Defaults to False.
        '''
        super().__init__(dataset)  

        if not isinstance(dataset, CSVDataset):
            raise TypeError(
                f"dataset should be an instance of 'CSVDataset' class, but got '{type(dataset)}'"
                )
        
        if batch_size % 2 != 0:
            raise ValueError(f"batch size should be even, but got {batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        df = self.dataset.data.copy()
        df = group_by_image(df)
        self.col = col
        if  self.col not in df.columns:
            raise ValueError(f"'{col}' column is missing from the csv file")

        self.n = self.batch_size // 2
        self.c0_idx = df.loc[df[col]==0].index.values.tolist()
        self.c1_idx = df.loc[df[col]==1].index.values.tolist()
        
        c0_idx, c1_idx = self._grouped(self.c0_idx, n=self.n), self._grouped(self.c1_idx, n=self.n)
        self.batch_idx = [[*c0, *c1] for c0, c1 in zip(c0_idx, c1_idx)]
    
    def __iter__(self) -> Iterator:

        if self.shuffle:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)

            c0_idx = [self.c0_idx[i] for i in torch.randperm(len(self.c0_idx), generator=generator)]
            c1_idx = [self.c1_idx[i] for i in torch.randperm(len(self.c1_idx), generator=generator)]
            c0_idx, c1_idx = self._grouped(c0_idx, n=self.n), self._grouped(c1_idx, n=self.n)
            batch_idx = [[*c0, *c1] for c0, c1 in zip(c0_idx, c1_idx)]

            yield from batch_idx
        
        else:
            yield from self.batch_idx
    
    def __len__(self) -> int:
        return len(self.batch_idx)
    
    def _grouped(self, iterable: Iterable, n: int) -> Iterable:
        return zip(*[iter(iterable)]*n)
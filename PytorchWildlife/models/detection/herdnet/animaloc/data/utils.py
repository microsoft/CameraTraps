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


import pandas
import numpy
import torch

from typing import Optional, Sequence


def group_by_image(df: pandas.DataFrame) -> pandas.DataFrame:
    ''' Group information by image and remove repeated rows '''

    df = df.groupby('images').first().reset_index()
    return df

def herding(df: pandas.DataFrame, size: int = 20) -> pandas.DataFrame:
    ''' Count objects per image and define a herd if the count is above 
    size argument. The dataframe must contain an 'images' column.
    
    Args:
        df (pandas.DataFrame): the dataframe containing annotations
        size (int, optional): size threshold. Defaults to 20.
    
    Returns:
        pandas.DataFrame
            the dataframe with "counts" and "herd" new columns
    '''

    counts = df['images'].value_counts().rename_axis('images').reset_index(name='counts')
    counts['herd'] = 0
    counts.loc[counts['counts'] >= size, 'herd'] = 1

    return pandas.DataFrame.merge(df, counts, on='images')

def weighted_samples(samples: Sequence[int], p: Optional[list] = None) -> torch.Tensor:
    ''' Create a tensor with a weight for each sample. Weights are
    computed according to the occurence of the sample in the given 
    sequence, by default. A list of probabilities can also be 
    specified.
    
    Args:
        sample (Sequence): a sequence of integers representing
            samples categories
        p (list, optional): a list of probabilities for each
            category. Defaults to None. 

    
    Returns:
        torch.Tensor
    '''

    class_sample_count = numpy.array(
        [len(numpy.where(samples == t)[0]) for t in numpy.unique(samples)]
        )

    weight = 1. / class_sample_count
    if p is not None:
        assert len(p) == len(numpy.unique(samples)), \
            'Number of probabilities specified must be equal to the number of category' \
            f', got {len(p)} and  {len(numpy.unique(samples))} respectively'
        
        weight = p

    samples_weight = numpy.array([weight[t] for t in samples])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    return samples_weight
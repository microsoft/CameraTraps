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
import matplotlib.pyplot as plt 
import random
import itertools

from typing import Optional
from torchvision.transforms import ToPILImage
from ..data.transforms import UnNormalize, GaussianMap

__all__ = ['PlotPrecisionRecall']

class PlotPrecisionRecall:

    def __init__(
        self,
        figsize: tuple = (7,7), 
        legend: bool = False, 
        seed: int = 1
        ) -> None:
        
        self.figsize = figsize
        self.legend = legend
        self.seed = seed

        self._data = []
        self._labels = []

    def feed(self, recalls: list, precisions: list, label: Optional[str] = None) -> None:
        # recalls.append(recalls[-1])
        # precisions.append(0)
        self._data.append((recalls, precisions))
        self._labels.append(label)
    
    def plot(self) -> None:
        
        random.seed(self.seed)
        colors = self._gen_colors(len(self._data))
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim(0,1.02)
        ax.set_ylim(0,1.02)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        markers = self._markers
        for i, (recall, precision) in enumerate(self._data):
            ax.plot(recall, precision,
                color=colors[i],
                marker=next(markers),
                markevery=0.1,
                alpha=0.7,
                label=self._labels[i])
        
        if self.legend:
            lg = plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
        
        self.fig = fig
    
    def save(self, path: str) -> None:
        if 'fig' not in self.__dict__:
            self.plot()

        self.fig.savefig(path, dpi=300, format='png', bbox_inches='tight')

    def _gen_colors(self, n: int) -> list:

        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            for i in range(n)]

        return colors
    
    @property
    def _markers(self) -> itertools.cycle:
        return itertools.cycle(('^','o','s','x','D','v','>'))
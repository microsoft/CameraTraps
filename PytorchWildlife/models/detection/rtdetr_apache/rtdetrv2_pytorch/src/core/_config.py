"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Callable


__all__ = ['BaseConfig', ]


class BaseConfig(object):

    def __init__(self) -> None:
        super().__init__()

        self.task :str = None 

        # instance / function 
        self._model :nn.Module = None 
        self._postprocessor :nn.Module = None 
        self._criterion :nn.Module = None 
        self._optimizer :Optimizer = None 
        self._lr_scheduler :LRScheduler = None 
        self._lr_warmup_scheduler: LRScheduler = None 
        self._train_dataloader :DataLoader = None 
        self._val_dataloader :DataLoader = None 
        self._ema :nn.Module = None 
        self._scaler :GradScaler = None 
        self._train_dataset :Dataset = None 
        self._val_dataset :Dataset = None
        self._collate_fn :Callable = None
        self._evaluator :Callable[[nn.Module, DataLoader, str], ] = None
        self._writer: SummaryWriter = None
        
        # dataset 
        self.num_workers :int = 0
        self.batch_size :int = None
        self._train_batch_size :int = None
        self._val_batch_size :int = None
        self._train_shuffle: bool = None  
        self._val_shuffle: bool = None 

        # runtime
        self.resume :str = None
        self.tuning :str = None 

        self.epoches :int = None
        self.last_epoch :int = -1

        self.use_amp :bool = False 
        self.use_ema :bool = False 
        self.ema_decay :float = 0.9999
        self.ema_warmups: int = 2000
        self.sync_bn :bool = False 
        self.clip_max_norm : float = 0.
        self.find_unused_parameters :bool = None

        self.seed :int = None
        self.print_freq :int = None 
        self.checkpoint_freq :int = 1
        self.output_dir :str = None
        self.summary_dir :str = None
        self.device : str = ''

    @property
    def model(self, ) -> nn.Module:
        return self._model 

    @property
    def postprocessor(self, ) -> nn.Module:
        return self._postprocessor
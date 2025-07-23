"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import copy

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict

class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg) 
        
        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self, ):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)
    
    @property
    def model(self, ) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        return super().model 

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return super().postprocessor
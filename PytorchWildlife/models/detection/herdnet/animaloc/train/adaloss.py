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

__all__ = ['Adaloss']

class Adaloss:

    def __init__(
        self, 
        param: torch.Tensor,
        w: int = 3, 
        rho: float = 0.9,
        delta_max: float = 5.
        ) -> None:
        
        assert isinstance(param, torch.Tensor), \
            f'param must be a torch.Tensor, got {type(param)}'

        assert isinstance(w, int) and w > 0, \
            'w must be a positive integer'
        
        assert 0. <= rho <= 1., \
            f'rho value must be between 0.0 and 1.0, got {rho}'
        
        self.param = param
        self.w = w
        self.rho = rho
        self.delta_max = delta_max

        self._step = 1
        self._losses = []
        self.loss_history = []
        self.var_history = []

        self.param_tracker = []
    
    def step(self) -> None:

        self._update_losses()
        self._update_vars()

        start = self.w

        if self._step > start:
            delta = self._cdelta()
            if abs(delta) < self.delta_max:
                self.param.add_(delta)
            elif delta < 0 :
                self.param.add_(- torch.tensor(self.delta_max))
            else:
                self.param.add_(torch.tensor(self.delta_max))
        
        self._step += 1
        self.param_tracker.append(torch.clone(self.param))
    
    def feed(self, loss: torch.Tensor) -> None:
        self._losses.append(loss)
    
    def _update_losses(self) -> None:
        self.loss_history.append(torch.tensor(self._losses))
        self._losses = []
    
    def _update_vars(self) -> None:
        window = torch.cat(self.loss_history[self._step - self.w : self._step])
        variance = torch.var(window)
        self.var_history.append(variance)
    
    def _cdelta(self):
        ratio = torch.div(self.var_history[-2], self.var_history[-1])
        return self.rho * torch.sub(1., ratio)
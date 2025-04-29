"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 

from ..misc import dist_utils
from ._solver import BaseSolver
from .det_engine import evaluate

class DetSolver(BaseSolver):
    
    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return

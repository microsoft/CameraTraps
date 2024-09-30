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
import pandas
import os
import numpy
import wandb
import matplotlib

matplotlib.use('Agg')

from typing import Any, Optional, Dict, List, Callable

import torch.nn.functional as F

from ..utils.logger import CustomLogger

from .stitchers import Stitcher
from .metrics import Metrics
from .lmds import HerdNetLMDS

from ..utils.registry import Registry

EVALUATORS = Registry('evaluators', module_key='animaloc.eval.evaluators')

__all__ = ['EVALUATORS', *EVALUATORS.registry_names]

@EVALUATORS.register()
class Evaluator:
    ''' Base class for evaluators '''

    def __init__(
        self,
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        metrics: Metrics,
        device_name: str = 'cuda', 
        print_freq: int = 10,
        stitcher: Optional[Stitcher] = None,
        vizual_fn: Optional[Callable] = None,
        work_dir: Optional[str] = None,
        header: Optional[str] = None
        ):
        '''
        Args:
            model (torch.nn.Module): CNN detection model to evaluate, that takes as 
                input tensor image and returns output and loss as tuple.
            dataloader (torch.utils.data.DataLoader): a pytorch's DataLoader that returns a tensor
                image and target.
            metrics (Metrics): Metrics instance used to compute model performances.
            device_name (str): the device name on which tensors will be allocated ('cpu' or  
                'cuda').
                Defaults to 'cuda'.
            print_freq (int, optional): define the frequency at which the logs will be
                printed and/or recorded. 
                Defaults to 10.
            stitcher (Stitcher, optional): optional Stitcher class instance to evaluate over
                large images. The specified dataloader should thus be composed of large images 
                for the use of this algorithm to make sense.
                Defaults to None.
            vizual_fn (callable, optional): a model specific function that will be use for plotting
                samples in Weights & Biases during validation. It must take 'image', 'target', and 
                'output' as arguments and return a matplotlib figure. Defaults to None.
            work_dir (str, optional): directory where logs (and results) will be saved. If
                None is given, results and logs files will be saved in current working 
                directory.
                Defaults to None.
            header (str, optional): string put at the beginning of the printed logs
                Defaults to None
        '''
        
        assert isinstance(model, torch.nn.Module), \
            'model argument must be an instance of nn.Module'
        
        assert isinstance(dataloader, torch.utils.data.DataLoader), \
            'dataset argument must be an instance of torch.utils.data.DataLoader'

        assert isinstance(metrics, Metrics), \
            'metrics argument must be an instance of Metrics'
        
        assert isinstance(stitcher, (type(None), Stitcher)), \
            'stitcher argument must be an instance of Stitcher class'
        
        assert callable(vizual_fn) or isinstance(vizual_fn, type(None)), \
            f'vizual_fn argument must be a callable function, got \'{type(vizual_fn)}\''
        
        self.model = model
        self.dataloader = dataloader
        self.metrics = metrics
        self.device = torch.device(device_name)
        self.print_freq = print_freq
        self.stitcher = stitcher
        self.vizual_fn = vizual_fn
        
        self.work_dir = work_dir
        if self.work_dir is None:
            self.work_dir = os.getcwd()

        self.header = header

        self._stored_metrics = None

        self.logs_filename = 'evaluation'

    def prepare_data(self, images: Any, targets: Any) -> tuple:
        ''' Method to prepare the data before feeding to the model. 
        Can be overriden by subclasses.

        Args:
            images (Any)
            targets (Any)
        
        Returns:
            tuple
        '''
        
        return images.to(self.device), targets.to(self.device)
    
    def prepare_feeding(self, targets: Any, output: Any) -> dict:
        ''' Method to prepare targets and output before feeding to the Metrics instance. 
        Can be overriden by subclasses.

        Args:
            targets (Any)
            output (Any)
        
        Returns:
            dict
        '''
        
        return dict(gt = targets, preds = output)
    
    def post_stitcher(self, output: torch.Tensor) -> Any:
        ''' Method to post-treat the output of the stitcher.
        Can be overriden by subclasses.

        Args:
            output (torch.Tensor): output of Stitcher call
        
        Returns:
            Any
        '''
        return output
    
    @torch.no_grad()
    def evaluate(self, returns: str = 'recall', wandb_flag: bool = False, viz: bool = False,
        log_meters: bool = True) -> float:
        ''' Evaluate the model
        
        Args:
            returns (str, optional): metric to be returned. Possible values are:
                'recall', 'precision', 'f1_score', 'mse', 'mae', 'rmse', 'accuracy'
                and 'mAP'. Defauts to 'recall'
            wandb_flag (bool, optional): set to True to log on Weight & Biases. 
                Defaults to False.
            viz (bool, optional): set to True to save vizual predictions on original
                images. Defaults to False.
            log_meters (bool, optional): set to False to disable meters logging. 
                Defaults to True.
        
        Returns:
            float
        '''
        self.model.eval()

        self.metrics.flush()

        logger = CustomLogger(delimiter=' ', filename=self.logs_filename, work_dir=self.work_dir)
        iter_metrics = self.metrics.copy()

        for i, (images, targets) in enumerate(logger.log_every(self.dataloader, self.print_freq, self.header)):
            # IMAGES IS A SINGLE TENSOR
            images, targets = self.prepare_data(images, targets)

            if self.stitcher is not None:
                output = self.stitcher(images[0])
                output = self.post_stitcher(output)
            else:
                # output, _ = self.model(images, targets)  
                output, _ = self.model(images)

            if viz and self.vizual_fn is not None:
                if i % self.print_freq == 0 or i == len(self.dataloader) - 1:
                    fig = self._vizual(image = images, target = targets, output = output)
                    wandb.log({'validation_vizuals': fig})

            output = self.prepare_feeding(targets, output)

            iter_metrics.feed(**output)
            iter_metrics.aggregate()
            if log_meters:
                logger.add_meter('n', sum(iter_metrics.tp) + sum(iter_metrics.fn))
                logger.add_meter('recall', round(iter_metrics.recall(),2))
                logger.add_meter('precision', round(iter_metrics.precision(),2))
                logger.add_meter('f1-score', round(iter_metrics.fbeta_score(),2))
                logger.add_meter('MAE', round(iter_metrics.mae(),2))
                logger.add_meter('MSE', round(iter_metrics.mse(),2))
                logger.add_meter('RMSE', round(iter_metrics.rmse(),2))

            if wandb_flag:
                wandb.log({
                    'n': sum(iter_metrics.tp) + sum(iter_metrics.fn),
                    'recall': iter_metrics.recall(),
                    'precision': iter_metrics.precision(),
                    'f1_score': iter_metrics.fbeta_score(),
                    'MAE': iter_metrics.mae(),
                    'MSE': iter_metrics.mse(),
                    'RMSE': iter_metrics.rmse()
                    })

            iter_metrics.flush()

            self.metrics.feed(**output)
        
        self._stored_metrics = self.metrics.copy()

        mAP = numpy.mean([self.metrics.ap(c) for c in range(1, self.metrics.num_classes)]).item()
        
        self.metrics.aggregate()

        if wandb_flag:
            wandb.run.summary['recall'] =  self.metrics.recall()
            wandb.run.summary['precision'] =  self.metrics.precision()
            wandb.run.summary['f1_score'] =  self.metrics.fbeta_score()
            wandb.run.summary['MAE'] =  self.metrics.mae()
            wandb.run.summary['MSE'] =  self.metrics.mse()
            wandb.run.summary['RMSE'] =  self.metrics.rmse()
            wandb.run.summary['accuracy'] =  self.metrics.accuracy()
            wandb.run.summary['mAP'] =  mAP
            wandb.run.finish()

        if returns == 'recall':
            return self.metrics.recall()
        elif returns == 'precision':
            return self.metrics.precision()
        elif returns == 'f1_score':
            return self.metrics.fbeta_score()
        elif returns == 'mse':
            return self.metrics.mse()
        elif returns == 'mae':
            return self.metrics.mae()
        elif returns == 'rmse':
            return self.metrics.rmse()
        elif returns == 'accuracy':
            return self.metrics.accuracy()
        elif returns == 'mAP':
            return mAP
    
    @property
    def results(self) -> pandas.DataFrame:
        ''' Returns metrics by class (recall, precision, f1_score, mse, mae, and rmse) 
        in a pandas dataframe '''
        
        assert self._stored_metrics is not None, \
            'No metrics have been stored, please use the evaluate method first.'
        
        metrics_cpy = self._stored_metrics.copy()
        
        res = []
        for c in range(1, metrics_cpy.num_classes):
            metrics = {
                'class': str(c),
                'n': metrics_cpy.tp[c-1] + metrics_cpy.fn[c-1],
                'recall': metrics_cpy.recall(c),
                'precision': metrics_cpy.precision(c),
                'f1_score': metrics_cpy.fbeta_score(c),
                'confusion': metrics_cpy.confusion(c), 
                'mae': metrics_cpy.mae(c),
                'mse': metrics_cpy.mse(c),
                'rmse': metrics_cpy.rmse(c),
                'ap': metrics_cpy.ap(c),
            }
            res.append(metrics)
        
        metrics_cpy.aggregate()
        res.append({
            'class': 'binary',
            'n': metrics_cpy.tp[0] + metrics_cpy.fn[0],
            'recall': metrics_cpy.recall(),
            'precision': metrics_cpy.precision(),
            'f1_score': metrics_cpy.fbeta_score(),
            'confusion': metrics_cpy.confusion(),
            'mae': metrics_cpy.mae(),
            'mse': metrics_cpy.mse(),
            'rmse': metrics_cpy.rmse(),
            'ap': metrics_cpy.ap()
        })

        return pandas.DataFrame(data = res)
    
    @property
    def detections(self) -> pandas.DataFrame:
        ''' Returns detections (image id, location, label and score) in a pandas
        dataframe '''
        
        assert self._stored_metrics is not None, \
            'No detections have been stored, please use the evaluate method first.'

        img_names = self.dataloader.dataset._img_names
        dets = self._stored_metrics.detections
        for det in dets:
            det['images'] = img_names[det['images']]

        return pandas.DataFrame(data = dets)
    
    def _vizual(self, image: Any, target: Any, output: Any):
        fig = self.vizual_fn(image=image, target=target, output=output)
        return fig

@EVALUATORS.register()
class HerdNetEvaluator(Evaluator):

    def __init__(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, metrics: Metrics, 
        lmds_kwargs: dict = {'kernel_size': (3,3)}, device_name: str = 'cuda', print_freq: int = 10, 
        stitcher: Optional[Stitcher] = None, vizual_fn: Optional[Callable] = None, work_dir: Optional[str] = None, 
        header: Optional[str] = None
        ) -> None:
        super().__init__(model, dataloader, metrics, device_name=device_name, print_freq=print_freq, 
            vizual_fn=vizual_fn, stitcher=stitcher, work_dir=work_dir, header=header)

        self.lmds_kwargs = lmds_kwargs

    def prepare_data(self, images: Any, targets: Any) -> tuple:        
        return images.to(self.device), targets
    
    def post_stitcher(self, output: torch.Tensor) -> Any:
        heatmap = output[:,:1,:,:]
        clsmap = output[:,1:,:,:]
        return heatmap, clsmap

    def prepare_feeding(self, targets: Dict[str, torch.Tensor], output: List[torch.Tensor]) -> dict:

        gt_coords = [p[::-1] for p in targets['points'].squeeze(0).tolist()]
        gt_labels = targets['labels'].squeeze(0).tolist()
        
        gt = dict(
            loc = gt_coords,
            labels = gt_labels
        )

        up = True
        if self.stitcher is not None:
            up = False
        lmds = HerdNetLMDS(up=up, **self.lmds_kwargs)
        counts, locs, labels, scores, dscores = lmds(output)
        
        preds = dict(
            loc = locs[0],
            labels = labels[0],
            scores = scores[0],
            dscores = dscores[0]
        )
        
        return dict(gt = gt, preds = preds, est_count = counts[0])

@EVALUATORS.register()
class DensityMapEvaluator(Evaluator):
  
    def prepare_data(self, images: Any, targets: Any) -> tuple:        
        return images.to(self.device), targets

    def prepare_feeding(self, targets: Dict[str, torch.Tensor], output: torch.Tensor) -> dict:

        gt_coords = [p[::-1] for p in targets['points'].squeeze(0).tolist()]
        gt_labels = targets['labels'].squeeze(0).tolist()
        
        gt = dict(loc = gt_coords, labels = gt_labels)
        preds = dict(loc = [], labels = [], scores = [])

        _, idx = torch.max(output, dim=1)
        masks = F.one_hot(idx, num_classes=output.shape[1]).permute(0,3,1,2)
        output = (output * masks)
        est_counts = output[0].sum(2).sum(1).tolist()
        
        return dict(gt = gt, preds = preds, est_count = est_counts)

@EVALUATORS.register()
class FasterRCNNEvaluator(Evaluator):

    def prepare_data(self, images: List[torch.Tensor], targets: List[dict]) -> tuple:
        images = list(image.to(self.device) for image in images)    
        targets = [{k: v.to(self.device) for k, v in t.items() if torch.is_tensor(v)} 
                            for t in targets]
        return images, targets
    
    def post_stitcher(self, output: dict) -> list:
        return [output]
    
    def prepare_feeding(self, targets: List[dict], output: List[dict]) -> dict:

        targets, output = targets[0], output[0]

        gt = dict(
            loc = targets['boxes'].tolist(),
            labels = targets['labels'].tolist()
            )
        
        preds = dict(
            loc = output['boxes'].tolist(),
            labels = output['labels'].tolist(),
            scores = output['scores'].tolist()
            )
        
        num_classes = self.metrics.num_classes - 1
        counts = [preds['labels'].count(i+1) for i in range(num_classes)]

        return dict(gt = gt, preds = preds, est_count = counts)
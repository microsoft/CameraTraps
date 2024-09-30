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
import math
import sys
import os
import wandb
import matplotlib

import matplotlib.pyplot as plt
matplotlib.use('Agg')
from torchvision.transforms import ToPILImage

from typing import List, Optional, Union, Callable, Any

from ..utils.torchvision_utils import SmoothedValue, reduce_dict
from ..utils.logger import CustomLogger
from ..eval.evaluators import Evaluator
from ..data.transforms import UnNormalize
from .adaloss import Adaloss

from ..utils.registry import Registry

TRAINERS = Registry('trainers', module_key='animaloc.train.trainers')

__all__ = ['TRAINERS', *TRAINERS.registry_names]

@TRAINERS.register()
class Trainer:
    ''' Base class for training a model '''

    def __init__(
        self, 
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer, 
        num_epochs: int, 
        lr_milestones: Optional[List[int]] = None,  
        auto_lr: Union[bool, dict] = False,
        adaloss: Optional[str] = None,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        evaluator: Optional[Evaluator] = None,
        vizual_fn: Optional[Callable] = None,
        work_dir: Optional[str] = None, 
        device_name: str = 'cuda', 
        print_freq: int = 50,
        valid_freq: int = 1,
        csv_logger: bool = False
        ) -> None:
        '''
        Args:
            model (torch.nn.Module): CNN model to train, that takes as 
                inputs image and target and returns a dict loss for training, and both output
                and dict loss for evaluation.
            train_dataloader (torch.utils.data.DataLoader): a pytorch's DataLoader used for 
                model training that combines a dataset and a sampler, and provides an iterable 
                over the given dataset. 
                see https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
            optimizer (torch.optim.Optimizer): a pytorch's optimization algorithm.
                see https://pytorch.org/docs/stable/optim.html
            num_epochs (int): number of epochs for training the model
            lr_milestones (list, optional): learning rate (lr) milestones representing a list
                of epoch indices (must be increasing) to build a lr scheduler that decays 
                the learning rate of each parameter group by 0.1 once the number of epoch 
                reaches one of the milestones. 
                Defaults to None.
            auto_lr (bool, optional): set to True to use the Pytorch's ReduceLROnPlateau scheduler
                with default parameters values. If specified, cancels the use of LR milestones 
                (i.e. MultiStepLR). If a dict is specified, it must contain some of the scheduler's 
                parameters with the parameter name as key, and the associated value.
                Defaults to False.
            adaloss (str, optional): specify the name of a dataset's end-transform parameter to be
                updated during training. This use "Adaloss", an objective function that adapts itself
                during the training by updating a parameter based on the training statistics (see
                https://arxiv.org/abs/1908.01070).
                Note that the dataset must contain an "update_end_transform()" method and the parameter
                chosen must be stored in a "end_params" attribute.
                Defaults to None.
            val_dataloader (torch.utils.data.DataLoader, optional): a pytorch's DataLoader used for 
                model validation that combines a dataset and a sampler, and provides an iterable 
                over the given dataset. 
                Defaults to None.
            evaluator (Evaluator, optional): if specified, used for evaluation.
                Override the default evaluator.
                Defaults to None.
            vizual_fn (callable, optional): a model specific function that will be use for plotting
                samples in Weights & Biases during validation. It must take 'image', 'target', and 
                'output' as arguments and return a matplotlib figure. Defaults to None.
            work_dir (str, optional): directory where checkpoints and logs will be saved. If None
                is given, results and logs files will be saved in current working directory.
                Defaults to None.
            device (str): the device name on which tensors will be allocated ('cpu' or 'cuda'). 
                Defaults to 'cuda'.
            print_freq (int, optional): define the frequency at which the logs will be
                printed and/or recorded. 
                Defaults to 50.
            valid_freq (int, optional): define the frequency at which the model will be validated.
                Note that first and last epoch are always validated.
                Defaults to 1 (i.e., after each epoch).
            csv_logger (bool, optional): set to True to store logs in a CSV file. Warning, long
                training session might slow down the process.
                Defaults to False.
        '''

        assert isinstance(model, torch.nn.Module), \
            f'model argument must be an instance of nn.Module(), ' \
                f'got \'{type(model)}\''
        
        assert isinstance(train_dataloader, torch.utils.data.DataLoader), \
            f'train_dataloader argument must be an instance of ' \
                f'torch.utils.data.DataLoader(), got \'{type(train_dataloader)}\''
        
        assert isinstance(val_dataloader, (torch.utils.data.DataLoader, type(None))), \
            f'val_dataloader argument must be an instance of ' \
                f'torch.utils.data.DataLoader(), got \'{type(val_dataloader)}\''
        
        assert isinstance(optimizer, torch.optim.Optimizer), \
            f'optimizer argument must be an instance of ' \
                f'torch.optim.Optimizer(), got \'{type(optimizer)}\''
        
        assert isinstance(lr_milestones, (list, type(None))), \
            f'lr_milestones argument must be a list, got \'{type(lr_milestones)}\''
        
        assert isinstance(auto_lr, (bool, dict)), \
            f'auto_lr argument must be a bool or a dict, got \'{type(auto_lr)}\''

        assert isinstance(evaluator, (Evaluator, type(None))), \
            f'evaluator argument must be an instance of Evaluator class, ' \
                f'got \'{type(evaluator)}\''
        
        assert callable(vizual_fn) or isinstance(vizual_fn, type(None)), \
            f'vizual_fn argument must be a callable function, got \'{type(vizual_fn)}\''
        
        assert valid_freq <= num_epochs, \
            'validation frequency must be lower or equal to the number of epochs'
        
        self.device = torch.device(device_name)

        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.epochs = num_epochs
        
        self.print_freq = print_freq
        self.valid_freq = valid_freq
        self.lr_milestones = lr_milestones
        self.evaluator = evaluator

        self.vizual_fn = vizual_fn

        # auto-learning rate reduction
        self.auto_lr = auto_lr
        self.auto_lr_flag = False
        if auto_lr or isinstance(auto_lr, dict):
            self.auto_lr_flag = True
        
        # adaloss
        self.adaloss = adaloss
        if isinstance(adaloss, str):
            assert 'end_params' in dir(self.train_dataloader.dataset), \
                'end_params attribute is missing from the training dataset'
            assert 'update_end_transforms' in dir(self.train_dataloader.dataset), \
                'update_end_transforms method is missing from the training dataset'
            assert adaloss in self.train_dataloader.dataset.end_params.keys(), \
                'Adaloss specified parameter is missing from the training dataset ' \
                    'end-transforms parameters'
            
            self.adaparam = adaloss
            self.adaloss = Adaloss(
                train_dataloader.dataset.end_params[adaloss], w=3, delta_max=1)

        # working directory
        self.work_dir = work_dir
        if self.work_dir is None:
            self.work_dir = os.getcwd()

        # loggers
        self.csv_logger = csv_logger
        self.train_logger = CustomLogger(delimiter=' ', filename='training', work_dir=self.work_dir, csv=self.csv_logger)
        self.val_logger = CustomLogger(delimiter=' ', filename='validation', work_dir=self.work_dir, csv=self.csv_logger)
    
    def prepare_data(self, images, targets) -> tuple:
        ''' Method to prepare the data before feeding to the model. 
        Can be override by subclass to create a custom Trainer.

        Args:
            images,
            targets
        
        Returns:
            tuple
        '''

        images = images.to(self.device)

        if isinstance(targets, (list, tuple)):
            targets = [tar.to(self.device) for tar in targets]
        else:
            targets = targets.to(self.device)

        return images, targets

    def start(
        self, 
        warmup_iters: Optional[int] = None, 
        checkpoints: str = 'best',
        select: str = 'min',
        validate_on: str = 'all',
        wandb_flag: bool = False
        ) -> torch.nn.Module:
        ''' Start training from epoch 1 
        
        Args:
            warmup_iters (int, optional): number of iterations to warm up the
                training, i.e. gradually increase the learning rate to reach its
                initial specified value.
            checkpoints (str, optional): mode for saving the checkpoints. Possible 
                values are:
                    - 'best' (default), to save the best checkpoint (based on validation
                        output),
                    - 'all', to save all the checkpoints.
                If no validation, all checkpoints are saved.
                Defaults to 'best'.
            select (str, optional): best epoch selection mode, used if 'checkpoints' is set
                to best. Possible values are:
                    - 'min' (default), for selecting the epoch that yields to a minimum validation value,
                    - 'max', for selecting the epoch that yields to a maximum validation value. 
                Defaults to 'min'.
            validate_on (str, optional): metrics/loss used for validation (i.e. best model and auto-lr).
                For validation with losses, possible values are the names returned by the model, or 'all'
                for using the sum of all losses (default). Possible values for evaluator are: 'recall', 
                'precision', 'f1_score', 'mse', 'mae', 'rmse', 'accuracy' or 'mAP'. 
                Defauts to 'all'
            wandb_flag (bool, optional): set to True to log on Weight & Biases. Defaults to False.
        
        Returns:
            torch.nn.Module:
                trained model
        '''

        assert checkpoints in ['best', 'all']
        assert select in ['min', 'max']

        lr_scheduler = self._lr_scheduler()
        val_flag = False

        if select =='min': 
            self.best_val = float('inf')
        elif select =='max': 
            self.best_val = 0
        
        if wandb_flag:
            wandb.log({'lr': self.optimizer.param_groups[0]["lr"]})

        for epoch in range(1,self.epochs + 1):

            # training
            train_output = self._train(epoch, warmup_iters, wandb_flag)
            if wandb_flag:
                wandb.log({'train_loss': train_output, 'epoch': epoch})
                wandb.log({'lr': self.optimizer.param_groups[0]["lr"]})

            # validation
            if epoch % self.valid_freq == 0 or epoch in [1, self.epochs]:

                if self.evaluator is not None:
                    val_flag = True
                    viz = False
                    if wandb_flag: viz = True
                    self._prepare_evaluator('validation', epoch)
                    val_output = self.evaluator.evaluate(returns=validate_on, viz=viz)
                    print(f'{self.evaluator.header} {validate_on}: {val_output:.4f}')

                    if wandb_flag:
                        wandb.log({validate_on: val_output, 'epoch': epoch})

                elif self.val_dataloader is not None:
                    val_flag = True
                    val_output = self.evaluate(epoch, wandb_flag=wandb_flag, returns=validate_on)
                    if wandb_flag:
                        wandb.log({'val_loss': val_output, 'epoch': epoch})
            
                # save checkpoint(s)
                if val_flag and checkpoints =='best' and self._is_best(val_output, mode = select):
                    print('Best model saved - Epoch {} - Validation value: {:.6f}'.format(epoch, val_output))
                    self._save_checkpoint(epoch, checkpoints)
                elif checkpoints == 'all':
                    self._save_checkpoint(epoch, checkpoints)
            
            self._save_checkpoint(epoch, 'latest')

            # scheduler
            if lr_scheduler is not None:
                if self.auto_lr_flag:
                    lr_scheduler.step(val_output)
                else:
                    lr_scheduler.step()
            
            # adaloss
            if self.adaloss is not None:
                self.adaloss.step()
                self.train_dataloader.dataset.load_end_param(self.adaparam, self.adaloss.param)
                print('Adaloss param: {}'.format(self.train_dataloader.dataset.end_params[self.adaparam]))
                self.train_dataloader.dataset.update_end_transforms()
                self.val_dataloader.dataset.end_params = self.train_dataloader.dataset.end_params
                self.val_dataloader.dataset.update_end_transforms()
        
        if wandb_flag:
            wandb.run.summary['best_validation'] = self.best_val
            wandb.run.finish()
        
        return self.model
    
    def resume(
        self, 
        pth_path: str, 
        checkpoints: str = 'best',
        select: str = 'min',
        validate_on: str = 'recall',
        load_optim: bool = False,
        wandb_flag: bool = False
        ) -> torch.nn.Module:
        ''' Resume training from a pth file 
        
        Args:
            pth_path (str): absolute path to the checkpoint (i.e. pth file)
            checkpoints (str, optional): mode for saving the checkpoints. Possible 
                values are:
                    - 'best' (default), to save the best checkpoint (based on validation
                        output),
                    - 'all', to save all the checkpoints.
                If no validation, all checkpoints are saved.
                Defaults to 'best'.
            select (str, optional): best epoch selection mode, used if 'checkpoints' is set
                to best. Possible values are:
                    - 'min' (default), for selecting the epoch that yields to a minimum validation value,
                    - 'max', for selecting the epoch that yields to a maximum validation value. 
                Defaults to 'min'.
            validate_on (str, optional): metrics used for validation (i.e. best model and auto-lr) when 
                custom evaluator is specified. Possible values are: 'recall', 'precision', 'f1_score', 
                'mse', 'mae', 'rmse' and 'mAP'. 
                Defauts to 'recall'
            load_optim (bool, optional): set to True to load the optimizer's state_dict.
                Defaults to False
            wandb_flag (bool, optional): set to True to log on Weight & Biases. Defaults to False.
        
        Returns:
            torch.nn.Module:
                trained model
        '''

        assert checkpoints in ['best', 'all']
        assert select in ['min', 'max']

        checkpoint = torch.load(pth_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optim is True:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        resume_epoch = checkpoint['epoch']
        self.losses = checkpoint['loss']

        self.best_val = checkpoint['best_val']

        lr_scheduler = self._lr_scheduler()
        val_flag = False

        if wandb_flag:
            wandb.log({'lr': self.optimizer.param_groups[0]["lr"]})

        for epoch in range(resume_epoch + 1, self.epochs + 1):

            # training
            train_output = self._train(epoch, wandb_flag=wandb_flag) 
            if wandb_flag:
                wandb.log({'train_loss': train_output, 'epoch': epoch})
                wandb.log({'lr': self.optimizer.param_groups[0]["lr"]})

            # validation
            if epoch % self.valid_freq == 0 or epoch in [1, self.epochs]:

                if self.evaluator is not None:
                    val_flag = True
                    viz = False
                    if wandb_flag: viz = True
                    self._prepare_evaluator('validation', epoch)
                    val_output = self.evaluator.evaluate(returns=validate_on, viz=viz)
                    print(f'{self.evaluator.header} {validate_on}: {val_output:.4f}')

                    if wandb_flag:
                        wandb.log({validate_on: val_output, 'epoch': epoch})

                elif self.val_dataloader is not None:
                    val_flag = True
                    val_output = self.evaluate(epoch, wandb_flag=wandb_flag, returns=validate_on)
                    if wandb_flag:
                        wandb.log({'val_loss': val_output, 'epoch': epoch})
                
                # save checkpoint(s)
                if val_flag and checkpoints =='best' and self._is_best(val_output, mode = select):
                    print('Best model saved - Epoch {} - Validation value: {:.6f}'.format(epoch, val_output))
                    self._save_checkpoint(epoch, checkpoints)
                elif checkpoints == 'all':
                    self._save_checkpoint(epoch, checkpoints)
            
            self._save_checkpoint(epoch, 'latest')

            # scheduler
            if lr_scheduler is not None:
                if self.auto_lr_flag:
                    if 'val_output' in locals():
                        lr_scheduler.step(val_output)
                    else:
                        lr_scheduler.step(self.best_val)
                else:
                    lr_scheduler.step()
            
            # adaloss
            if self.adaloss is not None:
                self.adaloss.step()
                self.train_dataloader.dataset.load_end_param(self.adaparam, self.adaloss.param)
                self.train_dataloader.dataset.update_end_transforms()
                self.val_dataloader.dataset.end_params = self.train_dataloader.dataset.end_params
                self.val_dataloader.dataset.update_end_transforms()
        
        if wandb_flag:
            wandb.run.summary['best_validation'] = self.best_val
            wandb.run.finish()

        return self.model
    
    @torch.no_grad()
    def evaluate(self, epoch: int, reduction: str = 'mean', wandb_flag: bool = False, returns: str = 'all') -> float:
        
        self.model.eval()

        header = '[VALIDATION] - Epoch: [{}]'.format(epoch)

        batches_losses = []

        for i, (images, targets) in enumerate(self.val_logger.log_every(self.val_dataloader, self.print_freq, header)):

            images, targets = self.prepare_data(images, targets)

            output, loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            if returns != 'all':
                losses = loss_dict[returns]

            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            self.val_logger.update(loss=losses_reduced, **loss_dict_reduced)

            batches_losses.append(losses)

            if wandb_flag and self.vizual_fn is not None:
                if (i % self.print_freq == 0 or i == len(self.val_dataloader) - 1):
                    fig = self._vizual(image = images, target = targets, output = output)
                    wandb.log({'validation_vizuals': fig})
        
        batches_losses = torch.stack(batches_losses)
        
        if reduction == 'mean':
            out = torch.mean(batches_losses).item()
            print(f'{header} mean loss: {out:.4f}')

            return out
        
        elif reduction == 'sum':
            out = torch.sum(batches_losses).item()
            print(f'{header} sum loss: {out:.4f}')

            return out

    def _train(
        self, 
        epoch: int, 
        warmup_iters: Optional[int] = None, 
        wandb_flag: bool = False
        ) -> torch.Tensor:
        ''' Training method '''

        self.model.train()

        self.train_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = '[TRAINING] - Epoch: [{}]'.format(epoch)

        if warmup_iters is not None and epoch == 1:
            self.start_lr_scheduler = self._warmup_lr_scheduler(
                min(warmup_iters, len(self.train_dataloader)-1), 
                1. / warmup_iters
                )

        batches_losses = []

        for images, targets in self.train_logger.log_every(self.train_dataloader, self.print_freq, header):

            images, targets = self.prepare_data(images, targets)

            self.optimizer.zero_grad()

            loss_dict = self.model(images, targets)

            if wandb_flag:
                wandb.log(loss_dict)

            self.losses = sum(loss for loss in loss_dict.values())
            batches_losses.append(self.losses.detach())

            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            self.losses.backward()
            self.optimizer.step()

            if self.adaloss is not None:
                self.adaloss.feed(self.losses)

            if warmup_iters is not None and epoch == 1:
                self.start_lr_scheduler.step()

            self.train_logger.update(loss=losses_reduced, **loss_dict_reduced)
            self.train_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        
        batches_losses = torch.stack(batches_losses)

        out = torch.mean(batches_losses).item()
        print(f'{header} mean loss: {out:.4f}')

        return out
    
    def _warmup_lr_scheduler(self, warmup_iters: int, warmup_factor: float):
        ''' Method to make a warmup lr scheduler '''

        def warmup_func(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_func)
    
    def _lr_scheduler(self):
        ''' Method to make the lr scheduler '''

        if self.auto_lr is True:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

        elif isinstance(self.auto_lr, dict):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.auto_lr)

        elif self.lr_milestones is not None:
            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.lr_milestones)
        
        else:
            return None
    
    def _prepare_evaluator(self, filename: str, epoch: int) -> None:
        ''' Evaluate the epoch model '''

        if self.evaluator is not None:
            self.evaluator.model = self.model
            self.evaluator.logs_filename = filename
            self.evaluator.header = '[{}] - Epoch: [{}]'.format(filename.upper(),epoch)
    
    def _is_best(self, val_output: float, mode: str = 'min') -> bool:
        ''' Method to determine the best model for saving checkpoint '''
        
        if mode == 'min':
            if val_output < self.best_val:
                self.best_val = val_output
                return True
            else:
                return False
        
        elif mode =='max':
            if val_output > self.best_val:
                self.best_val = val_output
                return True
            else:
                return False
    
    def _save_checkpoint(self, epoch: int, mode: str) -> None:
        ''' Method to save checkpoints '''

        check_dir = self.work_dir

        if mode == 'all':
            outpath = os.path.join(check_dir,f'epoch_{epoch}.pth')
        elif mode == 'best':
            outpath = os.path.join(check_dir,'best_model.pth')
        elif mode == 'latest':
            outpath = os.path.join(check_dir,'latest_model.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses ,
            'best_val': self.best_val
            }, outpath)
    
    def _vizual(self, image: Any, target: Any, output: Any):
        fig = self.vizual_fn(image=image, target=target, output=output)
        return fig

@TRAINERS.register()
class FasterRCNNTrainer(Trainer):
    ''' Class for training a Faster-RCNN model '''

    def prepare_data(self, images, targets) -> tuple:

        images = list(image.to(self.device) for image in images)
            
        targets = [{k: v.to(self.device) for k, v in t.items() if torch.is_tensor(v)} 
                            for t in targets]

        return images, targets
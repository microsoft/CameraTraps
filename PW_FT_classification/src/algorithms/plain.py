import os
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import random

import torch
import torch.optim as optim
import pytorch_lightning as pl

from .utils import acc
from src import models


__all__ = [
    'Plain'
]

class Plain(pl.LightningModule):
    """
    Defines the architecture for training a model using PyTorch Lightning.

    This class inherits from PyTorch Lightning's LightningModule and sets up the model, optimizers,
    and training/validation/testing steps for the training process.
    """

    name = 'Plain'

    def __init__(self, conf, train_class_counts, id_to_labels, **kwargs):
        """
        Initializes the Plain model.

        Args:
            conf: Configuration object with model parameters.
            train_class_counts: Counts of training classes.
            id_to_labels: Mapping from IDs to label names.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.hparams.update(conf.__dict__)
        self.save_hyperparameters(ignore=['conf', 'train_class_counts'])
        self.train_class_counts = train_class_counts
        self.id_to_labels = id_to_labels
        self.net = models.__dict__[self.hparams.model_name](num_cls=self.hparams.num_classes, 
                                                            num_layers=self.hparams.num_layers)

    def configure_optimizers(self):
        """
        Configures the optimizers and learning rate schedulers.

        Returns:
            Tuple[List, List]: A tuple containing the list of optimizers and the list of learning rate schedulers.
        """
        # Define parameters for the optimizer
        net_optim_params_list = [
            # Optimizer parameters for feature extraction
            {'params': self.net.feature.parameters(),
             'lr': self.hparams.lr_feature,
             'momentum': self.hparams.momentum_feature,
             'weight_decay': self.hparams.weight_decay_feature},
            # Optimizer parameters for the classifier
            {'params': self.net.classifier.parameters(),
             'lr': self.hparams.lr_classifier,
             'momentum': self.hparams.momentum_classifier,
             'weight_decay': self.hparams.weight_decay_classifier}
        ]
        # Setup optimizer and optimizer scheduler
        optimizer = torch.optim.SGD(net_optim_params_list)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)   
        return [optimizer], [scheduler]

    def on_train_start(self):
        """
        Hook function called at the start of training. Initializes best accuracy and the network.
        """
        self.best_acc = 0
        self.net.feat_init()
        self.net.setup_criteria()

    def training_step(self, batch, batch_idx):
        """
        Training step for each batch.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            Tensor: The loss for the current training step.
        """
        data, label_ids = batch[0], batch[1]
        
        # Forward pass
        feats = self.net.feature(data)
        logits = self.net.classifier(feats)
        # Calculate loss
        loss = self.net.criterion_cls(logits, label_ids)
        self.log("train_loss", loss)
        
        return loss

    def on_validation_start(self):
        """
        Hook function called at the start of validation. Initializes storage for validation outputs.
        """
        self.val_st_outs = []

    def validation_step(self, batch, batch_idx):
        """
        Validation step for each batch.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.
        """
        data, label_ids = batch[0], batch[1]
        # Forward pass
        feats = self.net.feature(data)
        logits = self.net.classifier(feats)
        preds = logits.argmax(dim=1)
        
        self.val_st_outs.append((preds.detach().cpu().numpy(),
                                 label_ids.detach().cpu().numpy()))

    def on_validation_epoch_end(self):
        """
        Hook function called at the end of the validation epoch. Aggregates and logs validation results.
        """
        total_preds = np.concatenate([x[0] for x in self.val_st_outs], axis=0)
        total_label_ids = np.concatenate([x[1] for x in self.val_st_outs], axis=0)
        self.eval_logging(total_preds, total_label_ids)

    def on_test_start(self):
        """
        Hook function called at the start of testing. Initializes storage for test outputs.
        """
        self.te_st_outs = []

    def test_step(self, batch, batch_idx):
        """
        Test step for each batch.

        Args:
            batch: The current batch of data, including metadata.
            batch_idx: The index of the current batch.
        """
        data, label_ids, labels, file_ids = batch
        # Forward pass
        feats = self.net.feature(data)
        logits = self.net.classifier(feats)
        preds = logits.argmax(dim=1)
        
        self.te_st_outs.append((preds.detach().cpu().numpy(),
                               label_ids.detach().cpu().numpy(),
                               feats.detach().cpu().numpy(),
                               logits.detach().cpu().numpy(), 
                               labels, file_ids 
                               ))
    

    def on_test_epoch_end(self):
        """
        Hook function called at the end of the test epoch. Aggregates and logs test results, and saves output.
        """
        # Concatenate outputs from all test steps
        total_preds = np.concatenate([x[0] for x in self.te_st_outs], axis=0)
        total_label_ids = np.concatenate([x[1] for x in self.te_st_outs], axis=0)
        total_feats = np.concatenate([x[2] for x in self.te_st_outs], axis=0)
        total_logits = np.concatenate([x[3] for x in self.te_st_outs], axis=0)
        total_labels = np.concatenate([x[4] for x in self.te_st_outs], axis=0)
        total_file_ids = np.concatenate([x[5] for x in self.te_st_outs], axis=0)

        # Calculate the metrics and save the output
        self.eval_logging(total_preds[total_label_ids != -1],
                          total_label_ids[total_label_ids != -1],
                          print_class_acc=False)

        output_path = self.hparams.evaluate.replace('.ckpt', 'eval.npz') 
        np.savez(output_path, preds=total_preds, label_ids=total_label_ids, feats=total_feats,
                 logits=total_logits, labels=total_labels, file_ids=total_file_ids)  
        print('Test output saved to {}.'.format(output_path))

    def on_predict_start(self):
        """
        Hook function called at the start of prediction. Initializes storage for prediction outputs.
        """
        self.pr_st_outs = []

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for each batch.

        Args:
            batch: The current batch of data, including metadata.
            batch_idx: The index of the current batch.
        """
        data, file_ids = batch
        # Forward pass
        feats = self.net.feature(data)
        logits = self.net.classifier(feats)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1).max(dim=1)[0]
        
        self.pr_st_outs.append((preds.detach().cpu().numpy(),
                                feats.detach().cpu().numpy(),
                                logits.detach().cpu().numpy(), 
                                probs.detach().cpu().numpy(),
                                file_ids 
                                ))
    

    def on_predict_epoch_end(self):
        """
        Hook function called at the end of the predict epoch. Aggregates and saves prediction outputs.
        """
        # Concatenate outputs from all predict steps
        total_preds = np.concatenate([x[0] for x in self.pr_st_outs], axis=0)
        total_feats = np.concatenate([x[1] for x in self.pr_st_outs], axis=0)
        total_logits = np.concatenate([x[2] for x in self.pr_st_outs], axis=0)
        total_probs = np.concatenate([x[3] for x in self.pr_st_outs], axis=0)
        total_file_ids = np.concatenate([x[4] for x in self.pr_st_outs], axis=0)

        json_output = []
        for i in range(len(total_preds)):
            json_output.append({
                "marker_id": "",
                "survey_pic_id": total_file_ids[i],
                "marker_confidence": float(total_probs[i]),
                "marker_gear_type": "ghostnet" if total_preds[i] == 1 else "neg",
                "marker_bounding_polygon": "",
                "marker_status": "unverified",
                "marker_ai_model": ""
            })

        output_path_full = self.hparams.evaluate.replace('.ckpt', '_predict.npz') 
        np.savez(output_path_full, preds=total_preds, feats=total_feats,
                 logits=total_logits, file_ids=total_file_ids)  
        print('Predict output saved to {}.'.format(output_path_full))

        output_path_json = self.hparams.evaluate.replace('.ckpt', '_predict.json') 
        json.dump(json_output, open(output_path_json, 'w'))
        print('Predict output json saved to {}.'.format(output_path_json))


    def eval_logging(self, preds, labels, print_class_acc=False):
        """
        Logs evaluation metrics such as accuracy.

        Args:
            preds: Predictions from the model.
            labels: Ground truth labels.
            print_class_acc (bool): Flag to print class-wise accuracy.
        """
        class_acc, mac_acc, mic_acc = acc(preds, labels)
        unique_eval_labels = np.unique(labels)

        self.log("valid_mac_acc", mac_acc * 100)
        self.log("valid_mic_acc", mic_acc * 100)

        if print_class_acc:

            if self.train_class_counts:
                acc_list = [(class_acc[i], unique_eval_labels[i],
                             self.id_to_labels[unique_eval_labels[i]],
                             self.train_class_counts[unique_eval_labels[i]])
                             for i in range(len(class_acc))]

                print('\n')
                for i in range(len(class_acc)):
                    info = '{:>20} ({:<3}, tr {:>3}) Acc: '.format(acc_list[i][2],
                                                                   acc_list[i][1],
                                                                   acc_list[i][3])
                    info += '{:.2f}'.format(acc_list[i][0] * 100)
                    print(info)
            else:
                acc_list = [(class_acc[i], unique_eval_labels[i],
                             self.id_to_labels[unique_eval_labels[i]])
                             for i in range(len(class_acc))]

                print('\n')
                for i in range(len(class_acc)):
                    info = '{:>20} ({:<3}) Acc: '.format(acc_list[i][2], acc_list[i][1])
                    info += '{:.2f}'.format(acc_list[i][0] * 100)
                    print(info)

#
# Engine.py
#
# Training driver
#

#%% Constants and imports

from .utils import *

import torch
import time
import sys
import numpy as np

#%% Utility functions

def log(total,current,t,l,epoch=-1,top1=None,top5=None):
    if epoch!=-1:
        print('Epoch %d'%(epoch), end =" ")
    if top1 is not None and top5 is not None:
        print('Batch [%d/%d]\t'
              'Time %.3f %.3f\t'
              'Loss %.4f %.4f\t'
              'Acc@1 %.3f %.3f\t'
              'Acc@5 %.3f %.3f'%(current, total, t.val, t.avg, l.val, l.avg, top1.val, top1.avg, top5.val, top5.avg))
    else:
        print('Batch [%d/%d]\t'
              'Time %.3f %.3f\t'
              'Loss %.4f %.4f'%(current, total, t.val, t.avg, l.val, l.avg))
    sys.stdout.flush()


#%% Training engine
    
class Engine():

    def __init__(self,model,criterion,optimizer, verbose=False, print_freq=10, progressBar=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.verbose = verbose
        self.print_freq = print_freq
        self.progressBar = progressBar

    def train_one_batch(self, input,target,iter_num,calcAccuracy):
        
        start = time.time()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = self.model(input)

        # measure accuracy and record loss
        if calcAccuracy:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss = self.criterion(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        end = time.time()
        if self.progressBar:
          self.progressBar.setValue(iter_num)
        if not calcAccuracy:
            return loss.item(),end-start
        else:
            return loss.item(),end-start,acc1,acc5

    def train_one_epoch(self, train_loader, epoch_num, calcAccuracy):
        batch_time = AverageMeter()
        losses = AverageMeter()
        if calcAccuracy:
            top1 = AverageMeter()
            top5 = AverageMeter()

    	# switch to train mode
        self.model.train()
        for i, batch in enumerate(train_loader):
            input= batch[0]
            target= batch[1]
            # measure accuracy and record loss
            if calcAccuracy:
                loss,t,acc1,acc5= self.train_one_batch(input,target,i,True)
                top1.update(acc1,input.size(0))
                top5.update(acc5,input.size(0))
            else:
                loss,t= self.train_one_batch(input,target,i,False)
            losses.update(loss, input.size(0))
            batch_time.update(t)
            if self.verbose and i % self.print_freq == 0:
                if calcAccuracy:
                    log(len(train_loader), i, batch_time, losses, epoch=epoch_num, top1=top1, top5=top5)
                else:
                    log(len(train_loader), i, batch_time, losses, epoch=epoch_num)

    def validate_one_batch(self, input, target, iter_num, calcAccuracy):
        with torch.no_grad():
            start = time.time()

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

	    # compute output
            output = self.model(input)

	    # measure accuracy and record loss
            if calcAccuracy:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss = self.criterion(output, target)

            end = time.time()
            if self.progressBar:
              self.progressBar.setValue(iter_num)

            if not calcAccuracy:
                return loss.item(),end-start
            else:
                return loss.item(),end-start,acc1,acc5

    def validate(self, val_loader, calcAccuracy):
        batch_time = AverageMeter()
        losses = AverageMeter()
        if calcAccuracy:
            top1 = AverageMeter()
            top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        for i, batch in enumerate(val_loader):
            input = batch[0]
            target = batch[1]
            if calcAccuracy:
                loss,t,acc1,acc5= self.validate_one_batch(input,target,i, True)
                top1.update(acc1,input.size(0))
                top5.update(acc5,input.size(0))
            else:
                loss,t= self.validate_one_batch(input,target,i, False)
            losses.update(loss, input.size(0))
            batch_time.update(t)

            if self.verbose and i % self.print_freq == 0:
                if calcAccuracy:
                    log(len(val_loader), i, batch_time, losses, top1=top1, top5=top5)
                else:
                    log(len(val_loader), i, batch_time, losses)
        if calcAccuracy:
            log(len(val_loader), i, batch_time, losses, top1=top1, top5=top5)
        else:
            log(len(val_loader), i, batch_time, losses)
 
        return losses.avg

    def predict_one_batch(self, input, iter_num):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            # compute output
            output = self.model(input)
        if self.progressBar:
           self.progressBar.setValue(iter_num)

        return output

    def predict(self, dataloader, load_info= False, dim=256):

        # switch to evaluate mode
        self.model.eval()
        embeddings = np.zeros((len(dataloader.dataset), dim), dtype=np.float32)
        labels = np.zeros(len(dataloader.dataset), dtype=np.int64)
        if load_info:
          paths=[None]*len(dataloader.dataset)
        k = 0
        for i, batch in enumerate(dataloader):
            images=batch[0]
            target= batch[1]
            if load_info:
                paths[k:k+len(batch[2])]=batch[2]
            embedding= self.predict_one_batch(images,i)
            embeddings[k:k+len(images)] = embedding.data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
            if self.verbose and i % self.print_freq == 0:
              print("Batch %d"%(i))
        if load_info:
            return embeddings, labels, paths
        else:
            return embeddings, labels


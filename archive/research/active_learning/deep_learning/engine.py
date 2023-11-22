from .utils import *

import time
import sys
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# utility function
def generate_description(desc, loss, top1 = None, top5 = None):
    if top1 is not None and top5 is not None:
        return '%s Avg. Loss: %.4f\tAvg. Top-1 Acc.: %.3f\tAvg. Top-5 Acc.: %.3f'%(desc, loss.avg, top1.avg, top5.avg)
    else:
        return '%s Avg. Loss: %.4f'%(desc, loss.avg)

class Engine():

    def __init__(self, device, model, criterion, optimizer):
        self.model= model.to(device)
        self.criterion= criterion
        if self.criterion is not None:
            self.criterion.to(device)
        self.optimizer= optimizer
        self.device = device

    def train_one_batch(self, input, target, iter_num, calcAccuracy):

        input = input.to(self.device)
        target = target.to(self.device)
        # compute output
        output, _ = self.model(input)

        # measure accuracy and record loss
        if calcAccuracy:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss = self.criterion(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if not calcAccuracy:
            return loss.item()
        else:
            return loss.item(), acc1, acc5

    def train_one_epoch(self, train_loader, epoch_num, calcAccuracy):
        losses = AverageMeter()
        if calcAccuracy:
            top1 = AverageMeter()
            top5 = AverageMeter()

	# switch to train mode
        self.model.train()
        train_loader = tqdm(train_loader)
        for i, batch in enumerate(train_loader):
            input= batch[0]
            target= batch[1]

	    # measure accuracy and record loss
            if calcAccuracy:
                loss, acc1, acc5= self.train_one_batch(input, target, i, True)
                top1.update(acc1, input.size(0))
                top5.update(acc5, input.size(0))
            else:
                loss= self.train_one_batch(input, target, i, False)
            losses.update(loss, input.size(0))
            if calcAccuracy:
                train_loader.set_description(generate_description("Training epoch %d:"%epoch_num, losses, top1 = top1, top5 = top5))
            else:
                train_loader.set_description(generate_description("Training epoch %d:"%epoch_num, losses))

    def validate_one_batch(self, input, target, iter_num, calcAccuracy):
        with torch.no_grad():
            input = input.to(self.device)
            target = target.to(self.device)

	    # compute output
            output, _ = self.model(input)

	    # measure accuracy and record loss
            if calcAccuracy:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss = self.criterion(output, target)

            if not calcAccuracy:
                return loss.item()
            else:
                return loss.item(), acc1, acc5

    def validate(self, val_loader, calcAccuracy):
        losses = AverageMeter()
        if calcAccuracy:
            top1 = AverageMeter()
            top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        val_loader = tqdm(val_loader)
        for i, batch in enumerate(val_loader):
            input= batch[0]
            target= batch[1]
            if calcAccuracy:
                loss, acc1, acc5= self.validate_one_batch(input, target, i, True)
                top1.update(acc1,input.size(0))
                top5.update(acc5,input.size(0))
            else:
                loss = self.validate_one_batch(input, target, i, False)
            losses.update(loss, input.size(0))

            if calcAccuracy:
                val_loader.set_description(generate_description("Validation:", losses, top1=top1, top5=top5))
            else:
                val_loader.set_description(generate_description("Validation:", losses))
 
        return losses.avg

    def predict_one_batch(self, input, iter_num):
        with torch.no_grad():
            input = input.to(self.device)
            # compute output
            output, _ = self.model(input)

        return output

    def predict(self, dataloader, load_info= False, dim=256):

        # switch to evaluate mode
        self.model.eval()
        embeddings = np.zeros((len(dataloader.dataset), dim), dtype=np.float32)
        labels = np.zeros(len(dataloader.dataset), dtype=np.int64)
        if load_info:
          paths=[None]*len(dataloader.dataset)
        k = 0
        dataloader = tqdm(dataloader, desc = "Prediction:")
        for i, batch in enumerate(dataloader):
            images=batch[0]
            target= batch[1]
            if load_info:
                paths[k:k+len(batch[2])]=batch[2]
            embedding= self.predict_one_batch(images,i)
            embeddings[k:k+len(images)] = embedding.data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
        if load_info:
            return embeddings, labels, paths
        else:
            return embeddings, labels

    def embedding_one_batch(self, input, iter_num):
        with torch.no_grad():
            input = input.to(self.device)
            # compute output
            _, output = self.model(input)

        return output

    def embedding(self, dataloader, dim=256, normalize = False):

        # switch to evaluate mode
        self.model.eval()
        embeddings = np.zeros((len(dataloader.dataset), dim), dtype=np.float32)
        k = 0
        dataloader = tqdm(dataloader, desc = "Embedding:")
        for i, batch in enumerate(dataloader):
            images=batch[0]
            embedding= self.embedding_one_batch(images,i)
            embeddings[k:k+len(images)] = embedding.data.cpu().numpy()
            k += len(images)
        if normalize:
            scaler = MinMaxScaler()
            return scaler.fit_transform(embeddings)
        else:
            return embeddings

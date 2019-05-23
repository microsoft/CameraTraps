import argparse
import os
import random
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR

import numpy as np

from DL.data_loader import BaseDataLoader
from DL.losses import *
from DL.utils import *
from DL.networks import *
from DL.Engine import Engine

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--train_data', metavar='DIR',
                    help='path to train dataset', default='../../crops_train')
parser.add_argument('--val_data', metavar='DIR',
                    help='path to validation dataset', default=None)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--checkpoint_prefix', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--plot_freq', dest='plot_freq', type= int, action='store',
                    help='plot embedding frequence', default=1)
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--loss_type', default='triplet', 
                    help='Loss Type')
parser.add_argument('--margin', default=1.0, type=float, metavar='M',
                    help='margin for siamese or triplet loss')
parser.add_argument('-f', '--feat_dim', default=256, type=int,
                    metavar='N', help='embedding size (default: 256)')

parser.add_argument('--raw_size', nargs= 2, default= [256,256], type= int, action= 'store', help= 'The width, height and number of channels of images for loading from disk')
parser.add_argument('--processed_size', nargs= 2, default= [224,224], type= int, action= 'store', help= 'The width and height of images after preprocessing')
parser.add_argument('--balanced_P', default= -1, type= int, action= 'store', help= 'The number of classes in each balanced batch')
parser.add_argument('--balanced_K', default= 10, type= int, action= 'store', help= 'The number of examples from each class in each balanced batch')
best_acc1 = 0

def custom_policy(step):
  details_str= '21, 36, 46, 0.001, 0.0005, 0.0001, 0.00005'
  details= [float(x) for x in details_str.split(",")]
  length = len(details)
  for i,x in enumerate(details[0:int((length-1)/2)]):
    if step<= int(x):
      return details[int((length-1)/2)+i]
  return details[-1]

def adjust_lr(optimizer, step):
  param= custom_policy(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = param
  print("Learning rate is set to %f"%param)

def main():
    global args, best_acc1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    checkpoint={}
    if args.resume!='':
      checkpoint= load_checkpoint(args.resume)
      args.loss_type= checkpoint['loss_type']
      args.feat_dim= checkpoint['feat_dim']
      best_accl= checkpoint['best_acc1']

    train_dataset = BaseDataLoader(args.train_data,True, batch_size= args.batch_size, num_workers= args.workers, raw_size= args.raw_size, processed_size= args.processed_size)
    if args.val_data is not None:
        val_dataset = BaseDataLoader(args.val_data,False, batch_size= args.batch_size, num_workers= args.workers)
    num_classes= len(train_dataset.getClassesInfo()[0])
    if args.balanced_P==-1:
      args.balanced_P= num_classes
    print("Num Classes= "+str(num_classes))
    if args.loss_type.lower()=='center':
      train_loader = train_dataset.getSingleLoader()
      train_embd_loader= train_loader
      if args.val_data is not None:
          val_loader = val_dataset.getSingleLoader()
          val_embd_loader= val_loader
    else:
      train_loader = train_dataset.getBalancedLoader(P=args.balanced_P, K=args.balanced_K)
      train_embd_loader= train_dataset.getSingleLoader()
      if args.val_data is not None:
          val_loader = val_dataset.getBalancedLoader(P=args.balanced_P, K=args.balanced_K)
          val_embd_loader = val_dataset.getSingleLoader()

    embedding_net = EmbeddingNet(args.arch, args.feat_dim, args.pretrained)
    center_loss= None
    if args.loss_type.lower()=='center':
      model = torch.nn.DataParallel(ClassificationNet(embedding_net, n_classes=num_classes)).cuda()
      criterion= CenterLoss(num_classes= num_classes, feat_dim= args.feat_dim)
      params = list(model.parameters()) + list(criterion.parameters())
    else:
      model= torch.nn.DataParallel(embedding_net).cuda()
      if args.loss_type.lower()=='siamese':
        criterion= OnlineContrastiveLoss(args.margin, RandomNegativePairSelector())
      else:
        criterion= OnlineTripletLoss(args.margin, SemihardNegativeTripletSelector(args.margin))
      params = model.parameters()

    # define loss function (criterion) and optimizer
    #optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay= args.weight_decay)
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay= args.weight_decay, momentum= 0.9)
    start_epoch= 0

    if checkpoint:
      start_epoch= checkpoint['epoch']
      model.load_state_dict(checkpoint['state_dict'])
      #optimizer.load_state_dict(checkpoint['optimizer'])
      if args.loss_type.lower()=='center':
        criterion.load_state_dict(checkpoint['centers'])

    e= Engine(model,criterion,optimizer, verbose= True, print_freq= args.print_freq)
    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        adjust_lr(optimizer, epoch)
        e.train_one_epoch(train_loader, epoch, True if args.loss_type.lower()=='center' else False)
        # evaluate on validation set
        if args.val_data is not None:
            e.validate(val_loader,True if args.loss_type.lower()=='center' else False)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'loss_type' : args.loss_type,
            'feat_dim' : args.feat_dim,
            'centers': criterion.state_dict() if args.loss_type.lower()=='center' else None
        }, False, "%s%s_%s_%04d.tar"%(args.checkpoint_prefix, args.loss_type, args.arch,epoch))


if __name__ == '__main__':
    main()

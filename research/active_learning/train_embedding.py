import argparse
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

from deep_learning.data_loader import ExtendedImageFolder
from deep_learning.losses import OnlineTripletLoss, OnlineContrastiveLoss
from deep_learning.utils import HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector, HardNegativePairSelector
from deep_learning.utils import load_checkpoint, save_checkpoint, plot_embedding, save_embedding_plot, getCriterion
from deep_learning.networks import NormalizedEmbeddingNet, SoftmaxNet, models
from deep_learning.engine import Engine

ARCHITECTURES = sorted(name for name in models.__dict__
                       if name.islower() and not name.startswith("__")
                       and callable(models.__dict__[name]))
LOSS_TYPES = ['softmax', 'triplet', 'siamese']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument(
    '--train_data', help='path to train dataset', default='crops')
parser.add_argument(
    '--val_data', help='path to validation dataset', default='smalls')
parser.add_argument('--arch', '-a', default='resnet18', choices=ARCHITECTURES,
                    help='model architecture: ' + ' | '.join(ARCHITECTURES) + ' (default: resnet18)')
parser.add_argument('-j', '--num_workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256,
                    type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.00001,
                    type=float, help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0.0005,
                    type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint_prefix', default='',
                    type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--plot_freq', dest='plot_freq', type=int,
                    help='plot embedding frequence', default=1)
parser.add_argument('--pretrained', dest='pretrained',
                    action='store_false', help='use pre-trained model')
parser.add_argument('--loss_type', default='triplet', choices=LOSS_TYPES,
                    help='loss type: ' + ' | '.join(LOSS_TYPES) + ' (default: triplet loss)')
parser.add_argument('--margin', default=1.0, type=float,
                    help='margin for siamese or triplet loss')
parser.add_argument('-f', '--feat_dim', default=256,
                    type=int, help='embedding size (default: 256)')
parser.add_argument('--raw_size', nargs=2, default=[256, 256], type=int,
                    help='The width, height and number of channels of images for loading from disk')
parser.add_argument('--processed_size', nargs=2, default=[224, 224], type=int,
                    help='The width and height of images after preprocessing')
parser.add_argument('--balanced_P', default=-1, type=int,
                    help='The number of classes in each balanced batch')
parser.add_argument('--balanced_K', default=10, type=int,
                    help='The number of examples from each class in each balanced batch')
parser.add_argument('--strategy', default='random', choices=['hardest', 'random', 'semi_hard', 'hard_pair'],
                    help='The number of examples from each class in each balanced batch')


def main():
    args = parser.parse_args()
    print(args)

    # Load a checkpoint if necessary
    checkpoint = {}
    if args.resume != '':
        checkpoint = load_checkpoint(args.resume)
        args.loss_type = checkpoint['loss_type']
        args.feat_dim = checkpoint['feat_dim']

    # setup the training dataset and the validation dataset
    train_dataset = ExtendedImageFolder(args.train_data)
    if args.val_data is not None:
        val_dataset = ExtendedImageFolder(args.val_data)
    
    num_classes = len(train_dataset.classes)
    if args.balanced_P == -1:
        args.balanced_P = num_classes

    # setup data loaders
    if args.loss_type.lower() == 'softmax':
        train_loader = train_dataset.getSingleLoader(batch_size = 128, shuffle = True, num_workers = args.num_workers)
        train_embd_loader = train_loader
        if args.val_data is not None:
            val_loader = val_dataset.getSingleLoader(batch_size = 128, shuffle = False, num_workers = args.num_workers, transfm = 'val')
            val_embd_loader = val_loader
    else:
        train_loader = train_dataset.getBalancedLoader(P = args.balanced_P, K = args.balanced_K, num_workers = args.num_workers)
        train_embd_loader = train_dataset.getSingleLoader(num_workers=args.num_workers)
        if args.val_data is not None:
            val_loader = val_dataset.getBalancedLoader(P = args.balanced_P, K = args.balanced_K, num_workers = args.num_workers, transfm = 'val')
            val_embd_loader = val_dataset.getSingleLoader(batch_size = 128, shuffle = False, num_workers = args.num_workers, transfm = 'val')
    
    # check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE: ", device)

    # create a model
    if args.loss_type.lower() == 'softmax':
        model = torch.nn.DataParallel(SoftmaxNet(
            args.arch, args.feat_dim, num_classes, use_pretrained=args.pretrained))
    else:
        model = torch.nn.DataParallel(NormalizedEmbeddingNet(
            args.arch, args.feat_dim, use_pretrained=args.pretrained))

    # setup loss criterion
    criterion = getCriterion(args.loss_type, args.strategy, args.margin)

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1

    # load a checkpoint if provided
    if checkpoint:
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    # setup a deep learning engine and start running
    e = Engine(device, model, criterion, optimizer)
    # train the model
    for epoch in range(start_epoch, args.epochs + 1):
        # train for one epoch
        e.train_one_epoch(train_loader, epoch, True if args.loss_type.lower() == 'softmax' else False)
        if epoch % args.plot_freq == 0 and epoch > 0:
            a, b, _ = e.predict(
                train_embd_loader, load_info=True, dim=args.feat_dim)
            save_embedding_plot("%s_train_%d.jpg"%(args.checkpoint_prefix, epoch), a, b, {})
            a, b, _ = e.predict(
                val_embd_loader, load_info=True, dim=args.feat_dim)
            save_embedding_plot("%s_val_%d.jpg"%(args.checkpoint_prefix, epoch), a, b, {})
        # evaluate on validation set
        if args.val_data is not None:
            e.validate(val_loader, True if args.loss_type.lower()
                       == 'softmax' else False)
        # save a checkpoint
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_type': args.loss_type,
            'num_classes': num_classes,
            'feat_dim': args.feat_dim
        }, False, "%s%s_%s_%04d.tar" % (args.checkpoint_prefix, args.loss_type, args.arch, epoch))


if __name__ == '__main__':
    main()

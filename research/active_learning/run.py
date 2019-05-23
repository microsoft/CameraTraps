import argparse
import os
import random
import time
import sys
import warnings
from shutil import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib
from scipy import stats
from itertools import count
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.extmath import softmax

from DL.sqlite_data_loader import SQLDataLoader
from DL.losses import *
from DL.utils import *
from DL.networks import *
from DL.Engine import Engine
from UIComponents.DBObjects import *

from sampling_methods.constants import get_AL_sampler
from sampling_methods.constants import get_wrapper_AL_mapping
get_wrapper_AL_mapping()

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import silhouette_samples, confusion_matrix
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.externals.joblib import parallel_backend
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.externals import joblib

from torch.utils.data import TensorDataset, DataLoader

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--run_data', metavar='DIR',
                    help='path to train dataset', default='../../NACTI_crops')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--base_model', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--experiment_name', default='', type=str, metavar='PATH',
                    help='prefix name for output files')
parser.add_argument('-N', '--active_batch', default=100, type=int,
                    help='number of queries per batch')
parser.add_argument('-A', '--active_budget', default= 25000, type=int,
                    help='number of queries per batch')
parser.add_argument('--K', default=5, type=int,
                    help='number of clusters')
parser.add_argument('--strategy', default='confidence', type=str,
                    help='Active learning strategy')
parser.add_argument('--limit', default=500000, type=int,
                    help='number of records to read')
parser.add_argument('--num_classes', default=50, type=int,
                    help='number of species')


def moveRecords(dataset, srcKind, destKind, rList):
      for e in rList:
          dataset.set_indices[srcKind].remove(e)
          dataset.set_indices[destKind].append(e)

def finetune_embedding(model, loss_type, train_dataset, P, K, epochs):
    train_dataset.image_mode()

    if loss_type.lower() == 'softmax':
        criterion = nn.CrossEntropyLoss().cuda()
        train_loader = train_dataset.getSingleLoader()
    elif loss_type.lower() == 'siamese':
        criterion = OnlineContrastiveLoss(1, HardNegativePairSelector())
        train_loader = train_dataset.getBalancedLoader(P = P, K = K)
    else:
        criterion = OnlineTripletLoss(1, RandomNegativeTripletSelector(1))
        train_loader = train_dataset.getBalancedLoader(P = P, K = K)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr = 0.0001)#, weight_decay = 0.0005)
    e= Engine(model, criterion, optimizer, verbose = True, print_freq = 10)
    for epoch in range(epochs):
        e.train_one_epoch(train_loader, epoch, False)

def main():
    args = parser.parse_args()
    print(args)
    print("DB Connect")
    db_path = os.path.join(args.run_data, os.path.basename(args.run_data)) + ".db"
    print(db_path)
    db = SqliteDatabase(db_path)
    proxy.initialize(db)
    db.connect()
    print("connected")
    print("CompleteLoop")
    
    checkpoint = load_checkpoint(args.base_model)
    if args.experiment_name == '':
        args.experiment_name = "experiment_%s_%s"%(checkpoint['loss_type'], args.strategy)
    if not os.path.exists(args.experiment_name):
        os.mkdir(args.experiment_name)

    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)

    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    dataset_query =  Detection.select(Detection.image_id,Oracle.label,Detection.kind).join(Oracle).order_by(fn.random()).limit(args.limit)
    dataset = SQLDataLoader('/lscratch/datasets/serengeti/resized', query=dataset_query, is_training= False, kind = DetectionKind.ModelDetection.value, num_workers= 8, limit = args.limit)
    #dataset = SQLDataLoader(os.path.join(args.run_data, "crops"), is_training= False, kind = DetectionKind.ModelDetection.value, num_workers= 8, limit = args.limit)
    dataset.updateEmbedding(model)
    #print('Embedding Done')
    #sys.stdout.flush()
    #plot_embedding(dataset.em[dataset.current_set], np.asarray(dataset.getlabels()) , dataset.getpaths(), {})
    # Random examples to start
    #random_ids = np.random.choice(dataset.current_set, 1000, replace=False).tolist()
    #random_ids = selectSamples(dataset.em[dataset.current_set], dataset.current_set, 2000)
    #print(random_ids)
    # Move Records
    #moveRecords(dataset, DetectionKind.ModelDetection.value, DetectionKind.UserDetection.value, random_ids)

    #print([len(x) for x in dataset.set_indices])
    # Finetune the embedding model
    #dataset.setKind(DetectionKind.UserDetection.value)
    #dataset.train()
    #train_dataset = SQLDataLoader(trainset_query, os.path.join(args.run_data, 'crops'), is_training= True)
    #finetune_embedding(model, checkpoint['loss_type'], dataset, 32, 4, 100)
    #save_checkpoint({
    #        'arch': model.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer' : optimizer.state_dict(),
    #        'loss_type' : loss_type,
    #        }, False, "%s%s_%s_%04d.tar"%('finetuned', loss_type, model.arch, len(dataset.set_indices[DetectionKind.UserDetection.value])))

    #unlabeled_dataset.updateEmbedding(model)
    #dataset.updateEmbedding(model)
    #dataset.setKind(DetectionKind.UserDetection.value)
    #print(dataset.em[dataset.current_set].shape, np.asarray(dataset.getlabels()).shape, len(dataset.getpaths()))
    #plot_embedding( dataset.em[dataset.current_set], np.asarray(dataset.getlabels()) , dataset.getpaths(), {})
    #plot_embedding( unlabeled_dataset.em, np.asarray(unlabeled_dataset.getlabels()) , unlabeled_dataset.getIDs(), {})
    dataset.embedding_mode()
    dataset.train()
    sampler = get_AL_sampler(args.strategy)(dataset.em, dataset.getalllabels(), 12)

    kwargs = {}
    kwargs["N"] = args.active_batch
    kwargs["already_selected"] = dataset.set_indices[DetectionKind.UserDetection.value]
    kwargs["model"] = MLPClassifier(alpha=0.0001)

    print("start the loop")
    sys.stdout.flush()
    numLabeled = len(dataset.set_indices[DetectionKind.UserDetection.value])
    while numLabeled <= args.active_budget:
        print([len(x) for x in dataset.set_indices])
        sys.stdout.flush()
        # Active Learning
        if numLabeled == 0:
            indices = np.random.choice(dataset.current_set, 1000, replace=False).tolist()
        else:
            indices = sampler.select_batch(**kwargs)
        numLabeled = len(dataset.set_indices[DetectionKind.UserDetection.value])
        #kwargs["already_selected"].extend(indices)
        moveRecords(dataset, DetectionKind.ModelDetection.value, DetectionKind.UserDetection.value, indices)

        dataset.setKind(DetectionKind.UserDetection.value)
        X_train= dataset.em[dataset.current_set]
        y_train= np.asarray(dataset.getlabels())

        kwargs["model"].fit(X_train, y_train)
        joblib.dump(kwargs["model"], "%s/%s_%04d.skmodel"%(args.experiment_name, 'embedding', numLabeled))
        dataset.setKind(DetectionKind.ModelDetection.value)
        dataset.embedding_mode()
        X_test= dataset.em[dataset.current_set]
        y_test= np.asarray(dataset.getlabels())
        print("Accuracy",kwargs["model"].score(X_test, y_test))
        sys.stdout.flush()
        if numLabeled % 2000 == 1000:
            dataset.setKind(DetectionKind.UserDetection.value)
            finetune_embedding(model, checkpoint['loss_type'], dataset, 32, 4, 100 if numLabeled == 1000 else 50)
            save_checkpoint({
            'arch': checkpoint['arch'],
            'state_dict': model.state_dict(),
            #'optimizer' : optimizer.state_dict(),
            'loss_type' : checkpoint['loss_type'],
            'feat_dim' : checkpoint['feat_dim'],
            'num_classes' : args.num_classes
            }, False, "%s/%s%s_%s_%04d.tar"%(args.experiment_name, 'finetuned', checkpoint['loss_type'], checkpoint['arch'], numLabeled))

            dataset.setKind(DetectionKind.ModelDetection.value)
            dataset.updateEmbedding(model)
            dataset.embedding_mode()

if __name__ == '__main__':
    print("start")
    main()

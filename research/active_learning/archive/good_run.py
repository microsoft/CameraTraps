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

from alipy.experiment.al_experiment import AlExperiment
from modAL.models import ActiveLearner
from modAL.uncertainty import *

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

from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch 

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
parser.add_argument('--name_prefix', default='', type=str, metavar='PATH',
                    help='prefix name for output files')
parser.add_argument('--num_clusters', default=20, type=int,
                    help='number of clusters')
parser.add_argument('--K', default=5, type=int,
                    help='number of clusters')


def activeLearning(prob_list, dataset):
      #uncertainty= np.apply_along_axis(stats.entropy,1,probs) * (1 - probs.max(axis=1))
      preds = prob_list[0].argmax(axis=1)
      paths = dataset.getpaths()
      uncertainty= prob_list[0].max(axis=1)
      for l in range(1,len(prob_list)):
          uncertainty += prob_list[l].max(axis=1)
      srt = np.argsort(uncertainty)#[::-1]
      base_ind= set()
      co = 0
      i = 0
      while co<200:
          base_ind.add(srt[i])
          #copy(paths[srt[i]], "active")
          co+=1
          i += 1
      #plot_together( dataset.em[dataset.current_set], np.asarray(dataset.getlabels()), preds, base_ind, dataset.getpaths(), {})
      return base_ind
      #return np.random.choice(range(0,prob_list[0].shape[0]), 100, replace=False).tolist()

def selectSamples(embd, current_set, n):
        print('Initial sample selection')
        sys.stdout.flush()
        selected_set= set()
        while len(selected_set)<n:
            print(len(selected_set))
            sys.stdout.flush()
            rand_ind= np.random.choice(np.arange(embd.shape[0]),1500, replace=False)
            db = DBSCAN(min_samples=5, n_jobs=-1).fit(embd[rand_ind])

            for i,x in enumerate(db.labels_):
                if x==-1 and getDistance(embd, selected_set, rand_ind[i]) >= 2.0:
                    selected_set.add(rand_ind[i])
        return [current_set[i] for i in selected_set]

def noveltySamples(embd, paths, n, initial_n= 50):
        print('Initial sample selection')
        sys.stdout.flush()
        selected_set= set()
        rand_ind= np.random.choice(np.arange(embd.shape[0]), initial_n, replace=False)
        for x in rand_ind:
            selected_set.add(x)
        while len(selected_set)<n:
            print(len(selected_set))
            print("Loop started")
            sys.stdout.flush()
            start = time.time()
            for i in range(embd.shape[0]):
                print(getDistance(embd, selected_set, i))
            end = time.time()
            print("Loop ended", end-start)
            break
                #    selected_set.add(rand_ind[i])
        return [paths[i] for i in selected_set]

def getDistance(embd,archive,sample):
      if len(archive)==0:
          return 100
      else:
          return pairwise_distances_argmin_min(embd[sample].reshape(1, -1),embd[np.asarray(list(archive),dtype=np.int32)])[1]

def moveRecords(dataset, srcKind, destKind, rList):
      #query= Detection.update(kind = destKind.value).where(Detection.id.in_(rList), Detection.kind == srcKind.value)
      
      #query.execute()
      for e in rList:
          dataset.set_indices[srcKind].remove(e)
          dataset.set_indices[destKind].append(e)

def train_eval_classifier(clf_model, unlabeled_dataset, model, pth, epochs = 15):
    trainset_query = Detection.select(Detection.id,Oracle.label).join(Oracle).where(Detection.kind == DetectionKind.UserDetection.value) 
    train_dataset = SQLDataLoader(trainset_query, os.path.join(args.run_data, 'crops'), is_training= True)
    train_dataset.updateEmbedding(model)
    train_dataset.embedding_mode()
    train_dataset.train()
    clf_criterion= nn.CrossEntropyLoss()
    clf_optimizer = torch.optim.Adam(clf_model.parameters(), lr=0.001, weight_decay= 0.0005)
    clf_e= Engine(clf_model,clf_criterion,clf_optimizer, verbose= True, print_freq= 1)

    clf_model.train()
    clf_train_loader = train_dataset.getSingleLoader( batch_size = 64)
    for i in range(epochs):
      clf_e.train_one_epoch(clf_train_loader, i, True)
    clf_model.eval()
    unlabeled_dataset.embedding_mode()
    unlabeled_dataset.eval()
    eval_loader = unlabeled_dataset.getSingleLoader(batch_size = 1024)
    clf_e.validate(eval_loader, True)

def custom_policy(step):
  details_str= '41, 56, 66, 0.001, 0.0005, 0.0001, 0.00005'
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

def finetune_embedding(model, train_dataset, P, K, epochs):
    train_dataset.image_mode()
    train_loader = train_dataset.getBalancedLoader(P = P, K = K)
    criterion = OnlineTripletLoss(1, RandomNegativeTripletSelector(1))
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr = 0.0001)#, weight_decay = 0.0005)
    e= Engine(model, criterion, optimizer, verbose = True, print_freq = 10)
    for epoch in range(epochs):
        e.train_one_epoch(train_loader, epoch, False)
 

def main():
    args = parser.parse_args()
    print("DB Connect")
    db_path = os.path.join(args.run_data, os.path.basename(args.run_data)) + ".db"
    print(db_path)
    db = SqliteDatabase(db_path)
    proxy.initialize(db)
    db.connect()
    print("connected")
    print("CompleteLoop")
    
    checkpoint = load_checkpoint(args.base_model)
    embedding_net = EmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)
    #embedding_net = EmbeddingNet('resnet50', 256, True)
    model = torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    #unlabeledset_query= Detection.select(Detection.id,Oracle.label).join(Oracle).where(Detection.kind==DetectionKind.ModelDetection.value).order_by(fn.random()).limit(150000)
    #unlabeled_dataset = SQLDataLoader(unlabeledset_query, os.path.join(args.run_data, "crops"), is_training= False, num_workers= 8)
    dataset = SQLDataLoader(os.path.join(args.run_data, "crops"), is_training= False, kind = DetectionKind.ModelDetection.value, num_workers= 8)
    dataset.updateEmbedding(model)
    #print('Embedding Done')
    #sys.stdout.flush()
    #plot_embedding(dataset.em[dataset.current_set], np.asarray(dataset.getlabels()) , dataset.getpaths(), {})
    # Random examples to start
    random_ids = np.random.choice(dataset.current_set, 5000, replace=False).tolist()
    #random_ids = selectSamples(dataset.em[dataset.current_set], dataset.current_set, 2000)
    #print(random_ids)
    # Move Records
    moveRecords(dataset, DetectionKind.ModelDetection.value, DetectionKind.UserDetection.value, random_ids)

    print([len(x) for x in dataset.set_indices])
    # Finetune the embedding model
    dataset.setKind(DetectionKind.UserDetection.value)
    dataset.train()
    #train_dataset = SQLDataLoader(trainset_query, os.path.join(args.run_data, 'crops'), is_training= True)
    finetune_embedding(model, dataset, 32, 4, 0)
    #unlabeled_dataset.updateEmbedding(model)
    dataset.updateEmbedding(model)
    dataset.setKind(DetectionKind.UserDetection.value)
    #print(dataset.em[dataset.current_set].shape, np.asarray(dataset.getlabels()).shape, len(dataset.getpaths()))
    #plot_embedding( dataset.em[dataset.current_set], np.asarray(dataset.getlabels()) , dataset.getpaths(), {})
    #plot_embedding( unlabeled_dataset.em, np.asarray(unlabeled_dataset.getlabels()) , unlabeled_dataset.getIDs(), {})
    dataset.embedding_mode()
    dataset.train()
    clf_model = ClassificationNet(256, 48).cuda()
    #train_eval_classifier()
    #clf_model = ClassificationNet(checkpoint['feat_dim'], 48).cuda()
    clf_criterion= FocalLoss(gamma=2)#nn.CrossEntropyLoss()
    clf_optimizer = torch.optim.Adam(clf_model.parameters(), lr=0.001, weight_decay= 0.0005)
    clf_e= Engine(clf_model,clf_criterion,clf_optimizer, verbose= True, print_freq= 10)
    #names = ["Linear SVM", "RBF SVM", "Random Forest", "Neural Net", "Naive Bayes"]
    #classifiers = [SVC(kernel="linear", C=0.025, probability= True, class_weight='balanced'),
    #    SVC(gamma=2, C=1, probability= True, class_weight='balanced'),
    #    RandomForestClassifier(max_depth=None, n_estimators=100, class_weight='balanced'),
    #    MLPClassifier(alpha=1),
    #    GaussianNB()]
    #estimators= []
    #for name, clf in zip(names, classifiers):
    #    estimators.append((name, clf))
    #eclf1 = VotingClassifier(estimators= estimators, voting='hard')
    #eclf2 = VotingClassifier(estimators= estimators, voting='soft')
    #names.append("ensemble hard")
    #classifiers.append(eclf1)
    #names.append("ensemble soft")
    #classifiers.append(eclf2)
    names = ["Neural Net"]
    classifiers = [MLPClassifier(alpha=1)]
    """dataset.setKind(DetectionKind.UserDetection.value)

    learner = ActiveLearner(
            estimator=MLPClassifier(),
            query_strategy=uncertainty_sampling,
            X_training = dataset.em[dataset.current_set], y_training = np.asarray(dataset.getlabels()))

    for step in range(91):
        dataset.setKind(DetectionKind.ModelDetection.value)
        query_idx, query_inst = learner.query(dataset.em[dataset.current_set], n_instances=100)
        moveRecords(dataset, DetectionKind.ModelDetection.value, DetectionKind.UserDetection.value, [dataset.current_set[i] for i in query_idx])
        dataset.setKind(DetectionKind.UserDetection.value)
        learner.teach(dataset.em[dataset.current_set], np.asarray(dataset.getlabels()))
        if step in [11, 31, 51, 71, 91, 101]:
            finetune_embedding(model, dataset, 32, 4, 100)
            dataset.updateEmbedding(model)
            dataset.embedding_mode()
        dataset.setKind(DetectionKind.ModelDetection.value)
        print(learner.score(dataset.em[dataset.current_set], np.asarray(dataset.getlabels())))
        print([len(x) for x in dataset.set_indices])
        sys.stdout.flush()"""
    sampler = get_AL_sampler('uniform')(dataset.em[dataset.current_set], dataset.getlabels(), 12)
    print(sampler, type(sampler), dir(sampler))
    kwargs = {}
    kwargs["N"] = 100
    kwargs["already_selected"] = []
    kwargs["model"] = SVC(kernel="linear", C=0.025, probability= True, class_weight='balanced')
    kwargs["model"].fit(dataset.em[dataset.current_set], dataset.getlabels())
    batch_AL = sampler.select_batch(**kwargs)
    print(batch_AL)
    for step in range(101):
        dataset.setKind(DetectionKind.UserDetection.value)
        clf_model.train()
        clf_train_loader = dataset.getSingleLoader( batch_size = 64)
        for i in range(15):
            clf_e.train_one_epoch(clf_train_loader, i, True)
        clf_model.eval()
        X_train= dataset.em[dataset.current_set]
        y_train= np.asarray(dataset.getlabels())
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            print(name)

        dataset.setKind(DetectionKind.ModelDetection.value)
        #dataset.image_mode()
        #dataset.updateEmbedding(model)
        dataset.embedding_mode()
        dataset.eval()
        eval_loader = dataset.getSingleLoader(batch_size = 1024)
        clf_e.validate(eval_loader, True)
        X_test= dataset.em[dataset.current_set]
        y_test= np.asarray(dataset.getlabels())
        prob_list=[]
        for name, clf in zip(names, classifiers):
            #y_pred= clf.predict(X_test)
            #print(confusion_matrix(y_test, y_pred))
            #paths= dataset.getpaths()
            #for i, (yp, yt) in enumerate(zip(y_pred, y_test)):
            #    if yp != yt:
                    #copy(paths[i],"mistakes")
                    #print(yt, yp, paths[i],i)
            if not name.startswith("ensemble"):
                prob_list.append(clf.predict_proba(X_test))
            score = clf.score(X_test, y_test)
            print(name, score)
        #clf_output= clf_e.embedding(eval_loader, dim=48)
        if step%10 ==1 and step>10:
            dataset.setKind(DetectionKind.UserDetection.value)
            finetune_embedding(model, dataset, 32, 4, 50)
            dataset.setKind(DetectionKind.ModelDetection.value)
            dataset.updateEmbedding(model)
            dataset.embedding_mode()
        indices= activeLearning(prob_list, dataset)
        moveRecords(dataset, DetectionKind.ModelDetection.value, DetectionKind.UserDetection.value, [dataset.current_set[i] for i in indices])
        print([len(x) for x in dataset.set_indices])

if __name__ == '__main__':
    print("start")
    main()

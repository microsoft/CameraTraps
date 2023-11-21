import argparse
import os
import random
import time
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import numpy as np
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

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name_prefix', default='', type=str, metavar='PATH',
                    help='prefix name for output files')
parser.add_argument('--num_clusters', default=20, type=int,
                    help='number of clusters')
parser.add_argument('--K', default=5, type=int,
                    help='number of clusters')

def find_probablemap(true_labels, clustering_labels, K=5):
    clusters= set(clustering_labels)
    mapping={}
    for x in clusters:
      sub= true_labels[clustering_labels==x]
      mapping[x]= int(stats.mode(np.random.choice(sub,K), axis=None)[0])#int(stats.mode(sub, axis=None)[0])
    return mapping

def apply_different_methods(X_train, y_train, X_test, y_test):
    """names = ["1-NN", "3-NN", "Linear SVM", "RBF SVM", "Neural Net", "AdaBoost",
         "Naive Bayes"]

    classifiers = [
    KNeighborsClassifier(1),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB()]
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name, score)"""

    trainset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    trainloader = DataLoader(trainset, batch_size=64, shuffle = True)
    testset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    testloader = DataLoader(testset, batch_size=32, shuffle = False)

    criterion = nn.CrossEntropyLoss()
    net= ClassificationNet(256,48)
    optimizer = optim.Adam(net.parameters()) 
    net.train()
    #conf= ConfusionMatrix(24)
    for epoch in range(50):  # loop over the dataset multiple times

      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs= inputs.float()
        labels= labels.long()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        conf.update(outputs,labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    print(conf.mat)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
          # get the inputs
          inputs, labels = data
          inputs= inputs.float()
          labels= labels.long()
          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
          # print statistics
          losses.update(loss.item(), inputs.size(0))
          top1.update(acc1[0], inputs.size(0))
          top5.update(acc5[0], inputs.size(0))
    print('loss: %.3f Top-1: %.3f Top-5: %.3f' % (losses.avg,top1.avg,top5.avg))

    

print('Finished Training')

"""def completeClassificationLoop(dataset,model, num_classes):
    clf= ClassificationNet(model,num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=0.0001) 
    print("Loop Started")
    base_ind = set(np.random.choice(np.arange(len(dataset)), 100, replace=False))
    for it in range(10):
      X_train= np.zeros((len(base_ind),3,224,224))
      y_train= np.zeros((len(base_ind)),dtype=np.int32)
      for i,ind in enumerate(base_ind):
        X_train[i,:,:,:], y_train[i],_= dataset[ind]

      trainset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
      trainloader = DataLoader(trainset, batch_size=32, shuffle = True)

      clf.train()
      for epoch in range(10):
        for i, data in enumerate(trainloader, 0):
          # get the inputs
          inputs, labels = data
          inputs= inputs.float().cuda()
          labels= labels.long().cuda()
          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          _,outputs = clf(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
        print(epoch,loss.item())
      clf.eval()
      testloader = DataLoader(dataset, batch_size=512, shuffle = False)
      losses = AverageMeter()
      top1 = AverageMeter()
      top5 = AverageMeter()

      uncertainty= np.zeros((len(dataset)))
      with torch.no_grad():
        for i, data in enumerate(testloader, 0):
          # get the inputs
          inputs, labels, _ = data
          inputs= inputs.float().cuda()
          labels= labels.long().cuda()
          # forward + backward + optimize
          _, outputs = clf(inputs)
          uncertainty
          acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
          # print statistics
          losses.update(loss.item(), inputs.size(0))
          top1.update(acc1[0], inputs.size(0))
          top5.update(acc5[0], inputs.size(0))
      print('loss: %.3f Top-1: %.3f Top-5: %.3f' % (losses.avg,top1.avg,top5.avg))


    #conf= NPConfusionMatrix(10)
    for it in range(9):
      clf.fit(X_train, y_train)
      #all_indices= set(range(len(y)))
      #diff= all_indices.difference(base_ind)
      print("Iteration %d, Accuracy %.3f"%(it,clf.score(X,y)))#[list(diff)],y[list(diff)])))
      preds= clf.predict_proba(X)
      preds_tr= clf.predict_proba(X_train)
      #conf.update(preds_tr,y_train)
      #classes= np.apply_along_axis(conf.classScore,0,preds.argmax(axis=1))
      uncertainty= preds.max(axis=1)
      srt = np.argsort(uncertainty)
      co=0
      i=0
      while co<100:
        if srt[i] not in base_ind:
          base_ind.add(srt[i])
          co+=1
        i+=1
      X_train=X[list(base_ind)]
      y_train=y[list(base_ind)]
      #conf.reset()"""

def completeLoop(X,y,base_ind):
    embedding_net = EmbeddingNet('densenet161', 256, True)
    center_loss= None
    model= torch.nn.DataParallel(embedding_net).cuda()
    criterion= OnlineTripletLoss(1.0, SemihardNegativeTripletSelector(1.0))
    params = model.parameters()

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.Adam(params, lr=0.0001)
    start_epoch= 0
    checkpoint= load_checkpoint('triplet_model_0086.tar')
    if checkpoint:
      model.load_state_dict(checkpoint['state_dict'])

    e= Engine(model,criterion,optimizer, verbose= True, print_freq= 10)
    trainset_query= Detection.select(Detection.id,Oracle.label,Detection.embedding).join(Oracle).where(Oracle.status==0, Detection.kind==DetectionKind.UserDetection.value) 
    embedding_dataset = SQLDataLoader(trainset_query, "all_crops/SS_full_crops", True, num_workers= 4, batch_size= 64)
    print(len(embedding_dataset))
    num_classes= 48#len(run_dataset.getClassesInfo()[0])
    print("Num Classes= "+str(num_classes))
    embedding_loader = embedding_dataset.getBalancedLoader(16,4)
    for i in range(200):
        e.train_one_epoch(embedding_loader,i,False)
    embedding_dataset2 = SQLDataLoader(trainset_query, "all_crops/SS_full_crops", False, num_workers= 4, batch_size= 512)
    em= e.embedding(embedding_dataset2.getSingleLoader())
    lb = np.asarray([ x[1] for x in embedding_dataset2.samples])
    pt = [x[0] for x in embedding_dataset2.samples]#e.predict(run_loader, load_info=True)

    co=0
    for r,s,t in zip(em,lb,pt):
        co+=1
        smp= Detection.get(id=t)
        smp.embedding= r
        smp.save()
        if co%100==0:
            print(co)
    print("Loop Started")
    train_dataset = SQLDataLoader(trainset_query, "all_crops/SS_full_crops", False, datatype='embedding', num_workers= 4, batch_size= 64)
    print(len(train_dataset))
    num_classes= 48#len(run_dataset.getClassesInfo()[0])
    print("Num Classes= "+str(num_classes))
    run_loader = train_dataset.getSingleLoader()
    clf_model = ClassificationNet(256,48).cuda()
    clf_criterion= nn.CrossEntropyLoss()
    clf_optimizer = torch.optim.Adam(clf_model.parameters(), lr=0.001, weight_decay=0.0005)
    clf_e= Engine(clf_model,clf_criterion,clf_optimizer, verbose= True, print_freq= 10)
    for i in range(250):
      clf_e.train_one_epoch(run_loader,i, True)
    testset_query= Detection.select(Detection.id,Oracle.label,Detection.embedding).join(Oracle).where(Oracle.status==0, Detection.kind==DetectionKind.ModelDetection.value)
    test_dataset = SQLDataLoader(testset_query, "all_crops/SS_full_crops", False, datatype='image', num_workers= 4, batch_size= 512)
    print(len(test_dataset))
    num_classes= 48#len(run_dataset.getClassesInfo()[0])
    print("Num Classes= "+str(num_classes))
    test_loader = test_dataset.getSingleLoader()
    test_em= e.embedding(test_loader)
    test_lb = np.asarray([ x[1] for x in test_dataset.samples])
    print(test_lb.shape,test_em.shape)
    testset = TensorDataset(torch.from_numpy(test_em), torch.from_numpy(test_lb))
    clf_e.validate(DataLoader(testset, batch_size= 512, shuffle= False), True)

    """X_train= X[list(base_ind)]
    y_train= y[list(base_ind)]
    clf= MLPClassifier(hidden_layer_sizes=(200,100), max_iter=300)
    #conf= NPConfusionMatrix(10)
    for it in range(39):
      clf.fit(X_train, y_train)
      #all_indices= set(range(len(y)))
      #diff= all_indices.difference(base_ind)
      print("Iteration %d, Accuracy %.3f"%(it,clf.score(X,y)))#[list(diff)],y[list(diff)])))
      preds= clf.predict_proba(X)
      preds_tr= clf.predict_proba(X_train)
      #conf.update(preds_tr,y_train)
      #classes= np.apply_along_axis(conf.classScore,0,preds.argmax(axis=1))
      uncertainty= preds.max(axis=1)
      srt = np.argsort(uncertainty)
      co=0
      i=0
      while co<100:
        if srt[i] not in base_ind:
          base_ind.add(srt[i])
          co+=1
        i+=1
      X_train=X[list(base_ind)]
      y_train=y[list(base_ind)]"""
      #conf.reset()

def active_learning(X,y, base_ind):
    print("Pass Started")
    X_train= X[base_ind]
    y_train= y[base_ind]
    uncertainty= np.zeros((X.shape[0]))
    classifiers = [SVC(gamma=2, C=1, probability= True), MLPClassifier()]
    #for clf in classifiers:
    #  clf.fit(X_train, y_train)
    #  preds= clf.predict_proba(X)
    #  uncertainty+= preds.max(axis=1)
    clf=classifiers[1]
    clf.fit(X_train, y_train)
    preds= clf.predict_proba(X)
    uncertainty+= preds.max(axis=1)

    ind = np.argsort(uncertainty)[0:100]
    #print(uncertainty[ind])
    return np.append(base_ind, ind)

def active_learning2(X, num_samples, k=20):
    clusters = MiniBatchKMeans(n_clusters= k).fit_predict(X)
    uncertainty= silhouette_samples(X,clusters)
    ind = np.argsort(uncertainty)[0:num_samples]
    return ind

def active_learning_entropy(X,y, base_ind):
    print("Pass Started")
    X_train= X[base_ind]
    y_train= y[base_ind]
    uncertainty= np.zeros((X.shape[0]))
    clf=MLPClassifier()
    clf.fit(X_train, y_train)
    preds= clf.predict_proba(X)
    uncertainty+= np.apply_along_axis(stats.entropy,1,preds)

    ind = np.argsort(uncertainty)[-100:]
    #print(uncertainty[ind])
    return np.append(base_ind, ind)

"""def active_learning3(X, y, base_ind):
    newy= y
    mask = np.ones(y.shape,dtype=bool) #np.ones_like(a,dtype=bool)
    mask[base_ind] = False
    newy[mask]= -1
    lp_model = label_propagation.LabelSpreading(kernel='knn', gamma=0.25, max_iter=5)
    lp_model.fit(X, newy)
    predicted_labels = lp_model.transduction_[mask]
    #true_labels = y[unlabeled_indices]
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

    # select up to 5 digit examples that the classifier is most uncertain about
    uncertainty_index = np.argsort(pred_entropies)[::-1]
    uncertainty_index = uncertainty_index[np.in1d(uncertainty_index, mask)][:100]
    return uncertainty_index"""
def selectSamples(embd, paths, n):
        selected_set= set()
        while len(selected_set)<n:
            print(len(selected_set))
            sys.stdout.flush()
            rand_ind= np.random.choice(np.arange(embd.shape[0]),3000, replace=False)
            db = DBSCAN(eps=1, min_samples=5,n_jobs=-1).fit(embd[rand_ind])
            indices=set()
            for i,x in enumerate(db.labels_):
              if x==-1 and getDistance(embd,indices,rand_ind[i])>=0.3 and getDistance(embd,selected_set,rand_ind[i])>=0.3:
                indices.add(rand_ind[i])
            moveRecords(DetectionKind.ModelDetection, DetectionKind.ActiveDetection, [paths[i] for i in indices.difference(selected_set)])
            selected_set= selected_set.union(indices)
            #print(indices,selected_set)
        return selected_set

def getDistance(embd,archive,sample):
      if len(archive)==0:
          return 100
      else:
          return pairwise_distances_argmin_min(embd[sample].reshape(1, -1),embd[np.asarray(list(archive),dtype=np.int32)])[1]

def moveRecords(srcKind,destKind,rList):
      query= Detection.update(kind=destKind.value).where(Detection.id.in_(rList), Detection.kind==srcKind.value)
      #print(query.sql())
      query.execute()

def main():
    args = parser.parse_args()
    print("DB Connect")
    db = SqliteDatabase('SS.db')
    proxy.initialize(db)
    db.connect()

    # remember best acc@1 and save checkpoint
    """checkpoint= load_checkpoint("triplet_model_0054.tar")
    runset_query= Detection.select(Detection.id,Oracle.label,Detection.embedding).join(Oracle).where(Oracle.status==0)
    print("Create Query")
    run_dataset = SQLDataLoader(runset_query, "all_crops/SS_full_crops", False, datatype='image', num_workers= args.workers*2, batch_size= 512)
    print(len(run_dataset))
    num_classes= 48#len(run_dataset.getClassesInfo()[0])
    print("Num Classes= "+str(num_classes))
    run_loader = run_dataset.getSingleLoader()
    embedding_net = EmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'])
    if checkpoint['loss_type'].lower()=='center':
      model = torch.nn.DataParallel(ClassificationNet(embedding_net, n_classes=14)).cuda()
    else:
      model= torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    e= Engine(model,None,None, verbose= True, print_freq= 10)
    embd = e.embedding(run_loader)
    print("inje")
    sys.stdout.flush()
    label = np.asarray([ x[1] for x in run_dataset.samples])
    paths = [x[0] for x in run_dataset.samples]#e.predict(run_loader, load_info=True)
    print("embedding done")
    sys.stdout.flush()"""
    #completeClassificationLoop(run_dataset, model,num_classes)
    #embd, label, paths = extract_embeddings(run_loader, model)
    #db = DBSCAN(eps=0.1, min_samples=5).fit(embd)
    #db = MiniBatchKMeans(n_clusters=args.num_clusters).fit(embd)
    #labels = db.labels_
    #mapp=(find_probablemap(label,labels, K=args.K))
    #print("Clusters")
    #for i,x in enumerate(labels):
    #  labels[i]= mapp[x] 
    #print(np.sum(labels == label)/labels.size)
    #print("Confidence Active Learning")
    #idx = np.random.choice(np.arange(len(paths)), 100, replace=False)
    #for i in range(9):
    #  idx= active_learning(embd, label, idx)
    #print(idx.shape)
    #apply_different_methods(embd[idx], label[idx], embd, label)

    #print("Entropy Active Learning")
    #idx = np.random.choice(np.arange(len(paths)), 100, replace=False)
    #for i in range(9):
    #  idx= active_learning_entropy(embd, label, idx)

    #apply_different_methods(embd[idx], label[idx], embd, label)
    #new_selected= selectSamples(embd,paths,3000)


    print("CompleteLoop")
    completeLoop(None,None,None)#new_selected)

    #print(idx,idx.shape)
    #for i in idx:
    #  print(paths[i])

    #print("Silohette active learning")
    #idx= active_learning2(embd, 1000, args.num_clusters)
    #print(idx.shape)
    #apply_different_methods(embd[idx], label[idx], embd, label)
    #print("Random")
    #idx = np.random.choice(np.arange(len(paths)), 1000, replace=False)
    #apply_different_methods(embd[idx], label[idx], embd, label)

    #apply_different_methods(embd[idx], label[idx], embd, label)
    #embd= reduce_dimensionality(embd)#[0:10000])
    #labels= labels[0:10000]
    #label= label[0:10000]
    #paths= paths[0:10000]
    #plot_embedding(embd, label, paths, run_dataset.getClassesInfo()[1])
    #plot_embedding(embd, labels, paths, run_dataset.getClassesInfo()[1])
    #plt.show()
    #np.save(args.name_prefix+"_embeddings.npy",embd)
    #np.save(args.name_prefix+"_labels.npy",label)
    #np.savetxt(args.name_prefix+"_paths.txt",paths, fmt="%s")

if __name__ == '__main__':
    print("start")
    main()

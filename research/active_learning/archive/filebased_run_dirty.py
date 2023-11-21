import argparse
import os
import random
import time
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
from DL.data_loader import BaseDataLoader
from DL.losses import *
from DL.utils import *
from DL.networks import *
from DL.Engine import Engine

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
    net= ClassificationNet(256,16)
    optimizer = optim.Adam(net.parameters()) 
    net.train()
    conf= ConfusionMatrix(24)
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

def completeLoop(X,y):
    print("Loop Started")
    base_ind = set(np.random.choice(np.arange(len(y)), 200, replace=False))
    X_train= X[list(base_ind)]
    y_train= y[list(base_ind)]
    clf= MLPClassifier(hidden_layer_sizes=(200,100), max_iter=300)
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
      while co<200:
        if srt[i] not in base_ind:
          base_ind.add(srt[i])
          co+=1
        i+=1
      X_train=X[list(base_ind)]
      y_train=y[list(base_ind)]
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

def main():
    args = parser.parse_args()
    # remember best acc@1 and save checkpoint
    checkpoint= load_checkpoint(args.resume)
    run_dataset = BaseDataLoader(args.run_data, False, num_workers= args.workers, batch_size= args.batch_size)
    num_classes= len(run_dataset.getClassesInfo()[0])
    print("Num Classes= "+str(num_classes))
    run_loader = run_dataset.getSingleLoader()
    embedding_net = EmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'])
    if checkpoint['loss_type'].lower()=='center':
      model = torch.nn.DataParallel(ClassificationNet(embedding_net, n_classes=14)).cuda()
    else:
      model= torch.nn.DataParallel(embedding_net).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    #completeClassificationLoop(run_dataset, model,num_classes)
    embd, label, paths = extract_embeddings(run_loader, model)
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

    print("CompleteLoop")
    completeLoop(embd,label)

    #print(idx,idx.shape)
    #for i in idx:
    #  print(paths[i])

    #print("Silohette active learning")
    #idx= active_learning2(embd, 1000, args.num_clusters)
    #print(idx.shape)
    #apply_different_methods(embd[idx], label[idx], embd, label)
    print("Random")
    idx = np.random.choice(np.arange(len(paths)), 1000, replace=False)
    apply_different_methods(embd[idx], label[idx], embd, label)

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



def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset),256))# 3*218*218))
        labels = np.zeros(len(dataloader.dataset))
        paths=[None]*len(dataloader.dataset)
        k = 0
        for images, target, path in dataloader:
            
            images = images.cuda()
            embedding = model(images)
            embeddings[k:k+len(images)] = embedding.data.cpu().numpy().reshape((len(images),-1))
            labels[k:k+len(images)] = target.numpy()
            paths[k:k+len(path)]=path
            k += len(images)
            del embedding
            #del output
    return embeddings, labels, paths

if __name__ == '__main__':
    main()

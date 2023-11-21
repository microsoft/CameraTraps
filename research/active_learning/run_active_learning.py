import argparse
import os
import random
import sys
import time
import pickle
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from deep_learning.data_loader import ExtendedImageFolder
from deep_learning.engine import Engine
from deep_learning.losses import OnlineContrastiveLoss, OnlineTripletLoss
from deep_learning.networks import NormalizedEmbeddingNet, SoftmaxNet
from deep_learning.utils import HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector, HardNegativePairSelector
from deep_learning.utils import load_checkpoint, save_checkpoint, getCriterion
from deep_learning.active_learning_manager import ActiveLearningManager
from active_learning_methods.constants import get_AL_sampler, get_wrapper_AL_mapping

np.set_printoptions(threshold=np.inf)


get_wrapper_AL_mapping()

LOSS_TYPES = ['softmax', 'triplet', 'siamese']
STRATEGY_TYPES = ['uniform', 'graph_density', 'entropy', 'confidence',
     'kcenter', 'margin', 'informative_diverse', 'margin_cluster_mean', 'hierarchical']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--run_data', help='path to train dataset', default='SS_crops_256')
parser.add_argument('-j', '--num_workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--base_model', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--experiment_name', default='', type=str, help='prefix name for output files')
parser.add_argument('-N', '--active_batch', default=100, type=int, help='number of queries per batch')
parser.add_argument('-A', '--active_budget', default= 30000, type=int, help='number of queries per batch')
parser.add_argument('--finetuning_P', default=16, type=int,
                    help='The number of classes in each balanced batch')
parser.add_argument('--finetuning_K', default=4, type=int,
                    help='The number of examples from each class in each balanced batch')
parser.add_argument('--finetuning_lr', default=0.0001,
                    type=float, help='initial learning rate')
parser.add_argument('--active_learning_strategy', default='margin', choices=STRATEGY_TYPES, help='Active learning strategy')
parser.add_argument('--loss_type', default='triplet', choices=LOSS_TYPES,
                    help='loss type: ' + ' | '.join(LOSS_TYPES) + ' (default: triplet loss)')
parser.add_argument('--margin', default=1.0, type=float,
                    help='margin for siamese or triplet loss')
parser.add_argument('--finetuning_strategy', default='random', choices=['hardest', 'random', 'semi_hard', 'hard_pair'],
                    help='data selection strategy')
parser.add_argument('--num_finetune_epochs', default= 100, type=int,
                    help='number of total epochs to run for finetuning')
parser.add_argument('--normalize_embedding', action="store_true",
                    help='If normalize embedding values or not')



def main():
    #parse arguments
    args = parser.parse_args()
    print(args)
    # check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE: ", device)
    # load a pretrained embedding model
    checkpoint = load_checkpoint(args.base_model)
    # setup experiment
    if args.experiment_name == '':
        args.experiment_name = "experiment_%s_%s"%(checkpoint['loss_type'], args.active_learning_strategy)
    if not os.path.exists(args.experiment_name):
        os.mkdir(args.experiment_name)
    # load the embedding model
    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)

    model = torch.nn.DataParallel(embedding_net).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    # setup the target dataset
    target_dataset = ExtendedImageFolder(args.run_data)
    # setup finetuning criterion
    criterion = getCriterion(args.loss_type, args.finetuning_strategy, args.margin)
    # setup an active learning environment
    env = ActiveLearningManager(target_dataset, model, device, criterion, args.normalize_embedding)
    sampler = None
    N = args.active_batch
    # create a classifier
    classifier = MLPClassifier(hidden_layer_sizes=(150, 100), alpha=0.0001, max_iter= 2000)
    # the main active learning loop
    print("Active learning loop is started")
    numQueries = len(env.active_pool)
    while numQueries <= args.active_budget:
        # Active Learning
        if numQueries == 0:
            indices = np.random.choice(env.default_pool, 1000, replace=False).tolist()
        else:
            indices = sampler.select_batch(N= N, already_selected= env.active_pool, model= classifier)

        env.active_pool.extend(indices)
        env.default_pool = list(set(env.default_pool).difference(indices))
        numQueries = len(env.active_pool)
        # finetune the embedding model and load new embedding values
        if numQueries % 2000 == 1000:
            env.finetune_embedding(args.num_finetune_epochs, args.finetuning_P, args.finetuning_K, args.finetuning_lr)
            save_checkpoint({
            'arch': checkpoint['arch'],
            'state_dict': model.state_dict(),
            'loss_type' : checkpoint['loss_type'],
            'feat_dim' : checkpoint['feat_dim']
            }, False, "%s/%s%s_%s_%04d.tar" 
            % (args.experiment_name, 'finetuned', checkpoint['loss_type'], checkpoint['arch'], numQueries))

            env.updateEmbedding(batch_size=args.batch_size, num_workers=args.num_workers)
            sampler = get_AL_sampler(args.active_learning_strategy)(env.embedding, None, random.randint(0, sys.maxsize * 2 + 1)) 
        # gather labeled pool and train the classifier
        X_train, y_train = env.getTrainSet()
        classifier.fit(X_train, y_train)
        X_test, y_test= env.getTestSet()
        print("Number of Queries: %d,  Accuracy: %.4f"%(numQueries, classifier.score(X_test, y_test)))
        cm = confusion_matrix(y_test, classifier.predict(X_test))
        pc_acc = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).diagonal()
        # save a snapshot
        torch.save({'classifier': pickle.dumps(classifier) , "pools": env.get_pools(), 
            "confusion_matrix":cm, "per_class_accuracy": pc_acc, "class_to_idx": target_dataset.class_to_idx},
            "%s/%s_%04d.pth"%(args.experiment_name, 'AL_snapshot', numQueries), pickle_protocol=4)
        sys.stdout.flush()

if __name__ == '__main__':
    main()
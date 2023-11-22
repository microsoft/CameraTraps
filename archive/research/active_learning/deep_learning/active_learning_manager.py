from .engine import Engine
import numpy as np
import torch

class ActiveLearningManager(object):
    #constructor
    def __init__(self, dataset, embedding_model, device, criterion, normalize):
        self.dataset = dataset
        self.default_pool = list(range(len(dataset)))
        self.active_pool = []
        self.current_pool = self.default_pool
        self.model = embedding_model
        self.embedding = None
        self.device = device
        self.criterion = criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
        self.engine= Engine(self.device, self.model, criterion, optimizer)
        self.normalize = normalize
    
    #update embedding values after a finetuning
    def updateEmbedding(self, batch_size = 128, num_workers = 4):
        print('Extracting embedding from the provided model ...')
        self.embedding = self.engine.embedding(self.dataset.getSingleLoader(batch_size= batch_size, shuffle = False, num_workers=num_workers, 
            sub_indices= list(range(len(self.dataset))), transfm ="val"), normalize=self.normalize)

    # select either the default or active pools
    def setPool(self, pool):
        assert pool in ["default", "active"]
        if pool == 'default':
            self.current_pool = self.default_pool
        else:
            self.current_pool = self.active_pool

    # gather test set
    def getTestSet(self):
        return self.embedding[self.default_pool], np.asarray([self.dataset.samples[self.dataset.indices[x]][1] for x in self.default_pool])

    # gather train set
    def getTrainSet(self):
        return self.embedding[self.active_pool], np.asarray([self.dataset.samples[self.dataset.indices[x]][1] for x in self.active_pool])

    # finetune the embedding model over the labeled pool
    def finetune_embedding(self, epochs, P, K, lr, num_workers=10):
        train_loader = self.dataset.getBalancedLoader(P= P, K= K, num_workers = num_workers, sub_indices= self.active_pool)
        for epoch in range(epochs):
            self.engine.train_one_epoch(train_loader, epoch, False)
    # a utility function for saving the snapshot
    def get_pools(self):
        return {"embedding":self.embedding, "active_indices": self.active_pool, "default_indices":self.default_pool,
               "active_pool":[self.dataset.samples[self.dataset.indices[x]] for x in self.active_pool], 
               "default_pool":[self.dataset.samples[self.dataset.indices[x]] for x in self.default_pool]}

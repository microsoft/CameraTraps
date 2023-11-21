

import os, random
import numpy as np
from PIL import Image as PILImage
from PIL import ImageStat
from peewee import *
from Database.DB_models import *
# from UIComponents.DBObjects import *
from .Engine import Engine

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import *
from torch.utils.data.sampler import BatchSampler


class SQLDataLoader(Dataset):
    """
    Class for SQL dataset used for active learning.

    Attributes:
        img_base: Path to directory containing image files for detections in the dataset.
        query: SQL query on Detection database table to get id, label and kind for samples in the dataset.
        samples: Samples returned from the query on the database.
        set_indices: Indices of detections of each kind.
        kind: Which detection kind to use for current set of samples.
        current_set: Subset of samples with specified kind.
        model: Embedding model to use on the samples.
        
        embedding: Whether to return embedding representation (True, used for classifier training) or actual image (False, used for embedding training).
        is_training: Whether the samples in current set are being used for (re)training (True) or validating (False) the embedding.
        train_transform: Image transforms to apply while training the embedding. 
        eval_transform: Image transforms to apply while evaluating the embedding.
        num_workers: Number of subprocesses to use for data loading.
    """

    def __init__(self, img_base, query=None, is_training=False, embedding_mode=False, kind=DetectionKind.ModelDetection.value, model=None, num_workers=8, raw_size=[256,256], processed_size=[224,224], limit=5000000):
        self.img_base = img_base
        if query is None:
            self.query = Detection.select(Detection.id, Oracle.label, Detection.kind).join(Oracle).order_by(fn.random()).limit(limit)
        else:
            self.query = query
        self.refresh(self.query)
        self.set_indices = [[],[],[],[],[]]# indices with ActiveDetection, ModelDetection, ConfirmedDetection, UserDetection, and all user labeled/confirmed detections (ConfirmedDetection U UserDetection)
        for i, s in enumerate(self.samples):
            self.set_indices[s[2]].append(i)
        self.set_indices[4] = list(set(self.set_indices[2]).union(self.set_indices[3]))
        print([len(x) for x in self.set_indices])
        self.kind = kind
        self.current_set = self.set_indices[kind]

        self.embedding = embedding_mode
        self.is_training = is_training
        
        mean, std = self.get_mean_std()
        self.train_transform = transforms.Compose([Resize(raw_size), RandomCrop(processed_size), RandomHorizontalFlip(), ColorJitter(), RandomRotation(20), ToTensor(), Normalize(mean, std)])
        self.eval_transform = transforms.Compose([Resize(raw_size), CenterCrop((processed_size)), ToTensor(), Normalize(mean, std)])
        
        self.num_workers = num_workers
        
        if model is not None:
            self.updateEmbedding(model)

    def refresh(self, query):
        """Updates samples in SQL dataset by executing query."""

        print('Reading database to get samples.')
        self.samples = list(query.tuples())

    def embedding_mode(self):
        """Sets embedding attribute of the SQL dataset to True."""
        
        self.embedding = True

    def image_mode(self):
        """Sets embedding attribute of the SQL dataset to False."""
        
        self.embedding = False
    
    def train(self):
        """Sets is_training attribute of the SQL dataset to True."""
        
        self.is_training = True
  
    def eval(self):
        """Sets is_training attribute of the SQL dataset to False."""
        
        self.is_training = False

    def set_kind(self, kind):
        """Changes current set of samples to specified kind."""

        self.current_set = self.set_indices[kind]
        # TODO: should this also change self.kind?

    def updateEmbedding(self, model):
        """Get embedding features for entire SQL dataset through a given model."""
        
        print('Extracting embedding from the provided model ...')
        self.model = model
        e = Engine(model, None, None, verbose=True, print_freq=10)
        temp = self.is_training
        # get the embedding representations for all samples (i.e. set current_set to all indices)
        self.current_set = list(range(len(self.samples)))
        self.eval() # temporarily set is_training = False while generating embedding features
        self.em = e.embedding(self.getSingleLoader(batch_size = 1024))
        self.is_training = temp # revert is_training to previous value
        self.set_kind(self.kind) # revert current_set to only samples of specified detection kind
        print('Embedding extraction is done.')

    def get_mean_std(self):
        """Get RGB channel means and stds for image samples in the SQL dataset."""

        info = Info.get()
        if info.RM == -1 and info.RS == -1: 
            print("Calculating dataset mean and std")
            means = np.zeros((3))
            stds = np.zeros((3))
            sample_size= min(len(self.samples), 10000)
            for i in range(sample_size):
                img = self.loader(random.choice(self.samples)[0])
                stat = ImageStat.Stat(img)
                means+= np.array(stat.mean)/255.0
                stds+= np.array(stat.stddev)/255.0
            means = means/sample_size
            stds = stds/sample_size
            info.RM, info.GM, info.BM = means
            info.RS, info.GS, info.BS = stds
            info.save()
            print("Updated dataset mean and std")
        else:
            print("Load dataset mean and std from database")
            means = [info.RM, info.GM, info.BM]
            stds =  [info.RS, info.GS, info.BS]
        return means, stds

    def __len__(self):
        return len(self.current_set)

    def loader(self, path):
        """Loads image given path to image file."""

        return PILImage.open(os.path.join(self.img_base,path+".JPG")).convert('RGB')
        # return PILImage.open(os.path.join(self.img_base,path)).convert('RGB')

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index (position) of sample in current set list.
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        index = self.current_set[idx]
        path = self.samples[index][0]
        target = self.samples[index][1]
        if target is None: # to get getSingleLoader to work when using Detection.category (not Oracle) as label, and there are no labels for the current sample set
            target = -1
        if not self.embedding:
            sample = self.loader(path)
            if self.is_training:
                sample = self.train_transform(sample)
            else:
                sample = self.eval_transform(sample)
            return sample, target, path
        else:
            return self.em[index], target, path
  
    def getpaths(self):
        """Returns list of paths to image files in current sample set."""

        return [os.path.join(self.img_base, self.samples[i][0]+".JPG") for i in self.current_set]
    
    def getallpaths(self):
        """Returns list of paths to image files in entire dataset sample set."""

        return [os.path.join(self.img_base,i[0]+".JPG") for i in self.samples]

    def getIDs(self):
        """Returns list of detection IDs for samples in current sample set."""

        return [self.samples[i][0] for i in self.current_set]

    def getallIDs(self):
        """Returns list of detection IDs for samples in entire dataset sample set."""

        return [self.samples[i][0] for i in range(len(self.samples))]

    def getlabels(self):
        """Returns list of labels for samples in current sample set."""

        return [self.samples[i][1] for i in self.current_set]

    def getalllabels(self):
        """Returns list of labels for samples in entire dataset sample set."""

        return [self.samples[i][1] for i in range(len(self.samples))]

    def writeback(self):
        for i, l in enumerate(self.set_indices):
            query = Detection.update(kind = i).where(Detection.id.in_(rList))  
            query.execute()

    def getClassesInfo(self):
        return list(Category.select())

    def getBalancedLoader(self, P=14, K=10):
        train_batch_sampler = BalancedBatchSampler(self, n_classes=P, n_samples=K)
        return DataLoader(self, batch_sampler=train_batch_sampler, num_workers= self.num_workers)

    def getSingleLoader(self, batch_size = -1):
        """Data loader for the SQL dataset that returns items from the current sample set."""

        if batch_size == -1:
            batch_size = 128 if self.is_training else 256
        return DataLoader(self, batch_size = batch_size, shuffle = self.is_training, num_workers = self.num_workers)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, underlying_dataset, n_classes, n_samples):
        self.labels = [underlying_dataset.samples[i][1] for i in underlying_dataset.current_set]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = min(n_classes, len(self.labels_set))
        self.n_samples = n_samples
        self.dataset = underlying_dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < (len(self.dataset) * 4):
            #print(self.labels_set, self.n_classes)
            classes = np.random.choice(list(self.labels_set), self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return (len(self.dataset) // (self.n_samples*self.n_classes)) * 4

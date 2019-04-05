#
# sqlite_data_loader.py
#
# Custom pytorch-friendly loader to prepare images for training/inference from a 
# DB source
#

#%% Constants and imports

from torch.utils.data import Dataset
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

import numpy as np
import os
import random
from PIL import Image as PILImage
from PIL import ImageStat
from peewee import *
from UIComponents.DBObjects import *


#%% Data loader

class SQLDataLoader(Dataset):

  def __init__(self, kind, base, is_training, batch_size=None, shuffle=None, num_workers=8, raw_size=[256,256], processed_size=[224,224]):
    self.is_training = is_training
    if batch_size is None:
      self.batch_size = 128 if self.is_training else 512
    else:
      self.batch_size = batch_size
    self.shuffle = shuffle if shuffle is not None else is_training 
    self.num_workers = num_workers
    transform_list =[]
    transform_list.append(Resize(raw_size))
    if self.is_training:
      transform_list.append(RandomCrop(processed_size))
      transform_list.append(RandomHorizontalFlip())
      transform_list.append(ColorJitter())
      transform_list.append(RandomRotation(20))
    else:
      transform_list.append(CenterCrop((processed_size)))
    transform_list.append(ToTensor())
    self.base = base
    self.kind = kind
    self.refresh(Detection.select(Detection.id, Category.id).join(Category).where(Detection.kind==DetectionKind.ModelDetection.value))
    mean, std = self.get_mean_std()
    transform_list.append(Normalize(mean,std))
    self.transform = transforms.Compose(transform_list)


  def refresh(self,query):
    print("Reading database")
    #self.samples= list(self.model.select(self.model.id,Category.id).join(Category).limit(500480).tuples())
    self.samples = list(query.tuples())
    #print(self.samples)


  def get_mean_std(self):
    info = Info.get()
    print(info)
    if info.RM == None and info.RS == None: 
        print("Calculating Mean and Std")
        means = np.zeros((3))
        stds = np.zeros((3))
        sample_size = min(len(self.samples), 10000)
        for i in range(sample_size):
          img = self.loader(random.choice(self.samples).id)
          stat = ImageStat.Stat(img)
          means += np.array(stat.mean)/255.0
          stds += np.array(stat.stddev)/255.0
        means = means/sample_size
        stds = stds/sample_size
        info.RM, info.GM, info.BM= means
        info.RS, info.GS, info.BS= stds
        info.save()
    else:
        print("Load Mean and Std from databse")
        means = [info.RM, info.GM, info.BM]
        stds =  [info.RS, info.GS, info.BS]
        print(means,stds)

    return means, stds


  def __len__(self):
    return len(self.samples)


  def loader(self,path):
    return PILImage.open(os.path.join(self.base,path+".JPG")).convert('RGB')


  def __getitem__(self, index):
    """
      Args:
        index (int): Index
      Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    path = self.samples[index][0]
    target = self.samples[index][1]
    sample = self.loader(path)
    if self.transform is not None:
      sample = self.transform(sample)
    #print(target,path)
    return sample, target, path


  def getClassesInfo(self):
    return list(Category.select())


  def getBalancedLoader(self, P=14, K=10):
    train_batch_sampler = BalancedBatchSampler(self, n_classes=P, n_samples=K)
    return DataLoader(self, batch_sampler=train_batch_sampler, num_workers= self.num_workers)


  def getSingleLoader(self):
    return DataLoader(self, batch_size= self.batch_size, shuffle= self.shuffle, num_workers= self.num_workers)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, underlying_dataset, n_classes, n_samples):
        self.labels = [s[1] for s in underlying_dataset.samples]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = underlying_dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
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
        return len(self.dataset) // (self.n_samples*self.n_classes)

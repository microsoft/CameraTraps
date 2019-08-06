from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import ImageStat

def normalize(X):
  mean= X.view(3,-1).mean(1)
  std= X.view(3,-1).std(1)
  for t, m, s in zip(X, mean, std):
    t.sub_(m).div_(s)
  return X

class BaseDataLoader(ImageFolder):

  def __init__(self, base_folder, is_training, batch_size=None, shuffle=None, num_workers=8, raw_size=[256,256], processed_size=[224,224]):
    self.base_folder= base_folder
    super().__init__(base_folder)
    self.is_training= is_training
    if batch_size is None:
      self.batch_size= 128 if self.is_training else 512
    else:
      self.batch_size= batch_size
    self.shuffle= shuffle if shuffle is not None else is_training 
    self.num_workers= num_workers
    transform_list=[]
    transform_list.append(Resize(raw_size))
    if self.is_training:
      transform_list.append(RandomCrop(processed_size))
      transform_list.append(RandomHorizontalFlip())
      transform_list.append(ColorJitter())
      transform_list.append(RandomRotation(20))
      #transform_list.append(CenterCrop((processed_size)))
    else:
      transform_list.append(CenterCrop((processed_size)))
    transform_list.append(ToTensor())
    mean, std= self.calc_mean_std()
    transform_list.append(Normalize(mean,std))
    #transform_list.append(Lambda(lambda X: normalize(X)))
    self.transform = transforms.Compose(transform_list)


  def calc_mean_std(self):

      cache_file= self.base_folder+"/.meanstd"+".cache"
      if not os.path.exists(cache_file):
        print("Calculating Mean and Std")
        means= np.zeros((3))
        stds = np.zeros((3))
        sample_size= min(len(self.samples), 10000)
        for i in range(sample_size):
          img = self.loader(random.choice(self.samples)[0])
          stat = ImageStat.Stat(img)
          means+= np.array(stat.mean)/255.0
          stds+= np.array(stat.stddev)/255.0
        means= means/sample_size
        stds= stds/sample_size
        np.savetxt(cache_file, np.vstack((means, stds)))
      else:
        print("Load Mean and Std from "+cache_file)
        contents= np.loadtxt(cache_file)
        means= contents[0,:]
        stds= contents[1,:]

      return means, stds


  def __getitem__(self, index):
    """
      Args:
        index (int): Index
      Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)

    return sample, target, path

  def getClassesInfo(self):
    return self.classes, self.class_to_idx

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

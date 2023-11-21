from torch.utils.data import Dataset
from torchvision.transforms import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler

import numpy as np
import os
import random
from PIL import Image as PILImage
from PIL import ImageStat
from peewee import *
from UIComponents.DBObjects import *
from .Engine import Engine

class SQLDataLoader(Dataset):

  def __init__(self, query, img_base, is_training= False, embedding_mode = False, model = None, num_workers=8, raw_size=[256,256], processed_size=[224,224]):
      self.is_training= is_training
      self.embedding= embedding_mode
      self.num_workers = num_workers
      self.img_base= img_base
      self.query= query
      self.refresh(self.query)
      if model is not None:
          self.updateEmbedding(model)
      mean, std= self.get_mean_std()
      self.train_transform = transforms.Compose([Resize(raw_size), RandomCrop(processed_size), RandomHorizontalFlip(), ColorJitter(), RandomRotation(20), ToTensor(), Normalize(mean, std)])
      #self.train_transform = transforms.Compose([Resize(raw_size), CenterCrop((processed_size)), ToTensor(), Normalize(mean, std)])
      self.eval_transform = transforms.Compose([Resize(raw_size), CenterCrop((processed_size)), ToTensor(), Normalize(mean, std)])

  def train(self):
      self.is_training = True
  
  def eval(self):
      self.is_training = False

  def embedding_mode(self):
      self.embedding = True
  
  def image_mode(self):
      self.embedding = False

  def refresh(self,query):
      print('Reading database')
      self.samples= list(query.tuples())
  
  def updateEmbedding(self, model):
      print('Extracting embedding from the provided model ...')
      self.model = model
      e = Engine(model, None, None, verbose = True, print_freq = 10)
      temp = self.is_training
      self.eval()
      self.em = e.embedding(self.getSingleLoader(batch_size = 1024))
      self.is_training = temp
      print('Embedding extraction is done.')

  def get_mean_std(self):
      info= Info.get()
      print(info)
      if info.RM == -1 and info.RS == -1: 
          print("Calculating Mean and Std")
          means = np.zeros((3))
          stds = np.zeros((3))
          sample_size= min(len(self.samples), 10000)
          for i in range(sample_size):
              img = self.loader(random.choice(self.samples)[0])
              stat = ImageStat.Stat(img)
              means+= np.array(stat.mean)/255.0
              stds+= np.array(stat.stddev)/255.0
          means= means/sample_size
          stds= stds/sample_size
          info.RM, info.GM, info.BM= means
          info.RS, info.GS, info.BS= stds
          info.save()
      else:
          print("Load Mean and Std from database")
          means= [info.RM, info.GM, info.BM]
          stds=  [info.RS, info.GS, info.BS]
          print(means,stds)

      return means, stds

  def __len__(self):
      return len(self.samples)

  def loader(self,path):
      return PILImage.open(os.path.join(self.img_base,path+".JPG")).convert('RGB')

  def __getitem__(self, index):
    """
      Args:
        index (int): Index
      Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    path = self.samples[index][0]
    target = self.samples[index][1]
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
      return [os.path.join(self.img_base,s[0]+".JPG") for s in self.samples]

  def getIDs(self):
      return [s[0] for s in self.samples]

  def getlabels(self):
      return [s[1] for s in self.samples]

  def getClassesInfo(self):
    return list(Category.select())

  def getBalancedLoader(self, P=14, K=10):
    train_batch_sampler = BalancedBatchSampler(self, n_classes=P, n_samples=K)
    return DataLoader(self, batch_sampler=train_batch_sampler, num_workers= self.num_workers)

  def getSingleLoader(self, batch_size = -1):
      if batch_size == -1:
          batch_size = 128 if self.is_training else 256
      return DataLoader(self, batch_size= batch_size, shuffle= self.is_training, num_workers= self.num_workers)


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
        self.n_classes = min(n_classes, len(self.labels_set))
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

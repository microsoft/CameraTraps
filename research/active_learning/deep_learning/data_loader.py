from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import (RandomCrop, RandomErasing, 
CenterCrop, ColorJitter, RandomRotation, RandomHorizontalFlip, RandomOrder,
Normalize, Resize, Compose, ToTensor, RandomGrayscale)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import numpy as np
import os
import random
from PIL import ImageStat
from .engine import Engine

class ExtendedImageFolder(ImageFolder):

    def __init__(self, base_folder, indices = None):
        super().__init__(base_folder)
        self.base_folder = base_folder
        self.mean, self.std = self.calc_mean_std()
        if indices is None:
            self.indices = list(range(len(self.samples)))
        elif isinstance(indices, list):
            self.indices = indices
        elif isinstance(indices, int):
            assert indices <= len(self.samples)
            self.indices = random.sample(range(len(self.samples)), indices)
        else:
            raise TypeError('Invalid type for indices')
        self.trnsfm = {}
        self.trnsfm['train'] = self.get_transform('train')
        self.trnsfm['val'] = self.get_transform('val')

    def setTransform(self, transform):
        assert transform in ["train", "val"]
        self.transform = self.trnsfm[transform]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, ind):
        """
          Args:
            index (int): Index
          Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        index = self.indices[ind]
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def get_transform(self, trns_mode):
        transform_list = []
        transform_list.append(Resize((256, 256)))
        if trns_mode == 'train':
            transform_list.append(RandomCrop((224, 224)))
            transform_list.append(RandomGrayscale())
            transform_list.append(RandomOrder(
                [RandomHorizontalFlip(), ColorJitter(), RandomRotation(20)]))
        else:
            transform_list.append(CenterCrop((224, 224)))
        transform_list.append(ToTensor())
        transform_list.append(Normalize(self.mean, self.std))
        if trns_mode == 'train':
            transform_list.append(RandomErasing(value='random'))

        return Compose(transform_list)

    def calc_mean_std(self):
        cache_file = os.path.join(self.base_folder, ".meanstd.cache")
        if not os.path.exists(cache_file):
            print("Calculating Mean and Std")
            means = np.zeros((3))
            stds = np.zeros((3))
            sample_size = min(len(self.samples), 10000)
            for i in range(sample_size):
                img = self.loader(random.choice(self.samples)[0])
                stat = ImageStat.Stat(img)
                means += np.array(stat.mean)/255.0
                stds += np.array(stat.stddev)/255.0
            means = means/sample_size
            stds = stds/sample_size
            np.savetxt(cache_file, np.vstack((means, stds)))
        else:
            print("Load Mean and Std from "+cache_file)
            contents = np.loadtxt(cache_file)
            means = contents[0, :]
            stds = contents[1, :]

        return means, stds

    def getClassesInfo(self):
        return self.classes, self.class_to_idx

    def getBalancedLoader(self, P= 10, K= 10, num_workers = 4, sub_indices= None, transfm = 'train'):
        self.setTransform(transfm)
        if sub_indices is not None:
            subset = Subset(self, sub_indices)
            train_batch_sampler = BalancedBatchSampler(subset, n_classes = P, n_samples = K)
            return DataLoader(subset, batch_sampler = train_batch_sampler, num_workers = num_workers)
        train_batch_sampler = BalancedBatchSampler(self, n_classes = P, n_samples = K)
        return DataLoader(self, batch_sampler = train_batch_sampler, num_workers = num_workers)

    def getSingleLoader(self, batch_size = 128, shuffle = True, num_workers = 4, sub_indices= None, transfm = 'train'):
        self.setTransform(transfm)
        if sub_indices is not None:
            return DataLoader(Subset(self, sub_indices), batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)   
        return DataLoader(self, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, underlying_dataset, n_classes, n_samples):
        if hasattr(underlying_dataset, "dataset"):
            self.labels = [underlying_dataset.dataset.samples[underlying_dataset.dataset.indices[i]][1] for i in underlying_dataset.indices]
        else:
            self.labels = [underlying_dataset.samples[i][1] for i in underlying_dataset.indices]
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
            classes = np.random.choice(
                list(self.labels_set), self.n_classes, replace=False)
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
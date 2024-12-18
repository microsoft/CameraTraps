# Import necessary libraries
import os
from glob import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Exportable class names for external use
__all__ = [
    'Custom_Crop'
]

# Define the allowed image extensions  
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")  
  
def has_file_allowed_extension(filename: str, extensions: tuple) -> bool:  
    """Checks if a file is an allowed extension."""  
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))
  
def is_image_file(filename: str) -> bool:  
    """Checks if a file is an allowed image extension."""  
    return has_file_allowed_extension(filename, IMG_EXTENSIONS) 

# Define normalization mean and standard deviation for image preprocessing
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Define data transformations for training and validation datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

class Custom_Base_DS(Dataset):
    """
    Base dataset class for handling custom datasets.

    Attributes:
        rootdir (str): Root directory containing the dataset.
        transform (callable, optional): Transformations to be applied to each data sample.
        predict (bool): Flag to indicate if the dataset is used for prediction.
    """

    def __init__(self, rootdir, transform=None, predict=False):
        """
        Initialize the Custom_Base_DS with the directory, transformations, and mode.

        Args:
            rootdir (str): Directory containing the dataset.
            transform (callable, optional): Transformations to be applied to each data sample.
            predict (bool): Flag to indicate if the dataset is used for prediction.
        """
        self.rootdir = rootdir
        self.transform = transform
        self.predict = predict
        self.data = []
        self.label_ids = []
        self.labels = []
        self.seq_ids = []

    def load_data(self):
        """
        Load data from the specified directory. Differentiates between prediction and training/validation mode.
        """
        if self.predict:
            # Load data for prediction
            # self.data = glob(os.path.join(self.img_root,"*.{}".format(self.extension)))
            self.data = [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.img_root) for f in filenames if is_image_file(f)] # dp: directory path, dn: directory name, f: filename
        else:
            # Load data for training/validation
            self.data = list(self.ann['path'])
            self.label_ids = list(self.ann['classification'])
            self.labels = list(self.ann['label'])
        print('Number of images loaded: ', len(self.data))

    def class_counts_cal(self):
        """
        Calculate the count of each class in the dataset.

        Returns:
            tuple: Unique label IDs and their respective counts.
        """
        unique_label_ids, unique_counts = np.unique(self.label_ids, return_counts=True)
        return unique_label_ids, unique_counts

    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns:
            int: Total number of items.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieve an item by its index.

        Args:
            index (int): Index of the item to be retrieved.

        Returns:
            tuple: Depending on the mode, returns different tuples containing the image and additional information.
        """
        file_id = self.data[index]
        file_dir = os.path.join(self.img_root, file_id) if not self.predict else file_id

        with open(file_dir, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        if self.predict:
            return sample, file_id

        label_id = self.label_ids[index]
        label = self.labels[index]

        return sample, label_id, label, file_dir


class Custom_Crop_DS(Custom_Base_DS):
    """
    Dataset class for handling custom cropped datasets.

    Inherits from Custom_Base_DS and includes specific handling for cropped data.
    """

    def __init__(self, rootdir, dset='train', transform=None):
        """
        Initialize the Custom_Crop_DS with the dataset directory, type, and transformations.

        Args:
            rootdir (str): Directory containing the dataset.
            dset (str): Type of dataset (train, val, test, predict).
            transform (callable, optional): Transformations to be applied to each data sample.
        """
        self.predict = dset == 'predict'
        super().__init__(rootdir=rootdir, transform=transform, predict=self.predict)
        self.img_root = rootdir if self.predict else os.path.join(self.rootdir, 'cropped_resized')
        if not self.predict:
            self.ann = pd.read_csv(os.path.join(self.rootdir, 'cropped_resized', '{}_annotations_cropped.csv'
                                                .format('test' if dset == 'test' else dset)))
        self.load_data()


class Custom_Base(pl.LightningDataModule):
    """
    Base data module for handling custom datasets in PyTorch Lightning.

    Manages the data loading pipeline for training, validation, testing, and prediction.
    """

    ds = None

    def __init__(self, conf):
        """
        Initialize the Custom_Base data module with configuration.

        Args:
            conf (object): Configuration object containing dataset paths and other settings.
        """
        super().__init__()
        self._log_hyperparams = True
        self.id_to_labels = None # We don't need this for evaluations. We should save this in model weights in the future
        self.train_class_counts = None

        self.conf = conf

        print('Loading datasets...')
        # Load datasets for different modes (training, validation, testing, prediction)
        if self.conf.predict:
            self.dset_pr = self.ds(rootdir=self.conf.predict_root, dset='predict', transform=data_transforms['val'])
        elif self.conf.test:
            self.dset_te = self.ds(rootdir=self.conf.dataset_root, dset='test', transform=data_transforms['val'])
            self.id_to_labels = {i: l for i, l in np.unique(pd.Series(zip(self.dset_te.label_ids, self.dset_te.labels)))}
        else:
            self.dset_tr = self.ds(rootdir=self.conf.dataset_root, dset='train', transform=data_transforms['train'])
            self.dset_val = self.ds(rootdir=self.conf.dataset_root, dset='val', transform=data_transforms['val'])

            self.id_to_labels = {i: l for i, l in np.unique(pd.Series(zip(self.dset_tr.label_ids, self.dset_tr.labels)))}
            # Calculate class counts and label mappings
            self.unique_label_ids, self.train_class_counts = self.dset_tr.class_counts_cal()

        print('Datasets loaded.')

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(self.dset_tr, batch_size=self.conf.batch_size, shuffle=True, pin_memory=True, num_workers=self.conf.num_workers, drop_last=False)

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(self.dset_val, batch_size=self.conf.batch_size, shuffle=False, pin_memory=True, num_workers=self.conf.num_workers, drop_last=False)

    def test_dataloader(self):
        """
        Create a DataLoader for the testing dataset.

        Returns:
            DataLoader: DataLoader for the testing dataset.
        """
        return DataLoader(self.dset_te, batch_size=256, shuffle=False, pin_memory=True, num_workers=self.conf.num_workers, drop_last=False)

    def predict_dataloader(self):
        """
        Create a DataLoader for the prediction dataset.

        Returns:
            DataLoader: DataLoader for the prediction dataset.
        """
        return DataLoader(self.dset_pr, batch_size=64, shuffle=False, pin_memory=True, num_workers=self.conf.num_workers, drop_last=False)


class Custom_Crop(Custom_Base):
    """
    Custom data module specifically for cropped datasets in PyTorch Lightning.

    Inherits from Custom_Base and specifies the dataset type as Custom_Crop_DS.
    """

    def __init__(self, conf):
        """
        Initialize the Custom_Crop data module with configuration.

        Args:
            conf (object): Configuration object containing dataset paths and other settings.
        """
        self.ds = Custom_Crop_DS
        super().__init__(conf=conf)

r"""Run a species classifier.

This script is the classifier counterpart to detection/run_tf_detector_batch.py.
This script takes as input:
1) a detections JSON file, usually the output of run_tf_detector_batch.py or the
    output of the Batch API in the "Batch processing API output format"
2) a path to a directory containing crops of bounding boxes from the detections
    JSON file
3) a path to a PyTorch TorchScript compiled model file
4) (if the model is EfficientNet) an image size

By default, this script overwrites the detections JSON file, adding in
classification results. To output a new JSON file, use the --output argument.

Example usage:
    python run_classifier.py \
        detections.json \
        /path/to/crops \
        /path/to/model.pt \
        --image-size 224
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import os
from typing import Any, Optional

import pandas as pd
import PIL
import torch
import torch.utils
import torchvision as tv
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from classification import train_classifier


class SimpleDataset(torch.utils.data.Dataset):
    """Very simple dataset."""

    def __init__(self, img_files: Sequence[str],
                 images_dir: Optional[str] = None,
                 transform: Optional[Callable[[PIL.Image.Image], Any]] = None):
        """Creates a SimpleDataset."""
        self.img_files = img_files
        self.images_dir = images_dir
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Any, str]:
        """
        Returns: tuple, (img, img_file)
        """
        img_file = self.img_files[index]
        if self.images_dir is not None:
            img_path = os.path.join(self.images_dir, img_file)
        else:
            img_path = img_file
        img = default_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_file

    def __len__(self) -> int:
        return len(self.img_files)


def create_loader(cropped_images_dir: str,
                  detections_json_path: Optional[str],
                  img_size: int,
                  batch_size: int,
                  num_workers: int
                  ) -> torch.utils.data.DataLoader:
    """Creates a DataLoader.

    Args:
        cropped_images_dir: str, path to image crops
        detections: optional dict, detections JSON
    """
    crop_files = []

    if detections_json_path is None:
        # recursively find all files in cropped_images_dir
        for subdir, _, files in os.walk(cropped_images_dir):
            for file_name in files:
                rel_dir = os.path.relpath(subdir, cropped_images_dir)
                rel_file = os.path.join(rel_dir, file_name)
                crop_files.append(rel_file)

    else:
        # only find crops of images from detections JSON
        print('Loading detections JSON')
        with open(detections_json_path, 'r') as f:
            js = json.load(f)
        detections = {img['file']: img for img in js['images']}
        detector_version = js['info']['detector']
        
        for img_file, info_dict in tqdm(detections.items()):
            if 'detections' not in info_dict or info_dict['detections'] is None:
                continue
            for i in range(len(info_dict['detections'])):
                crop_filename = img_file + f'___crop{i:02d}_{detector_version}.jpg'
                crop_path = os.path.join(cropped_images_dir, crop_filename)
                if os.path.exists(crop_path):
                    crop_files.append(crop_filename)

    transform = tv.transforms.Compose([
        # resizes smaller edge to img_size
        tv.transforms.Resize(img_size, interpolation=PIL.Image.BICUBIC),
        tv.transforms.CenterCrop(img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=train_classifier.MEANS,
                                std=train_classifier.STDS, inplace=True)
    ])

    dataset = SimpleDataset(img_files=crop_files, images_dir=cropped_images_dir,
                            transform=transform)
    assert len(dataset) > 0
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True)
    return loader


def main(model_path: str,
         cropped_images_dir: str,
         output_csv_path: str,
         detections_json_path: Optional[str],
         classifier_categories_json_path: Optional[str],
         img_size: int,
         batch_size: int,
         num_workers: int,
         device_id:int=None) -> None:
    """Main function."""
    # evaluating with accimage is much faster than Pillow or Pillow-SIMD
    try:
        tv.set_image_backend('accimage')
    except:
        print('Warning: could not start accimage backend (ignore this if you\'re not using Linux)')

    # create dataset
    print('Creating data loader')
    loader = create_loader(
        cropped_images_dir, detections_json_path=detections_json_path,
        img_size=img_size, batch_size=batch_size, num_workers=num_workers)

    label_names = None
    if classifier_categories_json_path is not None:
        with open(classifier_categories_json_path, 'r') as f:
            categories = json.load(f)
        label_names = [categories[str(i)] for i in range(len(categories))]

    # create model
    print('Loading saved model')
    model = torch.jit.load(model_path)
    model, device = train_classifier.prep_device(model,device_id=device_id)

    test_epoch(model, loader, device=device, label_names=label_names,
               output_csv_path=output_csv_path)


def test_epoch(model: torch.nn.Module,
               loader: torch.utils.data.DataLoader,
               device: torch.device,
               label_names: Optional[Sequence[str]],
               output_csv_path: str) -> None:
    """Runs for 1 epoch.

    Writes results to the output CSV in batches.

    Args:
        model: torch.nn.Module
        loader: torch.utils.data.DataLoader
        device: torch.device
        label_names: optional list of str, label names
        output_csv_path: str
    """
    # set dropout and BN layers to eval mode
    model.eval()

    header = True
    mode = 'w'  # new file on first write

    with torch.no_grad():
        for inputs, img_files in tqdm(loader):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

            if label_names is None:
                label_names = [str(i) for i in range(probs.shape[1])]

            df = pd.DataFrame(data=probs, columns=label_names,
                              index=pd.Index(img_files, name='path'))
            df.to_csv(output_csv_path, index=True, header=header, mode=mode)

            if header:
                header = False
                mode = 'a'


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run classifier.')
    parser.add_argument(
        'model',
        help='path to TorchScript compiled model')
    parser.add_argument(
        'crops_dir',
        help='path to directory containing cropped images')
    parser.add_argument(
        'output',
        help='path to save CSV file with classifier results (can use .csv.gz '
             'extension for compression)')
    parser.add_argument(
        '-d', '--detections-json',
        help='path to detections JSON file, used to filter paths within '
             'crops_dir')
    parser.add_argument(
        '-c', '--classifier-categories',
        help='path to JSON file for classifier categories. If not given, '
             'classes are numbered "0", "1", "2", ...')
    parser.add_argument(
        '--image-size', type=int, default=224,
        help='size of input image to model, usually 224px, but may be larger '
             'especially for EfficientNet models')
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='batch size for evaluating model')
    parser.add_argument(
        '--device', type=int, default=None,
        help='preferred CUDA device')
    parser.add_argument(
        '--num-workers', type=int, default=8,
        help='# of workers for data loading')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(model_path=args.model,
         cropped_images_dir=args.crops_dir,
         output_csv_path=args.output,
         detections_json_path=args.detections_json,
         classifier_categories_json_path=args.classifier_categories,
         img_size=args.image_size,
         batch_size=args.batch_size,
         num_workers=args.num_workers,         
         device_id=args.device)

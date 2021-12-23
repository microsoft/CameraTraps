r"""Train a EfficientNet or ResNet classifier.

NOTE: Currently implementation of multi-label multi-class classification is
non-functional.



During training, start tensorboard from within the classification/ directory:
    tensorboard --logdir run --bind_all --samples_per_plugin scalars=0,images=0

Example usage:
    python train_classifier.py run_idfg /ssd/crops_sq \
        -m "efficientnet-b0" --pretrained --finetune --label-weighted \
        --epochs 50 --batch-size 512 --lr 1e-4 \
        --num-workers 12 --seed 123 \
        --logdir run_idfg
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import datetime
import json
import os
from typing import Any, Optional

import numpy as np
import PIL.Image
import sklearn.metrics
import torch
from torch.utils import tensorboard
import torchvision as tv
from torchvision.datasets.folder import default_loader
import tqdm

from classification import efficientnet, evaluate_model
from classification.train_utils import (
    HeapItem, recall_from_confusion_matrix, add_to_heap, fig_to_img,
    imgs_with_confidences, load_dataset_csv, prefix_all_keys)
from visualization import plot_utils


# accimage backend is faster than Pillow/Pillow-SIMD, but occasionally crashes
# tv.set_image_backend('accimage')

# mean/std values from https://pytorch.org/docs/stable/torchvision/models.html
MEANS = np.asarray([0.485, 0.456, 0.406])
STDS = np.asarray([0.229, 0.224, 0.225])

VALID_MODELS = sorted(
    set(efficientnet.VALID_MODELS) |
    {'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50'})


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SimpleDataset(torch.utils.data.Dataset):
    """A simple dataset that simply returns images and labels."""

    def __init__(self,
                 img_files: Sequence[str],
                 labels: Sequence[Any],
                 sample_weights: Optional[Sequence[float]] = None,
                 img_base_dir: str = '',
                 transform: Optional[Callable[[PIL.Image.Image], Any]] = None,
                 target_transform: Optional[Callable[[Any], Any]] = None):
        """Creates a SimpleDataset."""
        self.img_files = img_files
        self.labels = labels
        self.sample_weights = sample_weights
        self.img_base_dir = img_base_dir
        self.transform = transform
        self.target_transform = target_transform

        self.len = len(img_files)
        assert len(labels) == self.len
        if sample_weights is not None:
            assert len(sample_weights) == self.len

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        """
        Args:
            index: int

        Returns: tuple, (sample, target) or (sample, target, sample_weight)
        """
        img_file = self.img_files[index]
        img = default_loader(os.path.join(self.img_base_dir, img_file))
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.sample_weights is not None:
            return img, target, img_file, self.sample_weights[index]
        return img, target, img_file

    def __len__(self) -> int:
        return self.len


def create_dataloaders(
        dataset_csv_path: str,
        label_index_json_path: str,
        splits_json_path: str,
        cropped_images_dir: str,
        img_size: int,
        multilabel: bool,
        label_weighted: bool,
        weight_by_detection_conf: bool | str,
        batch_size: int,
        num_workers: int,
        augment_train: bool
        ) -> tuple[dict[str, torch.utils.data.DataLoader], list[str]]:
    """
    Args:
        dataset_csv_path: str, path to CSV file with columns
            ['dataset', 'location', 'label'], where label is a comma-delimited
            list of labels
        splits_json_path: str, path to JSON file
        augment_train: bool, whether to shuffle/augment the training set

    Returns:
        datasets: dict, maps split to DataLoader
        label_names: list of str, label names in order of label id
    """
    df, label_names, split_to_locs = load_dataset_csv(
        dataset_csv_path, label_index_json_path, splits_json_path,
        multilabel=multilabel, label_weighted=label_weighted,
        weight_by_detection_conf=weight_by_detection_conf)

    # define the transforms
    normalize = tv.transforms.Normalize(mean=MEANS, std=STDS, inplace=True)
    train_transform = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(img_size),
        tv.transforms.RandomRotation(degrees=(-90, 90)),
        tv.transforms.RandomHorizontalFlip(p=0.5),
        tv.transforms.RandomVerticalFlip(p=0.1),
        tv.transforms.RandomGrayscale(p=0.1),
        tv.transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25),
        tv.transforms.ToTensor(),
        normalize
    ])
    test_transform = tv.transforms.Compose([
        # resizes smaller edge to img_size
        tv.transforms.Resize(img_size, interpolation=PIL.Image.BICUBIC),
        tv.transforms.CenterCrop(img_size),
        tv.transforms.ToTensor(),
        normalize
    ])

    dataloaders = {}
    for split, locs in split_to_locs.items():
        is_train = (split == 'train') and augment_train
        split_df = df[df['dataset_location'].isin(locs)]

        sampler: Optional[torch.utils.data.Sampler] = None
        weights = None
        if label_weighted or weight_by_detection_conf:
            # weights sums to:
            # - if weight_by_detection_conf: (# images in split - conf delta)
            # - otherwise: # images in split
            weights = split_df['weights'].to_numpy()
            if not weight_by_detection_conf:
                assert np.isclose(weights.sum(), len(split_df))
            if is_train:
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights, num_samples=len(split_df), replacement=True)
        elif is_train:
            # for normal (non-weighted) shuffling
            sampler = torch.utils.data.SubsetRandomSampler(range(len(split_df)))

        dataset = SimpleDataset(
            img_files=split_df['path'].tolist(),
            labels=split_df['label_index'].tolist(),
            sample_weights=weights,
            img_base_dir=cropped_images_dir,
            transform=train_transform if is_train else test_transform)
        assert len(dataset) > 0
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True)

    return dataloaders, label_names


def set_finetune(model: torch.nn.Module, model_name: str, finetune: bool
                 ) -> None:
    """Set the 'requires_grad' on each model parameter according to whether or
    not we are fine-tuning the model.
    """
    if finetune:
        if 'efficientnet' in model_name:
            final_layer = model._fc  # pylint: disable=protected-access
        else:  # torchvision resnet
            final_layer = model.fc
        assert isinstance(final_layer, torch.nn.Module)

        # set all parameters to not require gradients except final FC layer
        model.requires_grad_(False)
        for param in final_layer.parameters():
            param.requires_grad = True
    else:
        model.requires_grad_(True)


def build_model(model_name: str, num_classes: int, pretrained: bool | str,
                finetune: bool) -> torch.nn.Module:
    """Creates a model with an EfficientNet or ResNet base. The model outputs
    unnormalized logits.

    Args:
        model_name: str, name of EfficientNet or Resnet model
        num_classes: int, number of classes for output layer
        pretrained: bool or str, (bool) whether to initialize to ImageNet
            weights, (str) path to checkpoint
        finetune: bool, whether to freeze all layers except the final FC layer

    Returns: torch.nn.Module, model loaded on CPU
    """
    assert model_name in VALID_MODELS

    if 'efficientnet' in model_name:
        if pretrained is True:
            model = efficientnet.EfficientNet.from_pretrained(
                model_name, num_classes=num_classes)
        else:
            model = efficientnet.EfficientNet.from_name(
                model_name, num_classes=num_classes)
    else:
        model_class = getattr(tv.models, model_name)
        model = model_class(pretrained=(pretrained is True))

        # replace final fully-connected layer (which has 1000 ImageNet classes)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    if isinstance(pretrained, str):
        print(f'Loading saved weights from {pretrained}')
        ckpt = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(ckpt['model'])

    assert all(p.requires_grad for p in model.parameters())
    set_finetune(model=model, model_name=model_name, finetune=finetune)
    return model


def prep_device(model: torch.nn.Module, device_id:int=None) -> tuple[torch.nn.Module, torch.device]:
    """Place model on appropriate device.

    Args:
        model: torch.nn.Module, not already wrapped with DataParallel

    Returns:
        model: torch.nn.Module, model placed on <device>, wrapped with
            DataParallel if more than 1 GPU is found
        device: torch.device, 'cuda:0' if GPU is found, otherwise 'cpu'
    """
    # detect GPU, use all if available
    if torch.cuda.is_available():
        print('CUDA available')
        if device_id is not None:
            print('Starting CUDA device {}'.format(device_id))
            device = torch.device('cuda:{}'.format(str(device_id)))            
        else:
            device = torch.device('cuda:0')
            torch.backends.cudnn.benchmark = True
            device_ids = list(range(torch.cuda.device_count()))
            if len(device_ids) > 1:
                print('Found multiple devices, enabling data parallelism ({})'.format(str(device_ids)))
                model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        print('CUDA not available, running on the CPU')
        device = torch.device('cpu')
    model.to(device)  # in-place
    return model, device


def main(dataset_dir: str,
         cropped_images_dir: str,
         multilabel: bool,
         model_name: str,
         pretrained: bool | str,
         finetune: int,
         label_weighted: bool,
         weight_by_detection_conf: bool | str,
         epochs: int,
         batch_size: int,
         lr: float,
         weight_decay: float,
         num_workers: int,
         logdir: str,
         log_extreme_examples: int,
         seed: Optional[int] = None) -> None:
    """Main function."""
    # input validation
    assert os.path.exists(dataset_dir)
    assert os.path.exists(cropped_images_dir)
    if isinstance(weight_by_detection_conf, str):
        assert os.path.exists(weight_by_detection_conf)
    if isinstance(pretrained, str):
        assert os.path.exists(pretrained)

    # set seed
    seed = np.random.randint(10_000) if seed is None else seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # create logdir and save params
    params = dict(locals())  # make a copy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # '20200722_110816'
    logdir = os.path.join(logdir, timestamp)
    os.makedirs(logdir, exist_ok=True)
    print('Created logdir:', logdir)
    params_json_path = os.path.join(logdir, 'params.json')
    with open(params_json_path, 'w') as f:
        json.dump(params, f, indent=1)

    if 'efficientnet' in model_name:
        img_size = efficientnet.EfficientNet.get_image_size(model_name)
    else:
        img_size = 224

    # create dataloaders and log the index_to_label mapping
    print('Creating dataloaders')
    loaders, label_names = create_dataloaders(
        dataset_csv_path=os.path.join(dataset_dir, 'classification_ds.csv'),
        label_index_json_path=os.path.join(dataset_dir, 'label_index.json'),
        splits_json_path=os.path.join(dataset_dir, 'splits.json'),
        cropped_images_dir=cropped_images_dir,
        img_size=img_size,
        multilabel=multilabel,
        label_weighted=label_weighted,
        weight_by_detection_conf=weight_by_detection_conf,
        batch_size=batch_size,
        num_workers=num_workers,
        augment_train=True)

    writer = tensorboard.SummaryWriter(logdir)

    # create model
    model = build_model(model_name, num_classes=len(label_names),
                        pretrained=pretrained, finetune=finetune > 0)
    model, device = prep_device(model)

    # define loss function and optimizer
    loss_fn: torch.nn.Module
    if multilabel:
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    # using EfficientNet training defaults
    # - batch norm momentum: 0.99
    # - optimizer: RMSProp, decay 0.9 and momentum 0.9
    # - epochs: 350
    # - learning rate: 0.256, decays by 0.97 every 2.4 epochs
    # - weight decay: 1e-5
    optimizer: torch.optim.Optimizer
    if 'efficientnet' in model_name:
        optimizer = torch.optim.RMSprop(model.parameters(), lr, alpha=0.9,
                                        momentum=0.9, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=1, gamma=0.97 ** (1 / 2.4))
    else:  # resnet
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9,
                                    weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=8, gamma=0.1)  # lower every 8 epochs

    best_epoch_metrics: dict[str, float] = {}
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], epoch)

        if epoch > 0 and finetune == epoch:
            print('Turning off fine-tune!')
            set_finetune(model, model_name, finetune=False)

        print('- train:')
        train_metrics, train_heaps, train_cm = run_epoch(
            model, loader=loaders['train'], weighted=False, device=device,
            loss_fn=loss_fn, finetune=finetune > epoch, optimizer=optimizer,
            k_extreme=log_extreme_examples)
        train_metrics = prefix_all_keys(train_metrics, prefix='train/')
        log_run('train', epoch, writer, label_names,
                metrics=train_metrics, heaps=train_heaps, cm=train_cm)
        del train_heaps

        print('- val:')
        val_metrics, val_heaps, val_cm = run_epoch(
            model, loader=loaders['val'], weighted=label_weighted,
            device=device, loss_fn=loss_fn, k_extreme=log_extreme_examples)
        val_metrics = prefix_all_keys(val_metrics, prefix='val/')
        log_run('val', epoch, writer, label_names,
                metrics=val_metrics, heaps=val_heaps, cm=val_cm)
        del val_heaps

        lr_scheduler.step()  # decrease the learning rate

        if val_metrics['val/acc_top1'] > best_epoch_metrics.get('val/acc_top1', 0):  # pylint: disable=line-too-long
            filename = os.path.join(logdir, f'ckpt_{epoch}.pt')
            print(f'New best model! Saving checkpoint to {filename}')
            state = {
                'epoch': epoch,
                'model': getattr(model, 'module', model).state_dict(),
                'val/acc': val_metrics['val/acc_top1'],
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, filename)
            best_epoch_metrics.update(train_metrics)
            best_epoch_metrics.update(val_metrics)
            best_epoch_metrics['epoch'] = epoch

            print('- test:')
            test_metrics, test_heaps, test_cm = run_epoch(
                model, loader=loaders['test'], weighted=label_weighted,
                device=device, loss_fn=loss_fn, k_extreme=log_extreme_examples)
            test_metrics = prefix_all_keys(test_metrics, prefix='test/')
            log_run('test', epoch, writer, label_names,
                    metrics=test_metrics, heaps=test_heaps, cm=test_cm)
            del test_heaps

        # stop training after 8 epochs without improvement
        if epoch >= best_epoch_metrics['epoch'] + 8:
            break

    hparams_dict = {
        'model_name': model_name,
        'multilabel': multilabel,
        'finetune': finetune,
        'batch_size': batch_size,
        'epochs': epochs
    }
    metric_dict = prefix_all_keys(best_epoch_metrics, prefix='hparam/')
    writer.add_hparams(hparam_dict=hparams_dict, metric_dict=metric_dict)
    writer.close()

    # do a complete evaluation run
    best_epoch = best_epoch_metrics['epoch']
    evaluate_model.main(
        params_json_path=params_json_path,
        ckpt_path=os.path.join(logdir, f'ckpt_{best_epoch}.pt'),
        output_dir=logdir, splits=evaluate_model.SPLITS)


def log_run(split: str, epoch: int, writer: tensorboard.SummaryWriter,
            label_names: Sequence[str], metrics: MutableMapping[str, float],
            heaps: Optional[Mapping[str, Mapping[int, list[HeapItem]]]],
            cm: np.ndarray) -> None:
    """Logs the outputs (metrics, confusion matrix, tp/fp/fn images) from a
    single epoch run to Tensorboard.

    Args:
        metrics: dict, keys already prefixed with {split}/
    """
    per_label_recall = recall_from_confusion_matrix(cm, label_names)
    metrics.update(prefix_all_keys(per_label_recall, f'{split}/label_recall/'))

    # log metrics
    for metric, value in metrics.items():
        writer.add_scalar(metric, value, epoch)

    # log confusion matrix
    cm_fig = plot_utils.plot_confusion_matrix(cm, classes=label_names,
                                              normalize=True)
    cm_fig_img = fig_to_img(cm_fig)
    writer.add_image(tag=f'confusion_matrix/{split}', img_tensor=cm_fig_img,
                     global_step=epoch, dataformats='HWC')

    # log tp/fp/fn images
    if heaps is not None:
        for heap_type, heap_dict in heaps.items():
            log_images_with_confidence(writer, heap_dict, label_names,
                                       epoch=epoch, tag=f'{split}/{heap_type}')
    writer.flush()


def log_images_with_confidence(
        writer: tensorboard.SummaryWriter,
        heap_dict: Mapping[int, list[HeapItem]],
        label_names: Sequence[str],
        epoch: int,
        tag: str) -> None:
    """
    Note: performs image normalization in-place

    Args:
        writer: tensorboard.SummaryWriter
        heap_dict: dict, maps label_id to list of HeapItem, where each HeapItem
            data is a tuple (img, target, top3_conf, top3_preds, img_file)
        label_names: list of str, label names in order of label id
        epoch: int
        tag: str
    """
    # for every image: undo normalization, clamp to [0, 1], CHW -> HWC
    # - cannot be in-place, because the HeapItem might be in multiple heaps
    unnormalize = tv.transforms.Normalize(mean=-MEANS/STDS, std=1.0/STDS)
    for label_id, heap in heap_dict.items():
        label_name = label_names[label_id]

        imgs_list = []
        for item in sorted(heap, reverse=True):  # sort largest to smallest
            img = item.data[0].float()  # clamp() only supports fp32 on CPU
            img = unnormalize(img).clamp_(0, 1).permute(1, 2, 0)
            imgs_list.append((img, *item.data[1:]))

        fig, img_files = imgs_with_confidences(imgs_list, label_names)

        # writer.add_figure() has issues => using add_image() instead
        # writer.add_figure(f'{label_name}/{tag}', fig, global_step=epoch)
        writer.add_image(f'{label_name}/{tag}', fig_to_img(fig),
                         global_step=epoch, dataformats='HWC')
        writer.add_text(f'{label_name}/{tag}_files', '\n\n'.join(img_files),
                        global_step=epoch)


def track_extreme_examples(tp_heaps: dict[int, list[HeapItem]],
                           fp_heaps: dict[int, list[HeapItem]],
                           fn_heaps: dict[int, list[HeapItem]],
                           inputs: torch.Tensor,
                           labels: torch.Tensor,
                           img_files: Sequence[str],
                           logits: torch.Tensor,
                           k: int = 5) -> None:
    """Updates the k most extreme true-positive (tp), false-positive (fp), and
    false-negative (fn) examples with examples from this batch.

    Each HeapItem's data attribute is a tuple of:
    - img: torch.Tensor, shape [3, H, W], type float16, values in [0, 1]
    - label: int
    - top3_conf: list of float
    - top3_preds: list of float
    - img_file: str

    Args:
        *_heaps: dict, maps label_id (int) to heap of HeapItems
        inputs: torch.Tensor, shape [batch_size, 3, H, W]
        labels: torch.Tensor, shape [batch_size]
        img_files: list of str
        logits: torch.Tensor, shape [batch_size, num_classes]
        k: int, number of examples to track
    """
    with torch.no_grad():
        inputs = inputs.detach().to(device='cpu', dtype=torch.float16)
        labels_list = labels.tolist()  # new var name to satisfy mypy
        batch_probs = torch.nn.functional.softmax(logits, dim=1).cpu()
        zipped = zip(inputs, labels_list, batch_probs, img_files)  # all on CPU
        for img, label, confs, img_file in zipped:
            label_conf = confs[label].item()

            top3_conf, top3_preds = confs.topk(3)
            top3_conf = top3_conf.tolist()
            top3_preds = top3_preds.tolist()

            data = [img, label, top3_conf, top3_preds, img_file]
            if top3_preds[0] == label:  # true positive
                item = HeapItem(priority=label_conf - top3_conf[1], data=data)
                add_to_heap(tp_heaps[label], item, k=k)
            else:
                # false positive for top3_pred[0]
                # false negative for label
                item = HeapItem(priority=top3_conf[0] - label_conf, data=data)
                add_to_heap(fp_heaps[top3_preds[0]], item, k=k)
                add_to_heap(fn_heaps[label], item, k=k)


def correct(outputs: torch.Tensor, labels: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            top: Sequence[int] = (1,)) -> dict[int, float]:
    """
    Args:
        outputs: torch.Tensor, shape [N, num_classes],
            either logits (pre-softmax) or probabilities
        labels: torch.Tensor, shape [N]
        weights: optional torch.Tensor, shape [N]
        top: tuple of int, list of values of k for calculating top-K accuracy

    Returns: dict, maps k to (weighted) # of correct predictions @ each k
    """
    with torch.no_grad():
        # preds and labels both have shape [N, k]
        _, preds = outputs.topk(k=max(top), dim=1, largest=True, sorted=True)
        labels = labels.view(-1, 1).expand_as(preds)

        corrects = preds.eq(labels).cumsum(dim=1)  # shape [N, k]
        if weights is None:
            corrects = corrects.sum(dim=0)  # shape [k]
        else:
            corrects = weights.matmul(corrects.to(weights.dtype))  # shape [k]
        tops = {k: corrects[k - 1].item() for k in top}
    return tops


def run_epoch(model: torch.nn.Module,
              loader: torch.utils.data.DataLoader,
              weighted: bool,
              device: torch.device,
              top: Sequence[int] = (1, 3),
              loss_fn: Optional[torch.nn.Module] = None,
              finetune: bool = False,
              optimizer: Optional[torch.optim.Optimizer] = None,
              k_extreme: int = 0
              ) -> tuple[
                  dict[str, float],
                  Optional[dict[str, dict[int, list[HeapItem]]]],
                  np.ndarray
              ]:
    """Runs for 1 epoch.

    Args:
        model: torch.nn.Module
        loader: torch.utils.data.DataLoader
        weighted: bool, whether to use sample weights in calculating loss and
            accuracy
        device: torch.device
        top: tuple of int, list of values of k for calculating top-K accuracy
        loss_fn: optional loss function, calculates per-example loss
        finetune: bool, if true sets model's dropout and BN layers to eval mode
        optimizer: optional optimizer
        k_extreme: int, # of tp/fp/fn examples to track for each label

    Returns:
        metrics: dict, metrics from epoch, contains keys:
            'loss': float, mean per-example loss over entire epoch,
                only included if loss_fn is not None
            'acc_top{k}': float, accuracy@k over the entire epoch
        heaps: dict, keys are ['tp', 'fp', 'fn'], values are heap_dicts,
            each heap_dict maps label_id (int) to a heap of <= 5 HeapItems with
            data attribute (img, target, top3_conf, top3_preds, img_file)
            - 'tp': priority is the difference between target confidence and
                2nd highest confidence
            - 'fp': priority is the difference between highest confidence and
                target confidence
            - 'fn': same as 'fp'
        confusion_matrix: np.ndarray, shape [num_classes, num_classes],
            C[i, j] = # of samples with true label i, predicted as label j
    """
    if optimizer is not None:
        assert loss_fn is not None

    # if evaluating or finetuning, set dropout and BN layers to eval mode
    model.train(optimizer is not None and not finetune)

    if loss_fn is not None:
        losses = AverageMeter()
    accuracies_topk = {k: AverageMeter() for k in top}  # acc@k

    # for each label, track k_extreme most-confident and least-confident images
    if k_extreme > 0:
        tp_heaps: dict[int, list[HeapItem]] = defaultdict(list)
        fp_heaps: dict[int, list[HeapItem]] = defaultdict(list)
        fn_heaps: dict[int, list[HeapItem]] = defaultdict(list)

    all_labels = np.zeros(len(loader.dataset), dtype=np.int32)
    all_preds = np.zeros_like(all_labels)
    end_i = 0

    tqdm_loader = tqdm.tqdm(loader)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in tqdm_loader:
            if weighted:
                inputs, labels, img_files, weights = batch
                weights = weights.to(device, non_blocking=True)
            else:
                # even if batch contains sample weights, don't use them
                inputs, labels, img_files = batch[0:3]
                weights = None

            inputs = inputs.to(device, non_blocking=True)

            batch_size = labels.size(0)
            start_i = end_i
            end_i = start_i + batch_size
            all_labels[start_i:end_i] = labels

            desc = []
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            all_preds[start_i:end_i] = outputs.detach().argmax(dim=1).cpu()

            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                if weights is not None:
                    loss *= weights
                loss = loss.mean()
                losses.update(loss.item(), n=batch_size)
                desc.append(f'Loss {losses.val:.4f} ({losses.avg:.4f})')
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            top_correct = correct(outputs, labels, weights=weights, top=top)
            for k, acc in accuracies_topk.items():
                acc.update(top_correct[k] * (100. / batch_size), n=batch_size)
                desc.append(f'Acc@{k} {acc.val:.3f} ({acc.avg:.3f})')
            tqdm_loader.set_description(' '.join(desc))

            if k_extreme > 0:
                track_extreme_examples(tp_heaps, fp_heaps, fn_heaps, inputs,
                                       labels, img_files, outputs, k=k_extreme)

    num_classes = outputs.size(1)
    confusion_matrix = sklearn.metrics.confusion_matrix(
        all_labels, all_preds, labels=np.arange(num_classes))

    metrics = {}
    if loss_fn is not None:
        metrics['loss'] = losses.avg
    for k, acc in accuracies_topk.items():
        metrics[f'acc_top{k}'] = acc.avg
    heaps = None
    if k_extreme > 0:
        heaps = {'tp': tp_heaps, 'fp': fp_heaps, 'fn': fn_heaps}
    return metrics, heaps, confusion_matrix


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Trains classifier.')
    parser.add_argument(
        'dataset_dir',
        help='path to directory containing: 1) classification dataset CSV, '
             '2) label index JSON, 3) splits JSON')
    parser.add_argument(
        'cropped_images_dir',
        help='path to local directory where image crops are saved')
    parser.add_argument(
        '--multilabel', action='store_true',
        help='for multi-label, multi-class classification')
    parser.add_argument(
        '-m', '--model-name', default='efficientnet-b0',
        choices=VALID_MODELS,
        help='which EfficientNet or Resnet model')
    parser.add_argument(
        '--pretrained', nargs='?', const=True, default=False,
        help='start with ImageNet pretrained model or a specific checkpoint')
    parser.add_argument(
        '--finetune', type=int, default=0,
        help='only fine tune the final fully-connected layer for the first '
             '<finetune> epochs')
    parser.add_argument(
        '--label-weighted', action='store_true',
        help='weight training samples to balance labels')
    parser.add_argument(
        '--weight-by-detection-conf', nargs='?', const=True, default=False,
        help='weight training examples by detection confidence. '
             'Optionally takes a .npz file for isotonic calibration.')
    parser.add_argument(
        '--epochs', type=int, default=0,
        help='number of epochs for training, 0 for eval-only')
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='batch size for both training and eval')
    parser.add_argument(
        '--lr', type=float,
        help='initial learning rate, defaults to (0.016 * batch_size / 256)')
    parser.add_argument(
        '--weight-decay', type=float, default=1e-5,
        help='weight decay')
    parser.add_argument(
        '--num-workers', type=int, default=8,
        help='# of workers for data loading')
    parser.add_argument(
        '--logdir', default='.',
        help='directory where TensorBoard logs and a params file are saved')
    parser.add_argument(
        '--log-extreme-examples', type=int, default=0,
        help='# of tp/fp/fn examples to log for each label and split per epoch')
    parser.add_argument(
        '--seed', type=int,
        help='random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if args.lr is None:
        args.lr = 0.016 * args.batch_size / 256  # based on TF models repo
    main(dataset_dir=args.dataset_dir,
         cropped_images_dir=args.cropped_images_dir,
         multilabel=args.multilabel,
         model_name=args.model_name,
         pretrained=args.pretrained,
         finetune=args.finetune,
         label_weighted=args.label_weighted,
         weight_by_detection_conf=args.weight_by_detection_conf,
         epochs=args.epochs,
         batch_size=args.batch_size,
         lr=args.lr,
         weight_decay=args.weight_decay,
         num_workers=args.num_workers,
         logdir=args.logdir,
         log_extreme_examples=args.log_extreme_examples,
         seed=args.seed)

r"""Train an EfficientNet classifier.

Currently implementation of multi-label multi-class classification is
non-functional.

During training, start tensorboard from within the classification/ directory:
    tensorboard --logdir run --bind_all --samples_per_plugin scalars=0,images=0

TODO:
- verify that finetuning is really only changing the final-layer weights

Example usage:
    python train_classifier.py run_idfg /ssd/crops_sq \
        -m "efficientnet-b0" --pretrained --finetune --label-weighted \
        --epochs 50 --batch-size 512 --lr 1e-4 \
        --num-workers 12 --seed 123 \
        --logdir run_idfg
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
    HeapItem, accuracy_from_confusion_matrix, add_to_heap, fig_to_img,
    imgs_with_confidences, load_dataset_csv, prefix_all_keys)
from visualization import plot_utils


# accimage backend is faster than Pillow/Pillow-SIMD, but occasionally crashes
# tv.set_image_backend('accimage')

MEANS = np.asarray([0.485, 0.456, 0.406])
STDS = np.asarray([0.229, 0.224, 0.225])


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

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
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
        batch_size: int,
        num_workers: int,
        augment_train: bool
        ) -> Tuple[Dict[str, torch.utils.data.DataLoader], List[str]]:
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
        multilabel=multilabel, weight_by_detection_conf=False,
        label_weighted=label_weighted)

    # define the transforms
    normalize = tv.transforms.Normalize(mean=MEANS, std=STDS, inplace=True)
    train_transform = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(img_size),
        tv.transforms.RandomHorizontalFlip(),
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

        sampler = None
        weights = None
        if label_weighted:
            # weights sums to the # of images in the split
            weights = split_df['weights'].to_numpy()
            if is_train:
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights, num_samples=len(split_df), replacement=True)

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


def log_metrics(writer: tensorboard.SummaryWriter, metrics: Dict[str, float],
                epoch: int, prefix: str = '') -> None:
    """Logs metrics to TensorBoard."""
    for metric, value in metrics.items():
        writer.add_scalar(f'{prefix}{metric}', value, epoch)


def build_model(model_name: str, num_classes: int, pretrained: bool,
                finetune: bool, dropout: float, ckpt_path: Optional[str] = None
                ) -> Tuple[torch.nn.Module, torch.device]:
    """Creates a model with an EfficientNet base.

    Args:
        model_name: str, name of efficient model
        num_classes: int, number of classes for output layer
        pretrained: bool, whether to initialize to ImageNet weights
        finetune: bool, whether to freeze all layers except the final FC layer
        dropout: float, dropout probability, TODO
        ckpt_path: optional str, path to checkpoint from which to load weights

    Returns:
        model: torch.nn.Module, model placed on the proper device with
            DataParallel if more than 1 GPU is found
        device: torch.device, 'cuda:0' if GPU is found, otherwise 'cpu'
    """
    if pretrained:
        assert ckpt_path is None
        model = efficientnet.EfficientNet.from_pretrained(
            model_name, num_classes=num_classes)
    else:
        model = efficientnet.EfficientNet.from_name(
            model_name, num_classes=num_classes)

    if ckpt_path is not None:
        print(f'Loading saved weights from {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])

    if finetune:
        # set all parameters to not require gradients except final FC layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model._fc.parameters():  # pylint: disable=protected-access
            param.requires_grad = True

    # detect GPU, use all if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = torch.device('cpu')
    model.to(device)  # in-place

    return model, device


def main(dataset_dir: str,
         cropped_images_dir: str,
         multilabel: bool,
         model_name: str,
         pretrained: bool,
         finetune: bool,
         label_weighted: bool,
         epochs: int,
         batch_size: int,
         lr: float,
         weight_decay: float,
         num_workers: int,
         seed: Optional[int] = None,
         logdir: str = '') -> None:
    """Main function."""
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
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=1)

    # create dataloaders and log the index_to_label mapping
    loaders, label_names = create_dataloaders(
        dataset_csv_path=os.path.join(dataset_dir, 'classification_ds.csv'),
        label_index_json_path=os.path.join(dataset_dir, 'label_index.json'),
        splits_json_path=os.path.join(dataset_dir, 'splits.json'),
        cropped_images_dir=cropped_images_dir,
        img_size=efficientnet.EfficientNet.get_image_size(model_name),
        multilabel=multilabel,
        label_weighted=label_weighted,
        batch_size=batch_size,
        num_workers=num_workers,
        augment_train=True)

    writer = tensorboard.SummaryWriter(logdir)

    # create model
    model, device = build_model(
        model_name, num_classes=len(label_names), pretrained=pretrained,
        finetune=finetune, dropout=0)

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
    optimizer = torch.optim.RMSprop(model.parameters(), lr, alpha=0.9,
                                    momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=1, gamma=0.97 ** (1 / 2.4))

    best_epoch_metrics: Dict[str, float] = {}
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], epoch)

        print('- train:')
        train_metrics, train_heaps, train_cm = run_epoch(
            model, loader=loaders['train'], weighted=False, device=device,
            loss_fn=loss_fn, finetune=finetune, optimizer=optimizer,
            return_extreme_images=True)
        train_per_label_acc = accuracy_from_confusion_matrix(
            train_cm, label_names)
        train_metrics.update(prefix_all_keys(train_per_label_acc, 'label_acc/'))
        train_metrics = prefix_all_keys(train_metrics, prefix='train/')
        log_metrics(writer, train_metrics, epoch)
        log_confusion_matrix(train_cm, label_names=label_names, writer=writer,
                             tag='confusion_matrix/train', epoch=epoch)
        for heap_type, heap_dict in train_heaps.items():
            log_images_with_confidence(writer, heap_dict, label_names,
                                       epoch=epoch, tag=f'train/{heap_type}')

        print('- val:')
        val_metrics, val_heaps, val_cm = run_epoch(
            model, loader=loaders['val'], weighted=label_weighted,
            device=device, loss_fn=loss_fn, return_extreme_images=True)

        val_per_label_acc = accuracy_from_confusion_matrix(val_cm, label_names)
        val_metrics.update(prefix_all_keys(val_per_label_acc, 'label_acc/'))
        val_metrics = prefix_all_keys(val_metrics, prefix='val/')
        log_metrics(writer, val_metrics, epoch)
        log_confusion_matrix(val_cm, label_names=label_names, writer=writer,
                             tag='confusion_matrix/val', epoch=epoch)
        for heap_type, heap_dict in val_heaps.items():
            log_images_with_confidence(writer, heap_dict, label_names,
                                       epoch=epoch, tag=f'val/{heap_type}')
        writer.flush()

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
    evaluate_model.main(logdir, ckpt_name=f'ckpt_{best_epoch}.pt',
                        splits=evaluate_model.SPLITS)


def log_confusion_matrix(cm: np.ndarray, label_names: Sequence[str],
                         writer: tensorboard.SummaryWriter, tag: str,
                         epoch: int) -> None:
    """Log a confusion matrix in TensorBoard."""
    cm_fig = plot_utils.plot_confusion_matrix(
        cm, classes=label_names, normalize=True)
    cm_fig_img = fig_to_img(cm_fig)
    writer.add_image(tag, cm_fig_img, global_step=epoch, dataformats='HWC')


def log_images_with_confidence(
        writer: tensorboard.SummaryWriter,
        heap_dict: Mapping[int, List[HeapItem]],
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
            img = unnormalize(item.data[0]).clamp_(0, 1).permute(1, 2, 0)
            imgs_list.append((img, *item.data[1:]))

        fig, img_files = imgs_with_confidences(imgs_list, label_names)

        # writer.add_figure() has issues => using add_image() instead
        # writer.add_figure(f'{label_name}/{tag}', fig, global_step=epoch)
        writer.add_image(f'{label_name}/{tag}', fig_to_img(fig),
                         global_step=epoch, dataformats='HWC')
        writer.add_text(f'{label_name}/{tag}_files', '\n\n'.join(img_files),
                        global_step=epoch)


def track_extreme_examples(tp_heaps: Dict[int, List[HeapItem]],
                           fp_heaps: Dict[int, List[HeapItem]],
                           fn_heaps: Dict[int, List[HeapItem]],
                           inputs: torch.Tensor,
                           labels: torch.Tensor,
                           img_files: Sequence[str],
                           logits: torch.Tensor) -> None:
    """Updates the 5 most extreme true-positive (tp), false-positive (fp), and
    false-negative (fn) examples with examples from this batch.

    Each HeapItem's data attribute is a tuple of:
    - img: torch.Tensor, shape [H, W, 3], type float32, values in [0, 1]
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
    """
    with torch.no_grad():
        inputs = inputs.detach().cpu()
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
                add_to_heap(tp_heaps[label], item, k=5)
            else:
                # false positive for top3_pred[0]
                # false negative for label
                item = HeapItem(priority=top3_conf[0] - label_conf, data=data)
                add_to_heap(fp_heaps[top3_preds[0]], item, k=5)
                add_to_heap(fn_heaps[label], item, k=5)


def correct(outputs: torch.Tensor, labels: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
            top: Sequence[int] = (1,)) -> Dict[int, float]:
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
              return_extreme_images: bool = False
              ) -> Tuple[
                  Dict[str, float],
                  Dict[str, Dict[int, List[HeapItem]]],
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

    # for each label, track 5 most-confident and least-confident examples
    tp_heaps: Dict[int, List[HeapItem]] = defaultdict(list)
    fp_heaps: Dict[int, List[HeapItem]] = defaultdict(list)
    fn_heaps: Dict[int, List[HeapItem]] = defaultdict(list)

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

            batch_size = labels.size(0)
            start_i = end_i
            end_i = start_i + batch_size
            all_labels[start_i:end_i] = labels

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)

            desc = []
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

            if return_extreme_images:
                track_extreme_examples(tp_heaps, fp_heaps, fn_heaps, inputs,
                                       labels, img_files, outputs)

    num_classes = outputs.size(1)
    confusion_matrix = sklearn.metrics.confusion_matrix(
        all_labels, all_preds, labels=np.arange(num_classes))

    metrics = {}
    if loss_fn is not None:
        metrics['loss'] = losses.avg
    for k, acc in accuracies_topk.items():
        metrics[f'acc_top{k}'] = acc.avg
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
        choices=efficientnet.VALID_MODELS,
        help='which EfficientNet model')
    parser.add_argument(
        '--pretrained', action='store_true',
        help='start with pretrained model')
    parser.add_argument(
        '--finetune', action='store_true',
        help='only fine tune the final fully-connected layer')
    parser.add_argument(
        '--label-weighted', action='store_true',
        help='weight training samples to balance labels')
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
        help='number of workers for data loading')
    parser.add_argument(
        '--seed', type=int,
        help='random seed')
    parser.add_argument(
        '--logdir', default='.',
        help='directory where TensorBoard logs and a params file are saved')
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
         epochs=args.epochs,
         batch_size=args.batch_size,
         lr=args.lr,
         weight_decay=args.weight_decay,
         num_workers=args.num_workers,
         seed=args.seed,
         logdir=args.logdir)

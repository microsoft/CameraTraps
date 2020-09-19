r"""Evaluate a species classifier.

Currently implementation of multi-label multi-class classification is
non-functional.

Outputs the following files:

1) outputs_{split}.csv, one file per split, contains columns:
    - 'path': str, path to cropped image
    - 'label': str
    - 'weight': float
    - [label names]: float, confidence in each label

2) overall_metrics.csv, contains columns:
    - 'split': str
    - 'loss': float, mean per-example loss over entire epoch
    - 'acc_top{k}': float, accuracy@k over the entire epoch
    - 'loss_weighted' and 'acc_weighted_top{k}': float, weighted versions

3) confusion_matrices.npz
    - keys ['train', 'val', 'test']
    - values are np.ndarray, confusion matrices

4) label_stats.csv, per-label statistics, columns
    - 'split': str
    - 'label': str
    - 'precision': float
    - 'recall': float


Example usage:
    python evaluate_model.py run_idfg/logs/20200803_145515 ckpt_6.pt
"""
import argparse
import json
import os
from pprint import pprint
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torchvision
import tqdm

from classification import efficientnet, train_classifier


SPLITS = ['train', 'val', 'test']


def check_override(params: Mapping[str, Any], key: str,
                   override: Optional[Any]) -> Any:
    """Return desired value, with optional override."""
    if override is None:
        return params[key]
    saved = params.get(key, None)
    print(f'Overriding saved {key}. Saved: {saved}. Override with: {override}.')
    return override


def trace_model(model_name: str, ckpt_path: str, num_classes: int,
                img_size: int) -> str:
    """Use TorchScript tracing to compile trained model into standalone file.

    For now, we have to use tracing instead of scripting. See
        https://github.com/lukemelas/EfficientNet-PyTorch/issues/89
        https://github.com/lukemelas/EfficientNet-PyTorch/issues/218

    Args:
        model_name: str
        ckpt_path: str, path to checkpoint file
        num_labels: int, number of classification classes
        img_size: int, size of input image, used for tracing

    Returns: str, name of file for compiled model. For example, if ckpt_path is
        '/path/to/ckpt_16.pt', then the returned path is
        '/path/to/ckpt_16_compiled.pt'.
    """
    root, ext = os.path.splitext(ckpt_path)
    compiled_path = root + '_compiled' + ext
    if os.path.exists(compiled_path):
        return compiled_path

    model = train_classifier.build_model(model_name, num_classes=num_classes,
                                         pretrained=ckpt_path, finetune=False)
    if 'efficientnet' in model_name:
        model.set_swish(memory_efficient=False)
    model.eval()

    ex_img = torch.rand(1, 3, img_size, img_size)
    scripted_model = torch.jit.trace(model, (ex_img,))

    scripted_model.save(compiled_path)
    print('Saved TorchScript compiled model to', compiled_path)
    return compiled_path


def main(params_json_path: str, ckpt_path: str, output_dir: str,
         splits: Sequence[str], label_index_json_path: Optional[str] = None,
         **kwargs: Any) -> None:
    """Main function."""
    # input validation
    assert os.path.exists(params_json_path)
    assert os.path.exists(ckpt_path)

    # evaluating with accimage is much faster than Pillow or Pillow-SIMD
    torchvision.set_image_backend('accimage')

    # create output directory
    if not os.path.exists(output_dir):
        print('Creating output directory:', output_dir)
        os.makedirs(output_dir, exist_ok=True)

    with open(params_json_path, 'r') as f:
        params = json.load(f)
    pprint(params)

    # override saved params with kwargs
    for key, new in kwargs.items():
        if new is None:
            continue
        if key in params:
            saved = params[key]
            print(f'Overriding saved {key}. Saved: {saved}. '
                  f'Override with: {new}.')
        else:
            print(f'Did not find {key} in saved params. Using value {new}.')
        params[key] = new

    model_name: str = params['model_name']
    dataset_dir: str = params['dataset_dir']

    if 'efficientnet' in model_name:
        img_size = efficientnet.EfficientNet.get_image_size(model_name)
    else:
        img_size = 224

    # For now, we don't weight crops by detection confidence during
    # evaluation. But consider changing this.
    loaders, label_names = train_classifier.create_dataloaders(
        dataset_csv_path=os.path.join(dataset_dir, 'classification_ds.csv'),
        label_index_json_path=os.path.join(dataset_dir, 'label_index.json'),
        splits_json_path=os.path.join(dataset_dir, 'splits.json'),
        cropped_images_dir=params['cropped_images_dir'],
        img_size=img_size,
        multilabel=params['multilabel'],
        label_weighted=params['label_weighted'],
        weight_by_detection_conf=False,
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        augment_train=False)
    num_labels = len(label_names)

    # create model
    compiled_path = trace_model(model_name, ckpt_path, num_labels, img_size)
    model = torch.jit.load(compiled_path)
    model, device = train_classifier.prep_device(model)

    # define loss function (criterion)
    loss_fn: torch.nn.Module
    if params['multilabel']:
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    split_metrics = {}
    split_label_stats = {}
    cms = {}
    for split in splits:
        print(f'Evaluating {split}...')
        df, metrics, cm = test_epoch(
            model, loaders[split], weighted=True, device=device,
            label_names=label_names, loss_fn=loss_fn)

        # this file ends up being huge, so we GZIP compress it
        output_csv_path = os.path.join(output_dir, f'outputs_{split}.csv.gz')
        df.to_csv(output_csv_path, index=False, compression='gzip')

        split_metrics[split] = metrics
        cms[split] = cm
        split_label_stats[split] = calc_per_label_stats(cm, label_names)

        # double check that the accuracy metrics are computed properly
        preds = df[label_names].to_numpy().argmax(axis=1)
        preds = np.asarray(label_names)[preds]
        assert np.isclose(metrics['acc_top1'] / 100.,
                          sum(preds == df['label']) / len(df))
        assert np.isclose(metrics['acc_weighted_top1'] / 100.,
                          sum((preds == df['label']) * df['weight']) / len(df))

    metrics_df = pd.concat(split_metrics, names=['split']).unstack(level=1)
    metrics_df.to_csv(os.path.join(output_dir, 'overall_metrics.csv'))

    # save the confusion matrices to .npz
    npz_path = os.path.join(output_dir, 'confusion_matrices.npz')
    np.savez_compressed(npz_path, **cms)

    # save per-label statistics
    label_stats_df = pd.concat(
        split_label_stats, names=['split', 'label']).reset_index()
    label_stats_csv_path = os.path.join(output_dir, 'label_stats.csv')
    label_stats_df.to_csv(label_stats_csv_path, index=False)


def calc_per_label_stats(cm: np.ndarray, label_names: Sequence[str]
                         ) -> pd.DataFrame:
    """
    Args:
        cm: np.ndarray, confusion matrix C such that C[i,j] is the # of
            observations known to be in group i and predicted to be in group j
        label_names: list of str, label names in order of label id

    Returns: pd.DataFrame, index 'label', columns ['precision', 'recall']
    """
    df = pd.DataFrame()
    df['label'] = label_names
    df['precision'] = np.diag(cm) / (cm.sum(axis=0) + 1e-8)
    df['recall'] = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    df.set_index('label', inplace=True)
    return df


def test_epoch(model: torch.nn.Module,
               loader: torch.utils.data.DataLoader,
               weighted: bool,
               device: torch.device,
               label_names: Sequence[str],
               top: Sequence[int] = (1, 3),
               loss_fn: Optional[torch.nn.Module] = None,
               ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Runs for 1 epoch.

    Args:
        model: torch.nn.Module
        loader: torch.utils.data.DataLoader
        weighted: bool, whether to calculate weighted accuracy statistics
        device: torch.device
        label_names: list of str, label names in order of label id
        top: tuple of int, list of values of k for calculating top-K accuracy
        loss_fn: optional loss function, calculates per-example loss

    Returns:
        df: pd.DataFrame, columns ['img_file', 'label', 'weight', label_names]
        metrics: pd.Series, type float, index includes:
            'loss': mean per-example loss over entire epoch,
                only included if loss_fn is not None
            'acc_top{k}': accuracy@k over the entire epoch
            'loss_weighted' and 'acc_weighted_top{k}': weighted versions, only
                included if weighted=True
        cm: np.ndarray, confusion matrix C such that C[i,j] is the # of
            observations known to be in group i and predicted to be in group j
    """
    # set dropout and BN layers to eval mode
    model.eval()

    if loss_fn is not None:
        losses = train_classifier.AverageMeter()
    accuracies_topk = {k: train_classifier.AverageMeter() for k in top}  # acc@k
    if weighted:
        accs_weighted = {k: train_classifier.AverageMeter() for k in top}
        losses_weighted = train_classifier.AverageMeter()

    num_examples = len(loader.dataset)
    num_labels = len(label_names)

    all_img_files = []
    all_probs = np.zeros([num_examples, len(label_names)], dtype=np.float32)
    all_labels = np.zeros(num_examples, dtype=np.int32)
    if weighted:
        all_weights = np.zeros(num_examples, dtype=np.float32)

    end_i = 0
    tqdm_loader = tqdm.tqdm(loader)
    with torch.no_grad():
        for batch in tqdm_loader:
            if weighted:
                inputs, labels, img_files, weights = batch
            else:
                # even if batch contains sample weights, don't use them
                inputs, labels, img_files = batch[0:3]
                weights = None

            all_img_files.append(img_files)

            batch_size = labels.size(0)
            start_i = end_i
            end_i = start_i + batch_size
            all_labels[start_i:end_i] = labels
            if weighted:
                all_weights[start_i:end_i] = weights
                weights = weights.to(device, non_blocking=True)

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs[start_i:end_i] = probs.cpu()

            desc = []
            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                losses.update(loss.mean().item(), n=batch_size)
                desc.append(f'Loss {losses.val:.4f} ({losses.avg:.4f})')
                if weights is not None:
                    loss_weighted = (loss * weights).mean()
                    losses_weighted.update(loss_weighted.item(), n=batch_size)

            top_correct = train_classifier.correct(
                outputs, labels, weights=None, top=top)
            for k, acc in accuracies_topk.items():
                acc.update(top_correct[k] * (100. / batch_size), n=batch_size)
                desc.append(f'Acc@{k} {acc.val:.3f} ({acc.avg:.3f})')

            if weighted:
                top_correct = train_classifier.correct(
                    outputs, labels, weights=weights, top=top)
                for k, acc in accs_weighted.items():
                    acc.update(top_correct[k] * (100. / batch_size),
                               n=batch_size)
                    desc.append(f'Acc_w@{k} {acc.val:.3f} ({acc.avg:.3f})')

            tqdm_loader.set_description(' '.join(desc))

    # a confusion matrix C is such that C[i,j] is the # of observations known to
    # be in group i and predicted to be in group j.
    all_preds = all_probs.argmax(axis=1)
    cm = sklearn.metrics.confusion_matrix(
        y_true=all_labels, y_pred=all_preds, labels=np.arange(num_labels))

    df = pd.DataFrame()
    df['path'] = np.concatenate(all_img_files)
    df['label'] = list(map(label_names.__getitem__, all_labels))
    df['weight'] = all_weights
    df[label_names] = all_probs

    metrics = {}
    if loss_fn is not None:
        metrics['loss'] = losses.avg
        if weighted:
            metrics['loss_weighted'] = losses_weighted.avg
    for k, acc in accuracies_topk.items():
        metrics[f'acc_top{k}'] = acc.avg
    if weighted:
        for k, acc in accs_weighted.items():
            metrics[f'acc_weighted_top{k}'] = acc.avg
    return df, pd.Series(metrics), cm


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Evaluate trained model.')
    parser.add_argument(
        'params_json',
        help='path to params.json')
    parser.add_argument(
        'ckpt_path',
        help='path to checkpoint file')
    parser.add_argument(
        '-o', '--output-dir', required=True,
        help='(required) path to output directory')
    parser.add_argument(
        '--splits', nargs='*', choices=SPLITS, default=SPLITS,
        help='which splits to evaluate model on')

    override_group = parser.add_argument_group(
        'optional arguments to override values in params_json')
    override_group.add_argument(
        '--model-name',
        help='which EfficientNet or Resnet model')
    override_group.add_argument(
        '--batch-size', type=int,
        help='batch size for evaluating model')
    override_group.add_argument(
        '--num-workers', type=int,
        help='number of workers for data loading')
    override_group.add_argument(
        '--dataset-dir',
        help='path to directory containing classification_ds.csv, '
             'label_index.json, and splits.json')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(params_json_path=args.params_json, ckpt_path=args.ckpt_path,
         output_dir=args.output_dir, splits=args.splits,
         model_name=args.model_name, batch_size=args.batch_size,
         num_workers=args.num_workers, dataset_dir=args.dataset_dir)

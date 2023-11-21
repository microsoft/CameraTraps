r"""Train an EfficientNet classifier.

Currently implementation of multi-label multi-class classification is
non-functional.

During training, start tensorboard from within the classification/ directory:
    tensorboard --logdir run --bind_all --samples_per_plugin scalars=0,images=0

Example usage:
    python train_classifier_tf.py run_idfg /ssd/crops_sq \
        -m "efficientnet-b0" --pretrained --finetune --label-weighted \
        --epochs 50 --batch-size 512 --lr 1e-4 \
        --seed 123 \
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
import uuid

import numpy as np
import sklearn.metrics
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import tqdm

from classification.train_utils import (
    HeapItem, recall_from_confusion_matrix, add_to_heap, fig_to_img,
    imgs_with_confidences, load_dataset_csv, prefix_all_keys)
from visualization import plot_utils


AUTOTUNE = tf.data.experimental.AUTOTUNE

# match pytorch EfficientNet model names
EFFICIENTNET_MODELS: Mapping[str, Mapping[str, Any]] = {
    'efficientnet-b0': dict(cls='EfficientNetB0', img_size=224, dropout=0.2),
    'efficientnet-b1': dict(cls='EfficientNetB1', img_size=240, dropout=0.2),
    'efficientnet-b2': dict(cls='EfficientNetB2', img_size=260, dropout=0.3),
    'efficientnet-b3': dict(cls='EfficientNetB3', img_size=300, dropout=0.3),
    'efficientnet-b4': dict(cls='EfficientNetB4', img_size=380, dropout=0.4),
    'efficientnet-b5': dict(cls='EfficientNetB5', img_size=456, dropout=0.4),
    'efficientnet-b6': dict(cls='EfficientNetB6', img_size=528, dropout=0.5),
    'efficientnet-b7': dict(cls='EfficientNetB7', img_size=600, dropout=0.5)
}


def create_dataset(
        img_files: Sequence[str],
        labels: Sequence[Any],
        sample_weights: Optional[Sequence[float]] = None,
        img_base_dir: str = '',
        transform: Optional[Callable[[tf.Tensor], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        cache: bool | str = False
        ) -> tf.data.Dataset:
    """Create a tf.data.Dataset.

    The dataset returns elements (img, label, img_file, sample_weight) if
    sample_weights is not None, or (img, label, img_file) if
    sample_weights=None.
        img: tf.Tensor, shape [H, W, 3], type uint8
        label: tf.Tensor
        img_file: tf.Tensor, scalar, type str
        sample_weight: tf.Tensor, scalar, type float32

    Possible TODO: oversample the imbalanced classes
        see tf.data.experimental.sample_from_datasets

    Args:
        img_files: list of str, relative paths from img_base_dir
        labels: list of int if multilabel=False
        sample_weights: optional list of float
        img_base_dir: str, base directory for images
        transform: optional transform to apply to a single uint8 JPEG image
        target_transform: optional transform to apply to a single label
        cache: bool or str, cache images in memory if True, cache images to
            a file on disk if a str

    Returns: tf.data.Dataset
    """
    # images dataset
    img_ds = tf.data.Dataset.from_tensor_slices(img_files)
    img_ds = img_ds.map(lambda p: tf.io.read_file(img_base_dir + os.sep + p),
                        num_parallel_calls=AUTOTUNE)

    # for smaller disk / memory usage, we cache the raw JPEG bytes instead
    # of the decoded Tensor
    if isinstance(cache, str):
        img_ds = img_ds.cache(cache)
    elif cache:
        img_ds = img_ds.cache()

    # convert JPEG bytes to a 3D uint8 Tensor
    # keras EfficientNet already includes normalization from [0, 255] to [0, 1],
    #   so we don't need to do that here
    img_ds = img_ds.map(lambda img: tf.io.decode_jpeg(img, channels=3))

    if transform:
        img_ds = img_ds.map(transform, num_parallel_calls=AUTOTUNE)

    # labels dataset
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    if target_transform:
        labels_ds = labels_ds.map(target_transform, num_parallel_calls=AUTOTUNE)

    # img_files dataset
    img_files_ds = tf.data.Dataset.from_tensor_slices(img_files)

    if sample_weights is None:
        return tf.data.Dataset.zip((img_ds, labels_ds, img_files_ds))

    # weights dataset
    weights_ds = tf.data.Dataset.from_tensor_slices(sample_weights)
    return tf.data.Dataset.zip((img_ds, labels_ds, img_files_ds, weights_ds))


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
        augment_train: bool,
        cache_splits: Sequence[str]
        ) -> tuple[dict[str, tf.data.Dataset], list[str]]:
    """
    Args:
        dataset_csv_path: str, path to CSV file with columns
            ['dataset', 'location', 'label'], where label is a comma-delimited
            list of labels
        splits_json_path: str, path to JSON file
        augment_train: bool, whether to shuffle/augment the training set
        cache_splits: list of str, splits to cache
            training set is cached at /mnt/tempds/random_file_name
            validation and test sets are cached in memory

    Returns:
        datasets: dict, maps split to DataLoader
        label_names: list of str, label names in order of label id
    """
    df, label_names, split_to_locs = load_dataset_csv(
        dataset_csv_path, label_index_json_path, splits_json_path,
        multilabel=multilabel, label_weighted=label_weighted,
        weight_by_detection_conf=weight_by_detection_conf)

    # define the transforms

    # efficientnet data preprocessing:
    # - train:
    #   1) random crop: aspect_ratio_range=(0.75, 1.33), area_range=(0.08, 1.0)
    #   2) bicubic resize to img_size
    #   3) random horizontal flip
    # - test:
    #   1) center crop
    #   2) bicubic resize to img_size

    @tf.function
    def train_transform(img: tf.Tensor) -> tf.Tensor:
        """Returns: tf.Tensor, shape [img_size, img_size, C], type float32"""
        img = tf.image.resize_with_pad(img, img_size, img_size,
                                       method=tf.image.ResizeMethod.BICUBIC)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.25)
        img = tf.image.random_contrast(img, lower=0.75, upper=1.25)
        img = tf.image.random_saturation(img, lower=0.75, upper=1.25)
        return img

    @tf.function
    def test_transform(img: tf.Tensor) -> tf.Tensor:
        """Returns: tf.Tensor, shape [img_size, img_size, C], type float32"""
        img = tf.image.resize_with_pad(img, img_size, img_size,
                                       method=tf.image.ResizeMethod.BICUBIC)
        return img

    dataloaders = {}
    for split, locs in split_to_locs.items():
        is_train = (split == 'train') and augment_train
        split_df = df[df['dataset_location'].isin(locs)]

        weights = None
        if label_weighted or weight_by_detection_conf:
            # weights sums to:
            # - if weight_by_detection_conf: (# images in split - conf delta)
            # - otherwise: (# images in split)
            weights = split_df['weights'].tolist()
            if not weight_by_detection_conf:
                assert np.isclose(sum(weights), len(split_df))

        cache: bool | str = (split in cache_splits)
        if split == 'train' and 'train' in cache_splits:
            unique_filename = str(uuid.uuid4())
            os.makedirs('/mnt/tempds/', exist_ok=True)
            cache = f'/mnt/tempds/{unique_filename}'

        ds = create_dataset(
            img_files=split_df['path'].tolist(),
            labels=split_df['label_index'].tolist(),
            sample_weights=weights,
            img_base_dir=cropped_images_dir,
            transform=train_transform if is_train else test_transform,
            target_transform=None,
            cache=cache)
        if is_train:
            ds = ds.shuffle(1000, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        dataloaders[split] = ds

    return dataloaders, label_names


def build_model(model_name: str, num_classes: int, img_size: int,
                pretrained: bool, finetune: bool) -> tf.keras.Model:
    """Creates a model with an EfficientNet base."""
    class_name = EFFICIENTNET_MODELS[model_name]['cls']
    dropout = EFFICIENTNET_MODELS[model_name]['dropout']

    model_class = tf.keras.applications.__dict__[class_name]
    weights = 'imagenet' if pretrained else None
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    base_model = model_class(
        input_tensor=inputs, weights=weights, include_top=False, pooling='avg')

    if finetune:
        # freeze the base model's weights, including BatchNorm statistics
        # https://www.tensorflow.org/guide/keras/transfer_learning#fine-tuning
        base_model.trainable = False

    # rebuild output
    x = tf.keras.layers.Dropout(dropout, name='top_dropout')(base_model.output)
    outputs = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=1. / 3., mode='fan_out', distribution='uniform'),
        name='logits')(x)
    model = tf.keras.Model(inputs, outputs, name='complete_model')
    model.base_model = base_model  # cache this so that we can turn off finetune
    return model


def main(dataset_dir: str,
         cropped_images_dir: str,
         multilabel: bool,
         model_name: str,
         pretrained: bool,
         finetune: int,
         label_weighted: bool,
         weight_by_detection_conf: bool | str,
         epochs: int,
         batch_size: int,
         lr: float,
         weight_decay: float,
         seed: Optional[int] = None,
         logdir: str = '',
         cache_splits: Sequence[str] = ()) -> None:
    """Main function."""
    # input validation
    assert os.path.exists(dataset_dir)
    assert os.path.exists(cropped_images_dir)
    if isinstance(weight_by_detection_conf, str):
        assert os.path.exists(weight_by_detection_conf)

    # set seed
    seed = np.random.randint(10_000) if seed is None else seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # create logdir and save params
    params = dict(locals())  # make a copy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # '20200722_110816'
    logdir = os.path.join(logdir, timestamp)
    os.makedirs(logdir, exist_ok=True)
    print('Created logdir:', logdir)
    with open(os.path.join(logdir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=1)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    img_size = EFFICIENTNET_MODELS[model_name]['img_size']

    # create dataloaders and log the index_to_label mapping
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
        augment_train=True,
        cache_splits=cache_splits)

    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()

    model = build_model(
        model_name, num_classes=len(label_names), img_size=img_size,
        pretrained=pretrained, finetune=finetune > 0)

    # define loss function and optimizer
    loss_fn: tf.keras.losses.Loss
    if multilabel:
        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    # using EfficientNet training defaults
    # - batch norm momentum: 0.99
    # - optimizer: RMSProp, decay 0.9 and momentum 0.9
    # - epochs: 350
    # - learning rate: 0.256, decays by 0.97 every 2.4 epochs
    # - weight decay: 1e-5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr, decay_steps=1, decay_rate=0.97, staircase=True)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=lr, rho=0.9, momentum=0.9)

    best_epoch_metrics: dict[str, float] = {}
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        optimizer.learning_rate = lr_schedule(epoch)
        tf.summary.scalar('lr', optimizer.learning_rate, epoch)

        if epoch > 0 and finetune == epoch:
            print('Turning off fine-tune!')
            model.base_model.trainable = True

        print('- train:')
        # TODO: change weighted to False if oversampling minority classes
        train_metrics, train_heaps, train_cm = run_epoch(
            model, loader=loaders['train'], weighted=label_weighted,
            loss_fn=loss_fn, weight_decay=weight_decay, optimizer=optimizer,
            finetune=finetune > epoch, return_extreme_images=True)
        train_metrics = prefix_all_keys(train_metrics, prefix='train/')
        log_run('train', epoch, writer, label_names,
                metrics=train_metrics, heaps=train_heaps, cm=train_cm)

        print('- val:')
        val_metrics, val_heaps, val_cm = run_epoch(
            model, loader=loaders['val'], weighted=label_weighted,
            loss_fn=loss_fn, return_extreme_images=True)
        val_metrics = prefix_all_keys(val_metrics, prefix='val/')
        log_run('val', epoch, writer, label_names,
                metrics=val_metrics, heaps=val_heaps, cm=val_cm)

        if val_metrics['val/acc_top1'] > best_epoch_metrics.get('val/acc_top1', 0):  # pylint: disable=line-too-long
            filename = os.path.join(logdir, f'ckpt_{epoch}.h5')
            print(f'New best model! Saving checkpoint to {filename}')
            model.save(filename)
            best_epoch_metrics.update(train_metrics)
            best_epoch_metrics.update(val_metrics)
            best_epoch_metrics['epoch'] = epoch

            print('- test:')
            test_metrics, test_heaps, test_cm = run_epoch(
                model, loader=loaders['test'], weighted=label_weighted,
                loss_fn=loss_fn, return_extreme_images=True)
            test_metrics = prefix_all_keys(test_metrics, prefix='test/')
            log_run('test', epoch, writer, label_names,
                    metrics=test_metrics, heaps=test_heaps, cm=test_cm)

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
    hp.hparams(hparams_dict)
    writer.close()


def log_run(split: str, epoch: int, writer: tf.summary.SummaryWriter,
            label_names: Sequence[str], metrics: MutableMapping[str, float],
            heaps: Mapping[str, Mapping[int, list[HeapItem]]], cm: np.ndarray
            ) -> None:
    """Logs the outputs (metrics, confusion matrix, tp/fp/fn images) from a
    single epoch run to Tensorboard.

    Args:
        metrics: dict, keys already prefixed with {split}/
    """
    per_class_recall = recall_from_confusion_matrix(cm, label_names)
    metrics.update(prefix_all_keys(per_class_recall, f'{split}/label_recall/'))

    # log metrics
    for metric, value in metrics.items():
        tf.summary.scalar(metric, value, epoch)

    # log confusion matrix
    cm_fig = plot_utils.plot_confusion_matrix(cm, classes=label_names,
                                              normalize=True)
    cm_fig_img = tf.convert_to_tensor(fig_to_img(cm_fig)[np.newaxis, ...])
    tf.summary.image(f'confusion_matrix/{split}', cm_fig_img, step=epoch)

    # log tp/fp/fn images
    for heap_type, heap_dict in heaps.items():
        log_images_with_confidence(heap_dict, label_names, epoch=epoch,
                                   tag=f'{split}/{heap_type}')
    writer.flush()


def log_images_with_confidence(
        heap_dict: Mapping[int, list[HeapItem]],
        label_names: Sequence[str],
        epoch: int,
        tag: str) -> None:
    """
    Args:
        heap_dict: dict, maps label_id to list of HeapItem, where each HeapItem
            data is a list [img, target, top3_conf, top3_preds, img_file],
            and img is a tf.Tensor of shape [H, W, 3]
        label_names: list of str, label names in order of label id
        epoch: int
        tag: str
    """
    for label_id, heap in heap_dict.items():
        label_name = label_names[label_id]

        sorted_heap = sorted(heap, reverse=True)  # sort largest to smallest
        imgs_list = [item.data for item in sorted_heap]
        fig, img_files = imgs_with_confidences(imgs_list, label_names)

        # tf.summary.image requires input of shape [N, H, W, C]
        fig_img = tf.convert_to_tensor(fig_to_img(fig)[np.newaxis, ...])
        tf.summary.image(f'{label_name}/{tag}', fig_img, step=epoch)
        tf.summary.text(f'{label_name}/{tag}_files', '\n\n'.join(img_files),
                        step=epoch)


def track_extreme_examples(tp_heaps: dict[int, list[HeapItem]],
                           fp_heaps: dict[int, list[HeapItem]],
                           fn_heaps: dict[int, list[HeapItem]],
                           inputs: tf.Tensor,
                           labels: tf.Tensor,
                           img_files: tf.Tensor,
                           logits: tf.Tensor) -> None:
    """Updates the 5 most extreme true-positive (tp), false-positive (fp), and
    false-negative (fn) examples with examples from this batch.

    Each HeapItem's data attribute is a tuple with:
    - img: np.ndarray, shape [H, W, 3], type uint8
    - label: int
    - top3_conf: list of float
    - top3_preds: list of float
    - img_file: str

    Args:
        *_heaps: dict, maps label_id (int) to heap of HeapItems
        inputs: tf.Tensor, shape [batch_size, H, W, 3], type float32
        labels: tf.Tensor, shape [batch_size]
        img_files: tf.Tensor, shape [batch_size], type tf.string
        logits: tf.Tensor, shape [batch_size, num_classes]
    """
    labels = labels.numpy().tolist()
    inputs = inputs.numpy().astype(np.uint8)
    img_files = img_files.numpy().astype(str).tolist()
    batch_probs = tf.nn.softmax(logits, axis=1)
    iterable = zip(labels, inputs, img_files, batch_probs)
    for label, img, img_file, confs in iterable:
        label_conf = confs[label].numpy().item()

        top3_conf, top3_preds = tf.math.top_k(confs, k=3, sorted=True)
        top3_conf = top3_conf.numpy().tolist()
        top3_preds = top3_preds.numpy().tolist()

        data = (img, label, top3_conf, top3_preds, img_file)
        if top3_preds[0] == label:  # true positive
            item = HeapItem(priority=label_conf - top3_conf[1], data=data)
            add_to_heap(tp_heaps[label], item, k=5)
        else:
            # false positive for top3_pred[0]
            # false negative for label
            item = HeapItem(priority=top3_conf[0] - label_conf, data=data)
            add_to_heap(fp_heaps[top3_preds[0]], item, k=5)
            add_to_heap(fn_heaps[label], item, k=5)


def run_epoch(model: tf.keras.Model,
              loader: tf.data.Dataset,
              weighted: bool,
              top: Sequence[int] = (1, 3),
              loss_fn: Optional[tf.keras.losses.Loss] = None,
              weight_decay: float = 0,
              finetune: bool = False,
              optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
              return_extreme_images: bool = False
              ) -> tuple[
                  dict[str, float],
                  dict[str, dict[int, list[HeapItem]]],
                  np.ndarray
              ]:
    """Runs for 1 epoch.

    Args:
        model: tf.keras.Model
        loader: tf.data.Dataset
        weighted: bool, whether to use sample weights in calculating loss and
            accuracy
        top: tuple of int, list of values of k for calculating top-K accuracy
        loss_fn: optional loss function, calculates the mean loss over a batch
        weight_decay: float, L2-regularization constant
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
    # if evaluating or finetuning, set dropout & BN layers to eval mode
    is_train = False
    train_dropout_and_bn = False

    if optimizer is not None:
        assert loss_fn is not None
        is_train = True

        if not finetune:
            train_dropout_and_bn = True
            reg_vars = [
                v for v in model.trainable_variables if 'kernel' in v.name]

    if loss_fn is not None:
        losses = tf.keras.metrics.Mean()
    accuracies_topk = {
        k: tf.keras.metrics.SparseTopKCategoricalAccuracy(k) for k in top
    }

    # for each label, track 5 most-confident and least-confident examples
    tp_heaps: dict[int, list[HeapItem]] = defaultdict(list)
    fp_heaps: dict[int, list[HeapItem]] = defaultdict(list)
    fn_heaps: dict[int, list[HeapItem]] = defaultdict(list)

    all_labels = []
    all_preds = []

    tqdm_loader = tqdm.tqdm(loader)
    for batch in tqdm_loader:
        if weighted:
            inputs, labels, img_files, weights = batch
        else:
            # even if batch contains sample weights, don't use them
            inputs, labels, img_files = batch[0:3]
            weights = None

        all_labels.append(labels.numpy())
        desc = []
        with tf.GradientTape(watch_accessed_variables=is_train) as tape:
            outputs = model(inputs, training=train_dropout_and_bn)
            if loss_fn is not None:
                loss = loss_fn(labels, outputs)
                if weights is not None:
                    loss *= weights
                # we do not track L2-regularization loss in the loss metric
                losses.update_state(loss, sample_weight=weights)
                desc.append(f'Loss {losses.result().numpy():.4f}')

            if optimizer is not None:
                loss = tf.math.reduce_mean(loss)
                if not finetune:  # only regularize layers before the final FC
                    loss += weight_decay * tf.add_n(
                        tf.nn.l2_loss(v) for v in reg_vars)

        all_preds.append(tf.math.argmax(outputs, axis=1).numpy())

        if optimizer is not None:
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        for k, acc in accuracies_topk.items():
            acc.update_state(labels, outputs, sample_weight=weights)
            desc.append(f'Acc@{k} {acc.result().numpy() * 100:.3f}')
        tqdm_loader.set_description(' '.join(desc))

        if return_extreme_images:
            track_extreme_examples(tp_heaps, fp_heaps, fn_heaps, inputs,
                                   labels, img_files, outputs)

    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true=np.concatenate(all_labels), y_pred=np.concatenate(all_preds))

    metrics = {}
    if loss_fn is not None:
        metrics['loss'] = losses.result().numpy().item()
    for k, acc in accuracies_topk.items():
        metrics[f'acc_top{k}'] = acc.result().numpy().item() * 100
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
        choices=list(EFFICIENTNET_MODELS.keys()),
        help='which EfficientNet model')
    parser.add_argument(
        '--pretrained', action='store_true',
        help='start with pretrained model')
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
        '--lr', type=float, default=None,
        help='initial learning rate, defaults to (0.016 * batch_size / 256)')
    parser.add_argument(
        '--weight-decay', type=float, default=1e-5,
        help='weight decay')
    parser.add_argument(
        '--seed', type=int,
        help='random seed')
    parser.add_argument(
        '--logdir', default='.',
        help='directory where TensorBoard logs and a params file are saved')
    parser.add_argument(
        '--cache', nargs='*', choices=['train', 'val', 'test'], default=(),
        help='which splits of the dataset to cache')
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
         seed=args.seed,
         logdir=args.logdir,
         cache_splits=args.cache)

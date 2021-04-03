"""
Utility functions useful for training a classifier.

This script should NOT depend on any other file within this repo. It should
especially be agnostic to PyTorch vs. TensorFlow.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import heapq
import io
import json
from typing import Any, Optional

import matplotlib.figure
import numpy as np
import pandas as pd
import PIL.Image
import scipy.interpolate


@dataclasses.dataclass(order=True)
class HeapItem:
    """A wrapper over non-comparable data with a comparable priority value."""
    priority: Any
    data: Any = dataclasses.field(compare=False, repr=False)


def add_to_heap(h: list[Any], item: HeapItem, k: Optional[int] = None) -> None:
    """Tracks the max k elements using a heap.

    We use a min-heap for this task. When a new element comes in, we compare it
    to the smallest node in the heap, h[0]. If the new value is greater than
    h[0], we pop h[0] and add the new element in.

    Args:
        h: list, either empty [] or already heapified
        item: HeapItem
        k: int, desired capacity of the heap, or None for no limit
    """
    if k is None or len(h) < k:
        heapq.heappush(h, item)
    else:
        heapq.heappushpop(h, item)


def prefix_all_keys(d: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Returns a new dict where the keys are prefixed by <prefix>."""
    return {f'{prefix}{k}': v for k, v in d.items()}


def fig_to_img(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Converts a matplotlib figure to an image represented by a numpy array.

    TODO: potential speedup by avoiding PNG compression and PIL dependency
    See https://stackoverflow.com/a/61443397

    Returns: np.ndarray, type uint8, shape [H, W, 3]
    """
    with io.BytesIO() as b:
        fig.savefig(b, transparent=False, bbox_inches='tight', pad_inches=0,
                    format='png')
        b.seek(0)
        fig_img = np.asarray(PIL.Image.open(b).convert('RGB'))
    assert fig_img.dtype == np.uint8
    return fig_img


def imgs_with_confidences(imgs_list: list[tuple[Any, ...]],
                          label_names: Sequence[str],
                          ) -> tuple[matplotlib.figure.Figure, list[str]]:
    """
    Args:
        imgs_list: list of tuple, each tuple consists of:
            img: array_like, shape [H, W, C], type either float [0, 1] or uint8
            label_id: int, label index
            topk_conf: list of float, confidence scores for topk predictions
            topk_preds: list of int, label indices for topk predictions
            img_file: str, path to image file
        label_names: list of str, label names in order of label id

    Returns:
        fig: matplotlib.figure.Figure
        img_files: list of str
    """
    imgs, img_files, tags, titles = [], [], [], []
    for img, label_id, topk_conf, topk_preds, img_file in imgs_list:
        imgs.append(img)
        img_files.append(img_file)
        tags.append(label_names[label_id])

        lines = []
        for pred, conf in zip(topk_preds, topk_conf):
            pred_name = label_names[pred]
            lines.append(f'{pred_name}: {conf:.03f}')
        titles.append('\n'.join(lines))

    fig = plot_img_grid(imgs=imgs, row_h=3, col_w=2.5, tags=tags, titles=titles)
    return fig, img_files


def plot_img_grid(imgs: Sequence[Any], row_h: float, col_w: float,
                  ncols: Optional[int] = None,
                  tags: Optional[Sequence[str]] = None,
                  titles: Optional[Sequence[str]] = None
                  ) -> matplotlib.figure.Figure:
    """Plots a grid of images.

    Args:
        imgs: list of images, each image is either an array or a PIL Image,
            see matplotlib.axes.Axes.imshow() documentation for supported shapes
        row_h: float, row height in inches
        col_w: float, col width in inches
        ncols: optional int, number of columns, defaults to len(imgs)
        tags: optional list of str, tags are displayed in upper-left corner of
            each image on a white background
        titles: optional list of str, text displayed above each image

    Returns: matplotlib.figure.Figure
    """
    # input validation
    num_images = len(imgs)
    if tags is not None:
        assert len(tags) == len(imgs)
    if titles is not None:
        assert len(titles) == len(imgs)

    if ncols is None:
        ncols = num_images

    nrows = int(np.ceil(len(imgs) / ncols))
    fig = matplotlib.figure.Figure(figsize=(ncols * col_w, nrows * row_h),
                                   tight_layout=True)
    axs = fig.subplots(nrows, ncols, squeeze=False)

    # plot the images
    for i in range(num_images):
        r, c = i // ncols, i % ncols
        ax = axs[r, c]
        ax.imshow(imgs[i])
        if tags is not None:
            ax.text(-0.2, -0.2, tags[i], ha='left', va='top',
                    bbox=dict(lw=0, facecolor='white'))
        if titles is not None:
            ax.set_title(titles[i])

    # adjust the figure
    for r in range(nrows):
        for c in range(ncols):
            axs[r, c].set_axis_off()
            axs[r, c].set_aspect('equal')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def load_splits(splits_json_path: str) -> dict[str, set[tuple[str, str]]]:
    """Loads location splits from JSON file and assert that there are no
    overlaps between splits.

    Args:
        splits_json_path: str, path to JSON file

    Returns: dict, maps split to set of (dataset, location) tuples
    """
    with open(splits_json_path, 'r') as f:
        split_to_locs_js = json.load(f)
    split_to_locs = {
        split: set((loc[0], loc[1]) for loc in locs)
        for split, locs in split_to_locs_js.items()
    }
    assert split_to_locs['train'].isdisjoint(split_to_locs['val'])
    assert split_to_locs['train'].isdisjoint(split_to_locs['test'])
    assert split_to_locs['val'].isdisjoint(split_to_locs['test'])
    return split_to_locs


def load_dataset_csv(dataset_csv_path: str,
                     label_index_json_path: str,
                     splits_json_path: str,
                     multilabel: bool,
                     weight_by_detection_conf: bool | str,
                     label_weighted: bool
                     ) -> tuple[pd.DataFrame,
                                list[str],
                                dict[str, set[tuple[str, str]]]
                               ]:
    """
    Args:
        dataset_csv_path: str, path to CSV file with columns
            ['dataset', 'location', 'label', 'confidence'], where label is a
            comma-delimited list of labels
        label_index_json_path: str, path to label index JSON file
        splits_json_path: str, path to splits JSON file
        multilabel: bool, whether a single example can have multiple labels
        weight_by_detection_conf: bool or str
            - if True: assumes classification CSV's 'confidence' column
                represents calibrated scores
            - if str: path the .npz file containing x/y values for isotonic
                regression calibration function
        label_weighted: bool, whether to give each label equal weight

    Returns:
        df: pd.DataFrame, with columns
            dataset_location: tuples of (dataset, location)
            label: str if not multilabel, list of str if multilabel
            label_index: int if not multilabel, list of int if multilabel
            weights: float, weight for each example
                column exists if and only if label_weighted=True or
                weight_by_detection_conf is not False
        label_names: list of str, label names in order of label id
        split_to_locs: dict, maps split to set of (dataset, location) tuples
    """
    # read in dataset CSV and create merged (dataset, location) col
    df = pd.read_csv(dataset_csv_path, index_col=False, float_precision='high')
    df['dataset_location'] = list(zip(df['dataset'], df['location']))

    with open(label_index_json_path, 'r') as f:
        idx_to_label = json.load(f)
    label_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}

    # map label to label_index
    if multilabel:
        df['label'] = df['label'].map(lambda x: x.split(','))
        df['label_index'] = df['label'].map(
            lambda labellist: tuple(sorted(label_to_idx[y] for y in labellist)))
    else:
        assert not any(df['label'].str.contains(','))
        df['label_index'] = df['label'].map(label_to_idx.__getitem__)

    # load the splits
    split_to_locs = load_splits(splits_json_path)

    if weight_by_detection_conf:
        df['weights'] = 1.0

        # only weight the training set by detection confidence
        # TODO: consider weighting val and test set as well
        train_mask = df['dataset_location'].isin(split_to_locs['train'])
        df.loc[train_mask, 'weights'] = df.loc[train_mask, 'confidence']

        if isinstance(weight_by_detection_conf, str):
            # isotonic regression calibration of MegaDetector confidence
            with np.load(weight_by_detection_conf) as npz:
                calib = scipy.interpolate.interp1d(
                    x=npz['x'], y=npz['y'], kind='linear')
            df.loc[train_mask, 'weights'] = calib(df.loc[train_mask, 'weights'])

    if label_weighted:
        if multilabel:
            raise NotImplementedError  # TODO

        if 'weights' not in df.columns:
            df['weights'] = 1.0

        # treat each split separately
        # new_weight[i] = confidence[i] * (n / c) / total_confidence(i's label)
        # - n = # examples in split (weighted by confidence); c = # labels
        # - weight allocated to each label is n/c
        # - within each label, weigh each example proportional to confidence
        # - new weights sum to n
        c = len(label_names)
        for split, locs in split_to_locs.items():
            split_mask = df['dataset_location'].isin(locs)
            n = df.loc[split_mask, 'weights'].sum()
            per_label_conf = df[split_mask].groupby('label')['weights'].sum()
            assert len(per_label_conf) == c, (
                f'{split} split only has {len(per_label_conf)}/{c} labels')
            scaling = (n / c) / per_label_conf[df.loc[split_mask, 'label']]
            df.loc[split_mask, 'weights'] *= scaling.to_numpy()
            w_sum = df.loc[split_mask, 'weights'].sum()
            assert np.isclose(w_sum, n), (
                f'Expected {split} weights to sum to {n}, got {w_sum} instead')

        # error checking
        assert (df['weights'] > 0).all()

    return df, label_names, split_to_locs


def recall_from_confusion_matrix(
        confusion_matrix: np.ndarray,
        label_names: Sequence[str],
        ) -> dict[str, float]:
    """
    Args:
        confusion_matrix: np.ndarray, shape [n_classes, n_classes], type int
            C[i, j] = # of samples with true label i, predicted as label j
        label_names: list of str, label names in order by label id

    Returns: dict, label_name => recall
    """
    result = {
        label_name: confusion_matrix[i, i] / (confusion_matrix[i].sum() + 1e-8)
        for i, label_name in enumerate(label_names)
    }
    return result

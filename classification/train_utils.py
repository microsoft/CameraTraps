"""Track the max-k elements using a heap."""
import dataclasses
import heapq
import io
import json
from typing import (Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple,
                    Union)

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


def add_to_heap(h: List[Any], item: HeapItem, k: Optional[int] = None) -> None:
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


def prefix_all_keys(d: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
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


def imgs_with_confidences(imgs_list: List[Tuple[Any, ...]],
                          label_names: Sequence[str],
                          ) -> Tuple[np.ndarray, List[str]]:
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
        fig_img: np.ndarray, type uint8, shape [H, W, C]
        img_files: list of str
    """
    num_images = len(imgs_list)
    fig = matplotlib.figure.Figure(figsize=(num_images * 2.5, 3),
                                   tight_layout=True)
    axs = fig.subplots(1, num_images, squeeze=False)[0, :]
    img_files = []
    for ax, item in zip(axs, imgs_list):
        img, label_id, topk_conf, topk_preds, img_file = item
        img_files.append(img_file)

        ax.imshow(img)
        ax.axis('off')
        ax.text(-0.2, -0.2, label_names[label_id], ha='left', va='top',
                bbox=dict(lw=0, facecolor='white'))

        lines = []
        for pred, conf in zip(topk_preds, topk_conf):
            pred_label_name = label_names[pred]
            lines.append(f'{pred_label_name}: {conf:.03f}')
        ax.set_title('\n'.join(lines))

    return fig, img_files


def load_dataset_csv(dataset_csv_path: str,
                     label_index_json_path: str,
                     splits_json_path: str,
                     multilabel: bool,
                     weight_by_detection_conf: Union[bool, str],
                     label_weighted: bool
                     ) -> Tuple[pd.DataFrame,
                                List[str],
                                Dict[str, Set[Tuple[str, str]]]
                               ]:
    """
    Args:
        classification_dataset_csv_path: str, path to CSV file with columns
            ['dataset', 'location', 'label', 'confidence'], where label is a
            comma-delimited list of labels
        label_index_json_path: str, TODO
        splits_json_path: str, path to JSON file
        multilabel: bool, TODO
        weight_by_detection_conf: bool or str
            - if True: assumes classification CSV's 'confidence' column
                represents calibrated scores
            - if str: path the .npz file containing x/y values for isotonic
                regression calibration function
        label_weighted: bool, TODO

    Returns:
        df: pd.DataFrame, with columns
            dataset_location: tuples of (dataset, location)
            label: str if not multilabel, list of str if multilabel
            label_index: int if not multilabel, list of int if multilabel
            weights: float, column sums to C where C = number of labels,
                column exists only if label_weighted=True
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

    # load the splits and assert that there are no overlaps in locs
    with open(splits_json_path, 'r') as f:
        split_to_locs_js = json.load(f)
    split_to_locs = {
        split: set((loc[0], loc[1]) for loc in locs)
        for split, locs in split_to_locs_js.items()
    }
    assert split_to_locs['train'].isdisjoint(split_to_locs['val'])
    assert split_to_locs['train'].isdisjoint(split_to_locs['test'])
    assert split_to_locs['val'].isdisjoint(split_to_locs['test'])

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
        # sample_weight[i] = (N / C) / count(i's label)
        # - N = # of examples in split (weighted by confidence); C = # of labels
        # - weight allocated to each label is N/C
        # - within each label, weigh each example as 1/# of examples in label
        # - sample_weights sums to N
        c = len(label_names)
        for locs in split_to_locs.values():
            split_mask = df['dataset_location'].isin(locs)
            n = df.loc[split_mask, 'weights'].sum()
            label_sizes = df[split_mask].groupby('label').size()
            sample_weights = (n / c) / label_sizes[df.loc[split_mask, 'label']]
            assert np.isclose(sample_weights.sum(), n)
            df.loc[split_mask, 'weights'] = sample_weights.to_numpy()

        # error checking
        assert (df['weights'] > 0).all()

    return df, label_names, split_to_locs


def recall_from_confusion_matrix(
        confusion_matrix: np.ndarray,
        label_names: Sequence[str],
        ) -> Dict[str, float]:
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

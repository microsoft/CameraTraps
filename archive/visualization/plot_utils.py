"""Functions for plotting.

NOTE: Do NOT import matplotlib.pyplot here to avoid the interactive backend.
Thus, the matplotlib.figure.Figure objects returned by the functions here do not
need to be "closed" with `plt.close(fig)`.
"""
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.figure  # this also imports mpl.{cm, axes, colors}


def plot_confusion_matrix(
        matrix: np.ndarray,
        classes: Sequence[str],
        normalize: bool = False,
        title: str = 'Confusion matrix',
        cmap: Union[str, matplotlib.colors.Colormap] = matplotlib.cm.Blues,
        vmax: Optional[float] = None,
        use_colorbar: bool = True,
        y_label: bool = True,
        fmt: str = '{:.0f}'
        ) -> matplotlib.figure.Figure:
    """Plot a confusion matrix. By default, assumes values in the given matrix
    are percentages. If the matrix contains counts, normalization can be applied
    by setting `normalize=True`.

    Args:
        matrix: np.ndarray, shape [num_classes, num_classes], confusion matrix
            where rows are ground-truth classes and cols are predicted classes.
        classes: list of str, class names for each row/column
        normalize: bool, whether to perform row-wise normalization to sum 1
        title: str, figure title
        cmap: colormap, default: matplotlib.cm.Blues
        vmax: float, value corresponding s to the largest value of the colormap.
            If None, the maximum value in *matrix* will be used. Default: None
        use_colorbar: bool, whether to show colorbar
        y_label: bool, whether to show class names on the y-axis
        fmt: str, format string

    Returns: matplotlib.figure.Figure, a reference to the figure
    """
    num_classes = matrix.shape[0]
    assert matrix.shape[1] == num_classes
    assert len(classes) == num_classes

    if normalize:
        matrix = matrix.astype(np.float64) / (
            matrix.sum(axis=1, keepdims=True) + 1e-7)

    fig_h = 3 + 0.3 * num_classes
    fig_w = fig_h
    if use_colorbar:
        fig_w += 0.5

    fig = matplotlib.figure.Figure(figsize=(fig_w, fig_h), tight_layout=True)
    ax = fig.subplots(1, 1)
    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap, vmax=vmax)
    ax.set_title(title)

    if use_colorbar:
        cbar = fig.colorbar(im, fraction=0.046, pad=0.04,
                            ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=90)
    ax.set_xlabel('Predicted class')

    if y_label:
        ax.set_yticklabels(classes)
        ax.set_ylabel('Ground-truth class')

    for i, j in np.ndindex(matrix.shape):
        ax.text(j, i, fmt.format(matrix[i, j] * 100),
                horizontalalignment='center',
                verticalalignment='center',
                color='white' if matrix[i, j] > 0.5 else 'black')

    return fig


def plot_precision_recall_curve(
        precisions: Sequence[float], recalls: Sequence[float],
        title: str = 'Precision/recall curve',
        xlim=(0.0,1.05),ylim=(0.0,1.05)
        ) -> matplotlib.figure.Figure:
    """
    Plots the precision recall curve given lists of (ordered) precision
    and recall values.

    Args:
        precisions: list of float, precision for corresponding recall values,
            should have same length as *recalls*.
        recalls: list of float, recall for corresponding precision values,
            should have same length as *precisions*.
        title: str, plot title

    Returns: matplotlib.figure.Figure, reference to the figure
    """
    assert len(precisions) == len(recalls)

    fig = matplotlib.figure.Figure(tight_layout=True)
    ax = fig.subplots(1, 1)
    ax.step(recalls, precisions, color='b', alpha=0.2, where='post')
    ax.fill_between(recalls, precisions, alpha=0.2, color='b', step='post')

    try:
        ax.set(x_label='Recall', y_label='Precision', title=title)
        ax.set(x_lim=xlim, y_lim=ylim)    
    # 
    except Exception:
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.set_xlim(xlim[0],xlim[1])
        ax.set_ylim(ylim[0],ylim[1])
        
    return fig


def plot_stacked_bar_chart(data: np.ndarray,
                           series_labels: Sequence[str],
                           col_labels: Optional[Sequence[str]] = None,
                           x_label: Optional[str] = None,
                           y_label: Optional[str] = None,
                           log_scale: bool = False
                           ) -> matplotlib.figure.Figure:
    """
    For plotting e.g. species distribution across locations.
    Reference: https://stackoverflow.com/q/44309507

    Args:
        data: 2-D np.ndarray or nested list, rows (series) are species, columns
            are locations
        series_labels: list of str, e.g., species names
        col_labels: list of str, e.g., location names
        x_label: str
        y_label: str
        log_scale: bool, whether to plot y-axis in log-scale

    Returns: matplotlib.figure.Figure, reference to figure
    """
    data = np.asarray(data)
    num_series, num_columns = data.shape
    ind = np.arange(num_columns)

    fig = matplotlib.figure.Figure(tight_layout=True)
    ax = fig.subplots(1, 1)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, num_series))

    # stacked bar charts are made with each segment starting from a y position
    cumulative_size = np.zeros(num_columns)
    for i, row_data in enumerate(data):
        ax.bar(ind, row_data, bottom=cumulative_size, label=series_labels[i],
               color=colors[i])
        cumulative_size += row_data

    if col_labels and len(col_labels) < 25:
        ax.set_xticks(ind)
        ax.set_xticklabels(col_labels, rotation=90)
    elif col_labels:
        ax.set_xticks(list(range(0, len(col_labels), 20)))
        ax.set_xticklabels(col_labels, rotation=90)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if log_scale:
        ax.set_yscale('log')

    # To fit the legend in, shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(0.99, 0.5), frameon=False)

    return fig


def calibration_ece(true_scores: Sequence[int], pred_scores: Sequence[float],
                    num_bins: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Expected calibration error (ECE) as defined in equation (3) of
        Guo et al. "On Calibration of Modern Neural Networks." (2017).

    Implementation modified from sklearn.calibration.calibration_curve()
    in order to implement ECE calculation. See
        https://github.com/scikit-learn/scikit-learn/issues/18268

    Args:
        pred_scores: list of float, length N, pred_scores[i] is the predicted
            confidence that example i is positive
        true_scores: list of int, length N, binary-valued (0 = neg, 1 = pos)
        num_bins: int, number of bins to use (`M` in eq. (3) of Guo 2017)

    Returns:
        accs: np.ndarray, shape [M], type float64, accuracy in each bin,
            M <= num_bins because bins with no samples are not returned
        confs: np.ndarray, shape [M], type float64, mean model confidence in
            each bin
        ece: float, expected calibration error
    """
    assert len(true_scores) == len(pred_scores)

    bins = np.linspace(0., 1. + 1e-8, num=num_bins + 1)
    binids = np.digitize(pred_scores, bins) - 1

    bin_sums = np.bincount(binids, weights=pred_scores, minlength=len(bins))
    bin_true = np.bincount(binids, weights=true_scores, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    accs = bin_true[nonzero] / bin_total[nonzero]
    confs = bin_sums[nonzero] / bin_total[nonzero]

    weights = bin_total[nonzero] / len(true_scores)
    ece = np.abs(accs - confs) @ weights
    return accs, confs, ece


def plot_calibration_curve(true_scores: Sequence[int],
                           pred_scores: Sequence[float],
                           num_bins: int,
                           name: str = 'calibration',
                           plot_perf: bool = True,
                           plot_hist: bool = True,
                           ax: Optional[matplotlib.axes.Axes] = None,
                           **fig_kwargs: Any
                           ) -> matplotlib.figure.Figure:
    """Plot a calibration curve.

    Consider rewriting / removing this function if
        https://github.com/scikit-learn/scikit-learn/pull/17443
    is merged into an actual scikit-learn release.

    Args:
        see calibration_ece() for args
        name: str, label in legend for the calibration curve
        plot_perf: bool, whether to plot y=x line indicating perfect calibration
        plot_hist: bool, whether to plot histogram of counts
        ax: optional matplotlib Axes, if given then no legend is drawn, and
            fig_kwargs are ignored
        fig_kwargs: only used if ax is None

    Returns: matplotlib Figure
    """
    accs, confs, ece = calibration_ece(true_scores, pred_scores, num_bins)

    created_fig = False
    if ax is None:
        created_fig = True
        fig = matplotlib.figure.Figure(**fig_kwargs)
        ax = fig.subplots(1, 1)
    ax.plot(confs, accs, 's-', label=name)  # 's-': squares on line
    ax.set(xlabel='Model confidence', ylabel='Actual accuracy',
           title=f'Calibration plot (ECE: {ece:.02g})')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    if plot_perf:
        ax.plot([0, 1], [0, 1], color='black', label='perfect calibration')
    ax.grid(True)

    if plot_hist:
        ax1 = ax.twinx()
        bins = np.linspace(0., 1. + 1e-8, num=num_bins + 1)
        counts = ax1.hist(pred_scores, alpha=0.5, label='histogram of examples',
                          bins=bins, color='tab:red')[0]
        max_count = np.max(counts)
        ax1.set_ylim([-0.05 * max_count, 1.05 * max_count])
        ax1.set_ylabel('Count')

    if created_fig:
        fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))

    return ax.figure

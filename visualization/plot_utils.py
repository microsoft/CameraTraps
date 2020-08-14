"""Functions for plotting."""
from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt


def plot_confusion_matrix(
        matrix: np.ndarray,
        classes: Sequence[str],
        normalize: bool = False,
        title: str = 'Confusion matrix',
        cmap: Union[str, matplotlib.colors.Colormap] = plt.cm.Blues,
        vmax: Optional[float] = None,
        use_colorbar: bool = True,
        y_label: bool = True
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
        cmap: pyplot colormap, default: matplotlib.pyplot.cm.Blues
        vmax: float, value corresponding s to the largest value of the colormap.
            If None, the maximum value in *matrix* will be used. Default: None
        use_colorbar: bool, whether to show colorbar
        y_label: bool, whether to show class names on the y-axis

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

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), tight_layout=True)
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
        ax.text(j, i, '{:.0f}'.format(matrix[i, j] * 100),
                horizontalalignment='center',
                verticalalignment='center',
                color='white' if matrix[i, j] > 0.5 else 'black')

    return fig


def plot_precision_recall_curve(
        precisions: Sequence[float], recalls: Sequence[float],
        title: str = 'Precision/Recall curve'
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

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.step(recalls, precisions, color='b', alpha=0.2, where='post')
    ax.fill_between(recalls, precisions, alpha=0.2, color='b', step='post')

    ax.set(x_label='Recall', y_label='Precision', title=title)
    ax.set(x_lim=(0.0, 1.05), y_lim=(0.0, 1.05))
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

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_series))

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

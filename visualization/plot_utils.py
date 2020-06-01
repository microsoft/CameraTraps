import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_confusion_matrix(matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          vmax=None,
                          use_colorbar=True,
                          y_label=True):
    """
    This function plots a confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        matrix: confusion matrix as a numpy 2D matrix. Rows are ground-truth classes
            and columns the predicted classes. Number of rows and columns have to match
        classes: list of strings, which contain the corresponding class names for each row/column
        normalize: boolean indicating whether to perform row-wise normalization to sum 1
        title: string which will be used as title
        cmap: pyplot colormap, default: matplotlib.pyplot.cm.Blues
        vmax: float, specifies the value that corresponds to the largest value of the colormap.
            If None, the maximum value in *matrix* will be used. Default: None
        use_colorbar: boolean indicating if a colorbar should be plotted
        y_label: boolean indicating whether class names should be plotted on the y-axis as well

    Returns a reference to the figure
    """

    assert matrix.shape[0] == matrix.shape[1]
    fig = plt.figure(figsize=[3 + 0.5 * len(classes)] * 2)

    if normalize:
        matrix = matrix.astype(np.double) / (matrix.sum(axis=1, keepdims=True) + 1e-7)

    plt.imshow(matrix, interpolation='nearest', cmap=cmap, vmax=vmax)
    plt.title(title)  # ,fontsize=22)

    if use_colorbar:
        plt.colorbar(fraction=0.046, pad=0.04,
                     ticks=[0.0, 0.25, 0.5, 0.75, 1.0]).set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)

    if y_label:
        plt.yticks(tick_marks, classes)
    else:
        plt.yticks(tick_marks, ['' for cn in classes])

    for i, j in np.ndindex(matrix.shape):
        plt.text(j, i, '{:.0f}%'.format(matrix[i, j] * 100),
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='white' if matrix[i, j] > 0.5 else 'black',
                 fontsize='x-small')

    if y_label:
        plt.ylabel('Ground-truth class')

    plt.xlabel('Predicted class')
    # plt.grid(False)
    plt.tight_layout()

    return fig


def plot_precision_recall_curve(precisions, recalls, title='Precision/Recall curve'):
    """
    Plots the precision recall curve given lists of (ordered) precision
    and recall values
    Args:
        precisions: list of floats, the precision for the corresponding recall values.
            Should have same length as *recalls*.
        recalls: list of floats, the recall values for corresponding precision values.
            Should have same length as *precisions*.
        title: string that will be as as plot title

    Returns a reference to the figure
    """

    step_kwargs = ({'step': 'post'})
    fig = plt.figure()
    plt.title(title)
    plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
    plt.fill_between(recalls, precisions, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])

    return fig


def plot_stacked_bar_chart(data, series_labels, col_labels=None, x_label=None, y_label=None, log_scale=False):
    """
    For plotting e.g. species distribution across locations.
    Reference: https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
    Args:
        data: a 2-dimensional numpy array or nested list containing data for each series (species)
              in rows (1st dimension) across locations (columns, 2nd dimension)

    Returns:
        the plot that can then be saved as a png.
    """

    fig = plt.figure()
    ax = plt.subplot(111)

    data = np.array(data)
    num_series, num_columns = data.shape
    ind = list(range(num_columns))

    colors = cm.rainbow(np.linspace(0, 1, num_series))

    cumulative_size = np.zeros(num_columns)  # stacked bar charts are made with each segment starting from a y position

    for i, row_data in enumerate(data):
        ax.bar(ind, row_data, bottom=cumulative_size, label=series_labels[i], color=colors[i])
        cumulative_size += row_data

    if col_labels and len(col_labels) < 25:
        ax.set_xticks(ind)
        ax.set_xticklabels(col_labels, rotation=90)
    elif col_labels:
        ax.set_xticks(list(range(0, len(col_labels), 20)))
        ax.set_xticklabels(col_labels, rotation=90)

    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if log_scale:
        ax.set_yscale('log')

    # To fit the legend in, shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(0.99, 0.5), frameon=False)

    return fig

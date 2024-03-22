from sklearn.metrics import confusion_matrix

def acc(preds, labels):
    """
    Calculate the accuracy metrics based on predictions and true labels.

    This function computes the confusion matrix and derives three types of accuracies:
    class-wise accuracy (cls_acc), micro accuracy (mic_acc), and macro accuracy (mac_acc).

    Args:
        preds (array-like): Predicted labels.
        labels (array-like): True labels.

    Returns:
        tuple: A tuple containing:
            - cls_acc (ndarray): Class-wise accuracy.
            - mac_acc (float): Macro accuracy (average of class-wise accuracies).
            - mic_acc (float): Micro accuracy (overall accuracy).
    """
    # Compute the confusion matrix from true labels and predictions
    matrix = confusion_matrix(labels, preds)

    # Calculate class-wise accuracy (accuracy for each class)
    cls_acc = matrix.diagonal() / matrix.sum(axis=1)

    # Calculate micro accuracy (overall accuracy)
    mic_acc = matrix.diagonal().sum() / matrix.sum()

    # Calculate macro accuracy (mean of class-wise accuracies)
    mac_acc = cls_acc.mean()

    return cls_acc, mac_acc, mic_acc

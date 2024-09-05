"""Functions to plot the outputs of the models."""
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from sklearn.metrics import confusion_matrix
from pathlib import Path


def plot_confusion_matrix(y_real: ArrayLike, y_pred: ArrayLike,
                          train_or_test: str) -> None:
    """Plot the confusion matrix of the model.

    Args:
        y_real (ArrayLike): Real target values.
        y_pred (ArrayLike): Predicted target values.
        train_or_test (str): Whether the confusion matrix is for the train or
            test

    Returns:
        None
    """
    fig_path = Path("./reports/figures")
    cm = confusion_matrix(y_real, y_pred)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Include the number of samples in each cell
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                     color='red')
    plt.savefig(fig_path / f"{train_or_test}_confusion_matrix.png", 
                dpi=300, bbox_inches='tight', pad_inches = 0.1)
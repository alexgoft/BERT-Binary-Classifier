from sklearn.metrics import classification_report, confusion_matrix
from plot_utils import plot_confusion_matrix
import torch.nn.functional as F
import numpy as np


def accumulate_elements(arr):
    # Detach the batch_out from the graph and move it to the CPU.
    # If the batch_out is a scalar, then convert it to a list with one element.
    arr = arr.detach().cpu().numpy().squeeze()
    if len(arr.shape) == 0:
        arr = [arr]
    return arr


def evaluate_on_dataset(model, dr, return_outputs=False):
    """Evaluate the model on the input data"""
    print('[INFO] Evaluating...')
    model.eval()
    outputs_acc = []
    labels_acc = []
    total_loss = 0.0
    for batch_idx, batch in enumerate(dr):
        loss, labels, outputs = model(batch)
        total_loss += loss.item()

        # Only relevant for the test mode. Reduce some memory usage.
        if return_outputs:
            outputs_acc.extend(accumulate_elements(outputs))
            labels_acc.extend(accumulate_elements(labels))

    return round(total_loss / len(dr), 5), outputs_acc, labels_acc


def test(config, test_dr, model):
    """Test the model on the test set. Print classification report and plot confusion matrix."""
    model.load_model(model_path=config.test.model_path)

    # Evaluate the model on the test set.
    test_loss, outputs, labels = evaluate_on_dataset(model=model, dr=test_dr, return_outputs=True)
    print(f'[INFO] Test loss: {test_loss}')

    # Calculate classification metrics.
    # y_pred = [1 if output > config.test.threshold else 0 for output in outputs]
    y_pred = [np.argmax(output) for output in outputs]
    y_true = [np.argmax(label) for label in labels]

    print('[INFO] Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))

    print('[INFO] Plotting confusion matrix...')
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm=cm, y_true=y_true, y_pred=y_pred)

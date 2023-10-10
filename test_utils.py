import os
import numpy as np

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from plot_utils import plot_confusion_matrix, plot_roc_curve


def calculate_classification_report(y_pred, y_true, output_dir):
    """Calculate the classification report and save it."""
    cfr = classification_report(y_true, y_pred, digits=4)
    print('[INFO] Classification Report:')
    print(cfr)
    print('[INFO] Saving Classification Report...')
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write(str(cfr))


def calculate_confusion_matrix(y_pred, y_true, output_dir):
    """Calculate the confusion matrix and plot it."""
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm=cm, y_true=y_true, y_pred=y_pred, output_dir=output_dir)
    print('[INFO] Saving Confusion Matrix...')


def calculate_auc(y_true, y_pred, output_dir):
    """Calculate the area under the ROC curve and plot the ROC curve."""
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print(f'[INFO] AUC: {roc_auc: .4f}')
    print('[INFO] Saving ROC Curve...')
    plot_roc_curve(fpr, tpr, roc_auc, output_dir)


def test(config, test_dr, model):
    """Test the model on the test set. Print classification report and plot confusion matrix."""
    model.load_model(model_path=config.test.model_path)

    model_name = config.test.model_path.rpartition('/')[-1].split('.pt')[0].replace('.', '_')
    model_output_dir = config.test.model_path.rpartition('/')[0]
    metrics_output_dir = f'{model_output_dir}/{model_name}_metrics'
    os.makedirs(metrics_output_dir, exist_ok=True)

    # Evaluate the model on the test set.
    test_loss, outputs, labels = evaluate_on_dataset(model=model, dr=test_dr)
    print(f'[INFO] Test loss: {test_loss}')

    # Calculate classification metrics.
    y_pred = [np.argmax(output) for output in outputs]
    y_true = [np.argmax(label) for label in labels]

    calculate_classification_report(y_pred, y_true, output_dir=metrics_output_dir)
    calculate_confusion_matrix(y_pred, y_true, output_dir=metrics_output_dir)

    # y_pred is the probability of the positive class.
    calculate_auc(y_true, np.array(outputs)[:, 1], output_dir=metrics_output_dir)
    print('[INFO] Done.')


# TODO: Move this to a utils file? Is it needed?
def accumulate_elements(arr):
    # Detach the batch_out from the graph and move it to the CPU.
    # If the batch_out is a scalar, then convert it to a list with one element.
    arr = arr.detach().cpu().numpy().squeeze()
    if len(arr.shape) == 0:
        arr = [arr]
    return arr


def evaluate_on_dataset(model, dr):
    """Evaluate the model on the input data"""
    print('[INFO] Evaluating...')
    model.eval()

    total_loss = 0.0

    # Accumulate the outputs and labels for the whole dataset.
    outputs_acc = []
    labels_acc = []
    for batch_idx, batch in enumerate(dr):
        loss, labels, outputs = model(batch)
        total_loss += loss.item()

        # Only relevant for the test mode. Reduce some memory usage.
        outputs_acc.extend(accumulate_elements(outputs))
        labels_acc.extend(accumulate_elements(labels))

    return round(total_loss / len(dr), 5), outputs_acc, labels_acc

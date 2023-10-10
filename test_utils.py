from sklearn.metrics import classification_report, confusion_matrix

from plot_utils import plot_confusion_matrix


def evaluate_on_dataset(model, dr):
    """Evaluate the model on the input data"""
    print('[INFO] Evaluating...')
    model.eval()
    outputs = []
    total_loss = 0.0
    for batch_idx, batch in enumerate(dr):
        loss, batch_out = model(batch)
        total_loss += loss.item()

        # Detach the batch_out from the graph and move it to the CPU.
        # If the batch_out is a scalar, then convert it to a list with one element.
        batch_out = batch_out.detach().cpu().numpy().squeeze()
        if len(batch_out.shape) == 0:
            batch_out = [batch_out]
        outputs.extend(batch_out)

    return round(total_loss / len(dr), 5), outputs


def test(config, test_dr, model):
    """Test the model on the test set. Print classification report and plot confusion matrix."""
    model.load_model(model_path=config.test.model_path)

    # Evaluate the model on the test set.
    test_loss, outputs = evaluate_on_dataset(model=model, dr=test_dr)
    print(f'[INFO] Test loss: {test_loss}')

    # Calculate classification metrics.
    y_pred = [1 if output > config.test.threshold else 0 for output in outputs]
    y_true = [data['label'] for _, data in test_dr.dataset.data.iterrows()]

    print('[INFO] Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))

    print('[INFO] Plotting confusion matrix...')
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm=cm, y_true=y_true, y_pred=y_pred)

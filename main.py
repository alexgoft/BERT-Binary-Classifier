import os
import time
import torch

from sklearn.metrics import classification_report, confusion_matrix
from transformers import AdamW
from config_file import ConfigFile
from data_utils.data_loaders import create_datasets
from early_stopper import EarlyStopper
from model import BERTNewsClassifier
from utils import plot_losses, plot_confusion_matrix

# Model parameters.
CFG = {
    'general': {
        'seed': 42,
        'output_dir': 'outputs',
        'mode': 'train'  # 'test' or 'train'
    }, 'data': {
        'data_path': 'assignment_data_en.csv',
        'train_size': 0.6,
        'val_size': 0.5,  # Percentage of the data_utils left for validation (rest is for test).
        'max_seq_length': 512,
        'plot_histograms': True
    }, 'train': {
        'num_epochs': 20,
        'batch_size': 4,
        'lr': 1e-5,
        'weight_decay': 0.01,
        'eps': 1e-8,
        'dropout': 0.3,
        'sampler': "BalancedBatchSampler",   # "WeightedRandomSampler" / "BalancedBatchSampler", None
        'early_stopping': {
            'patience': 2,
            'min_delta': 0
        }},
    'test': {
        'model_path': 'outputs/20231007-193004/model_0.43864.pt',
        'threshold': 0.5},
    'model': {
        # 'model_name': 'google/bert_uncased_L-4_H-256_A-4',
        'model_name': 'bert-base-uncased',
        'n_classes': 1,  # If n_classes > 1, one-hot encoding is used. else integer encoding is used.
        'linear_layers_num': 1,  # Number of linear layers after the BERT model.
        'freeze_bert': False,  # If True, only train the classifier layers.
        'uncased': True,  # Bert uncased or cased (meaning case-sensitive)
    }
}


def test(config, test_dr, model):
    model.load_model(model_path=config.test.model_path)

    # Evaluate the model on the test set.
    test_loss, outputs = evaluate_end_epoch(model=model, val_dr=test_dr)
    print(f'[INFO] Test loss: {test_loss}')

    # Calculate classification metrics.
    y_pred = [1 if output > config.test.threshold else 0 for output in outputs]
    y_true = [data['label'] for _, data in test_dr.dataset.data.iterrows()]

    print('[INFO] Classification Report:')
    print(classification_report(y_true, y_pred, digits=4))

    print('[INFO] Plotting confusion matrix...')
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm=cm, y_true=y_true, y_pred=y_pred)


def train_epoch(model, optimizer, train_dr, epoch_idx, epochs_num,
                every_n_batches=25, scheduler=None):
    """Train the model for one epoch."""
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_dr):
        optimizer.zero_grad()

        loss, _ = model(batch)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        print(f'[INFO] '
              f'Epoch: {epoch_idx + 1}/{epochs_num} '
              f'Batch: {batch_idx + 1}/{len(train_dr)}')
        if (batch_idx + 1) % every_n_batches == 0:
            print(f'[INFO]\t\tTrain loss so far: {round(train_loss / (batch_idx + 1), 5)}')

    return round(train_loss / len(train_dr), 5)


def evaluate_end_epoch(model, val_dr):
    """Evaluate the model on the validation set."""
    print('[INFO] Evaluating...')
    model.eval()
    outputs = []
    val_loss = 0.0
    for batch_idx, batch in enumerate(val_dr):
        loss, batch_out = model(batch)
        val_loss += loss.item()

        batch_out = batch_out.detach().cpu().numpy().squeeze()
        outputs.extend(batch_out)
    return round(val_loss / len(val_dr), 5), outputs


def train(config, train_dr, val_dr, model):
    # Create output directory for the model and save the config file.
    output_dir_path = os.path.join(config.general.output_dir,
                                   time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir_path)
    config.save_config(os.path.join(output_dir_path, 'config.yaml'))

    # Initialize the early_stopping object.
    # If the validation loss does not improve after 'patience' epoch, stop training.
    early_stopping = EarlyStopper(patience=config.train.early_stopping.patience,
                                  min_delta=config.train.early_stopping.min_delta)

    # AdamW is an Adam variant with weight decay regularization.
    optimizer = AdamW(model.parameters(),
                      lr=config.train.lr,
                      weight_decay=config.train.weight_decay,
                      eps=config.train.eps,
                      correct_bias=False)
    scheduler = None

    # optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
    # total_steps = len(train_dr) * NUM_EPOCHS
    # num_warmup_steps = int(total_steps * 0.1)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=total_steps
    # )

    # training loop
    best_eval_loss = float('inf')

    train_loss_list = []
    valid_loss_list = []
    for epoch_idx in range(config.train.num_epochs):
        train_loss = train_epoch(model=model, train_dr=train_dr,
                                 optimizer=optimizer, scheduler=scheduler,
                                 epoch_idx=epoch_idx, epochs_num=config.train.num_epochs)
        train_loss_list.append(train_loss)

        with torch.no_grad():
            eval_loss, _ = evaluate_end_epoch(model=model, val_dr=val_dr)
            valid_loss_list.append(eval_loss)

            if eval_loss < best_eval_loss:
                model_path = os.path.join(output_dir_path, f'model_{eval_loss}.pt')
                torch.save(model.state_dict(), model_path)

                print(f'[INFO] Improved loss {best_eval_loss} ==> {eval_loss}. '
                      f'Saving model to {model_path}')

                best_eval_loss = eval_loss

        print(f'[INFO] Epoch: {epoch_idx + 1}/{config.train.num_epochs}')
        print(f'[INFO]\t\tTRAIN LOSS: {train_loss}')
        print(f'[INFO]\t\tVALIDATION LOSS: {eval_loss}')

        # Early stopping. If the validation loss stops improving, then stop training.
        if early_stopping.early_stop(validation_loss=eval_loss):
            print("[INFO] Early stopping...")
            break

    print(f'[INFO] Training finished. Output directory: {output_dir_path}')
    plot_losses(loss_values=train_loss_list, val_losses=valid_loss_list)


def main(config, mode='train'):
    # Set device to GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create model and datasets.
    model = BERTNewsClassifier(config=config, device=device)
    train_dr, val_dr, test_dr = create_datasets(config=config, device=device)

    # Train and test.
    if mode == 'train':
        train(config, train_dr, val_dr, model)
    elif mode == 'test':
        test(config, test_dr, model)


if __name__ == '__main__':
    config_ = ConfigFile.load(CFG)
    print(config_)

    main(config_, mode=config_.general.mode)

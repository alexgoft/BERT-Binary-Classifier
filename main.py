import torch
import os
import time

from config_file import ConfigFile
from data import create_datasets
from early_stopper import EarlyStopper
from model import BERTNewsClassifier
from utils import plot_losses

from transformers import AdamW, get_linear_schedule_with_warmup

# Model parameters.
CFG = {
    'general': {
        'seed': 42,
        'output_dir': 'outputs',
        'mode': 'train'  # 'test' or 'train'
    }, 'data': {
        'data_path': 'assignment_data_en.csv',
        'train_size': 0.8,
        'val_size': 0.5,  # Percentage of the data left for validation (rest is for test).
        'max_seq_length': 256
    }, 'train': {
        'num_epochs': 20,
        'batch_size': 16,
        'lr': 1e-5,
        'dropout': 0.3,
        'early_stopping': {
            'patience': 2,
            'min_delta': 0.1
        }},
    'test': {'model_path': 'outputs/20201220-155809/model_0.0001.pt'},
    'model': {
        'model_name': 'google/bert_uncased_L-4_H-256_A-4',
        # If n_classes is 1, integer encoding is used.
        # If n_classes > 1, one-hot encoding is used.
        'n_classes': 1,
        'linear_layers_num': 2,  # Number of linear layers after the BERT model.
        'freeze_bert': False,  # If True, only train the classifier layers.
        'max_seq_length': 256,  # Max sequence length for the BERT model.
        'uncased': True,  # Bert uncased or cased (meaning case-sensitive)
    }
}


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
    val_loss = 0.0
    for batch_idx, batch in enumerate(val_dr):
        loss, _ = model(batch)
        val_loss += loss.item()
    return round(val_loss / len(val_dr), 5)


def train(config, train_dr, val_dr, model, output_dir_path, device):
    # Initialize the early_stopping object.
    # If the validation loss does not improve after 'patience' epoch, stop training.
    early_stopping = EarlyStopper(patience=config.train.early_stopping.patience,
                                  min_delta=config.train.early_stopping.min_delta)

    # AdamW is an Adam variant with weight decay regularization.
    optimizer = AdamW(model.parameters(), lr=config.train.lr, correct_bias=False)
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
    train_acc_list = []
    valid_acc_list = []
    for epoch_idx in range(config.train.num_epochs):
        train_loss = train_epoch(model=model, train_dr=train_dr,
                                 optimizer=optimizer, scheduler=scheduler,
                                 epoch_idx=epoch_idx, epochs_num=config.train.num_epochs)
        train_loss_list.append(train_loss)

        with torch.no_grad():
            eval_loss = evaluate_end_epoch(model=model, val_dr=val_dr)
            valid_loss_list.append(eval_loss)

            if eval_loss < best_eval_loss:
                print(f'[INFO] Improved loss {best_eval_loss} ==> {eval_loss}. '
                      f'Saving model to {model_path}')

                best_eval_loss = eval_loss

                model_path = os.path.join(output_dir_path, f'model_{eval_loss}.pt')
                torch.save(model.state_dict(), model_path)

        print(f'[INFO] Epoch: {epoch_idx + 1}/{config.train.num_epochs}')
        print(f'[INFO]\t\tTRAIN LOSS: {train_loss}')
        print(f'[INFO]\t\tVALIDATION LOSS: {eval_loss}')

        # Early stopping. If the validation loss stops improving, then stop training.
        if early_stopping.early_stop(validation_loss=eval_loss):
            print("[INFO] Early stopping...", epoch_idx)
            break

    print(f'[INFO] Training finished. Output directory: {output_dir_path}')
    plot_losses(loss_values=train_loss_list, val_losses=valid_loss_list,
                train_accuracies=train_acc_list, val_accuracies=valid_acc_list)


def main(config, mode='train'):
    # Set device to GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create output directory.
    output_dir_path = os.path.join(config.general.output_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir_path)

    # Create model and datasets.
    model = BERTNewsClassifier(config=config, device=device)
    train_dr, val_dr, test_dr = create_datasets(config=config, device=device)

    # Train and test.
    if mode == 'train':
        train(config, train_dr, val_dr, model, output_dir_path, device)
    # else:
    #     model_path = config.test.model_path
    #     model.load_model(model_path)
    #
    #     test_loss = evaluate_end_epoch(model=model, val_dr=test_dr)
    #     print(f'[INFO] Test loss: {test_loss}')


if __name__ == '__main__':
    config_ = ConfigFile.load(CFG)
    main(config_, mode=config_.general.mode)

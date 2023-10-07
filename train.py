import torch
import os
import time

from data import create_datasets
from model import NewsClassifier
from utils import plot_losses
from argparse import Namespace

from transformers import AdamW, get_linear_schedule_with_warmup

# Training parameters.
NUM_EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-5

# Model parameters.
MODEL_CFG = Namespace(**
                      {
                          'model_name': 'google/bert_uncased_L-4_H-256_A-4',
                          'n_classes': 1,
                          'freeze_bert': False,
                          'max_seq_length': 512,
                          'uncased': True,  # Bert uncased or cased (meaning case-sensitive)
                      }
                      )


class EarlyStopper:
    """
    Early stopping to stop the training if the validation loss stops improving.
    Arguments:
        min_delta: float, minimum change in the monitored quantity to qualify as an improvement.
        patience: int, number of epochs to wait for improvement before stopping.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# def test(test_dr, model, output_dir_path, device):
#     pass


def train_epoch(model, optimizer, train_dr, epoch_idx,
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
              f'Epoch: {epoch_idx + 1}/{NUM_EPOCHS} '
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


def train(train_dr, val_dr, model, output_dir_path, device):
    # Initialize the early_stopping object.
    # If the validation loss does not improve after 'patience' epoch, stop training.
    early_stopping = EarlyStopper(patience=2)

    # AdamW is an Adam variant with weight decay regularization.
    optimizer = AdamW(model.parameters(), lr=LR)
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
    for epoch_idx in range(NUM_EPOCHS):

        train_loss = train_epoch(model=model,
                                 optimizer=optimizer, scheduler=scheduler,
                                 train_dr=train_dr, epoch_idx=epoch_idx)
        train_loss_list.append(train_loss)

        with torch.no_grad():
            eval_loss = evaluate_end_epoch(model=model, val_dr=val_dr)
            valid_loss_list.append(eval_loss)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss

                model_path = os.path.join(output_dir_path, f'model_{eval_loss}.pt')
                torch.save(model.state_dict(), model_path)

                print(f'[INFO] Improved loss {best_eval_loss} ==> {eval_loss}. '
                      f'Saving model to {model_path}')

        # Early stopping. If the validation loss stops improving, then stop training.
        if early_stopping.early_stop(validation_loss=eval_loss):
            print("[INFO] Early stopping..We are at epoch:", epoch_idx)
            break

        print(f'[INFO] Epoch: {epoch_idx + 1}/{NUM_EPOCHS}')
        print(f'[INFO]\t\tTRAIN LOSS: {train_loss}')
        print(f'[INFO]\t\tVALIDATION LOSS: {eval_loss}')
    print(f'[INFO] Training finished. Output directory: {output_dir_path}')

    plot_losses(loss_values=train_loss_list, val_losses=valid_loss_list,
                train_accuracies=train_acc_list, val_accuracies=valid_acc_list,
                n_epochs=NUM_EPOCHS)


def main():
    # Set device to GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create output directory.
    output_dir_path = os.path.join('outputs', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir_path)

    # Create model and datasets.
    model = NewsClassifier(config=MODEL_CFG, device=device)
    train_dr, val_dr, test_dr = create_datasets(model_config=MODEL_CFG,
                                                batch_size=BATCH_SIZE,
                                                device=device)

    # Train and test.
    train(train_dr, val_dr, model, output_dir_path, device)
    # test(test_dr, model, output_dir_path, device)


if __name__ == '__main__':
    main()

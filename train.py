import torch
import os
import time

from transformers import AdamW, get_linear_schedule_with_warmup

from data import create_datasets
from model import NewsClassifier
from utils import plot_losses

# Training parameters.
NUM_EPOCHS = 20
BATCH_SIZE = 8
LR = 2e-5


def test(test_dr, model, output_dir_path, device):
    pass


def train(train_dr, val_dr, model, output_dir_path, device):

    # Adam is an optimization algorithm that can used
    # instead of the classical stochastic gradient descent procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = None

    # # AdamW is an Adam variant with weight decay regularization.
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

        train_loss = train_epoch(model=model, optimizer=optimizer, scheduler=scheduler,
                                 train_dr=train_dr, epoch_idx=epoch_idx)
        eval_loss = evaluate_end_epoch(model=model, val_dr=val_dr)

        print(f'[INFO] Epoch: {epoch_idx + 1}/{NUM_EPOCHS}')
        print(f'[INFO]\t\tTRAIN LOSS: {train_loss}')
        print(f'[INFO]\t\tVALIDATION LOSS: {eval_loss}')

        if eval_loss < best_eval_loss:
            model_path = os.path.join(output_dir_path, f'model_{eval_loss}.pt')
            print(f'[INFO] Improved loss {best_eval_loss} ==> {eval_loss}. '
                  f'Saving model to {model_path}')

            torch.save(model.state_dict(), model_path)
            best_eval_loss = eval_loss

        # Save losses for plotting.
        train_loss_list.append(train_loss)
        valid_loss_list.append(eval_loss)
    print(f'[INFO] Training finished. Output directory: {output_dir_path}')

    plot_losses(loss_values=train_loss_list, val_losses=valid_loss_list,
                train_accuracies=train_acc_list, val_accuracies=valid_acc_list,
                n_epochs=NUM_EPOCHS)


def train_epoch(model, optimizer, train_dr, epoch_idx, every_n_batches=25, scheduler=None):
    """Train the model for one epoch."""
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_dr):

        loss, _ = model(batch)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        print(f'[INFO] '
              f'Epoch: {epoch_idx + 1}/{NUM_EPOCHS} '
              f'Batch: {batch_idx + 1}/{len(train_dr)}')
        if (batch_idx + 1) % every_n_batches == 0:
            print(f'[INFO]\t\tTrain loss: {round(train_loss / (batch_idx + 1), 5)} so far.')

    return round(train_loss / len(train_dr), 5)


def evaluate_end_epoch(model, val_dr):
    """Evaluate the model on the validation set."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dr):
            loss, _ = model(batch)

            val_loss += loss.item()
    return round(val_loss / len(val_dr), 5)


def main():
    # Set device to GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create output directory.
    output_dir_path = os.path.join('outputs', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir_path)

    # Create model and datasets.
    model = NewsClassifier(device=device)
    train_dr, val_dr, test_dr = create_datasets(batch_size=BATCH_SIZE, device=device)

    # Train and test.
    train(train_dr, val_dr, model, output_dir_path, device)
    test(test_dr, model, output_dir_path, device)


if __name__ == '__main__':
    main()

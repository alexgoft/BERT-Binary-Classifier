import os
import time

import numpy as np
import torch
from torch.utils.data import Sampler
from torch.utils.data import WeightedRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from plot_utils import plot_losses
from test_utils import evaluate_on_dataset


# -------------------------------------------------------- #
# -------------------- Early Stopping -------------------- #
# -------------------------------------------------------- #
class EarlyStopper:
    """
    Early stopping to stop the training if the validation loss stops improving.
    Arguments:
        min_delta: float, minimum change in the monitored quantity to qualify as an improvement.
        patience: int, number of epochs to wait for improvement before stopping.
    """

    def __init__(self, patience, min_delta):
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


# -------------------------------------------------------- #
# ----------------------- Sampler ------------------------ #
# -------------------------------------------------------- #
class BalancedBatchSampler(Sampler):
    """
        Ensure that each batch has the same number of samples from each class.
        In this case we oversample - adding more examples from the minority class.
    """

    def __init__(self, dataset, data_source=None):
        super().__init__(data_source)
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

    def __iter__(self):
        # Get the class with the maximum count
        labels = self.dataset['label'].to_numpy()
        class_counts = np.bincount(labels)
        max_class_count = class_counts.max().item()

        # Duplicate samples from classes with fewer samples
        indices = []
        for class_idx, class_count in enumerate(class_counts):
            class_indices = [i for i, label in enumerate(labels) if label == class_idx]
            indices += class_indices * (max_class_count // class_count)
            indices += class_indices[:max_class_count % class_count]

        assert len(indices) == max_class_count * len(class_counts)

        # Shuffle the indices
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_weighted_random_sampler(train_df):
    """
        WeightedRandomSampler does not guarantee that each batch will have an
        equal number of samples from each class. It only ensures that over
        the entire epoch, each class is represented approximately equally.
    """
    # Compute class distribution
    target = train_df['label'].to_numpy()
    class_count = np.bincount(target)

    # Assign weights to each sample in the dataset
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    sample_weights = class_weights[target]

    # Create a sampler with these weights
    return WeightedRandomSampler(weights=sample_weights,
                                 num_samples=len(sample_weights))


def get_sampler(train_df, config):
    """Get the sampler for the training data_utils."""
    sampler = None
    if config.train.sampler == "BalancedBatchSampler":
        return BalancedBatchSampler(train_df)
    elif config.train.sampler == "WeightedRandomSampler":
        return get_weighted_random_sampler(train_df)
    return sampler


# -------------------------------------------------------- #
# -------------------- Training Loop --------------------- #
# -------------------------------------------------------- #
def train_epoch(model, optimizer, train_dr, epoch_idx, epochs_num,
                every_n_batches=25, scheduler=None):
    """Train the model for one epoch."""
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(train_dr):
        optimizer.zero_grad()

        loss, _, _ = model(batch)
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

    # Create the learning rate scheduler. The learning rate is linearly
    # increased for the first 10% of the steps and then linearly decreased
    # for the remaining steps.
    total_steps = len(train_dr) * config.train.num_epochs
    num_warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

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
            eval_loss, _, _ = evaluate_on_dataset(model=model, dr=val_dr)
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

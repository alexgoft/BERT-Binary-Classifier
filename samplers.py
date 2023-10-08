import torch
import numpy as np

from torch.utils.data import WeightedRandomSampler

import torch
from torch.utils.data import DataLoader, Sampler


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

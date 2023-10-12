import torch
import os
import time

from config_file import ConfigFile
from model import BERTNewsClassifier
from data_utils import create_datasets
from test_utils import test
from train_utils import train

# <><><><><><><><><><><><><><><> #
#                                #
#   Pet the cat for good luck:   #
#           ,_     _,            #
#           |\\___//|            #
#           |=o   o=|            #
#           \=._Y_.=/            #
#            )  `  (    ,        #
#           /       \  ((        #
#           |       |   ))       #
#          /| |   | |\_//        #
#          \| |._.| |/-`         #
#                                #
# <><><><><><><><><><><><><><><> #

MODE = 'test'

# If mode is 'train', this is the path to the config file.
CFG_PATH = 'configs/config.yaml'

# If mode is 'test', this is the path to the model to test.
# path to config file is taken from the model's output directory.
# All the metrics will be saved to a directory with the same name
# as the model with the suffix "_metrics".
MODEL_PATH = 'outputs/20231010-210438/model_0.43796.pt'


def run(output_dir_path, config_path, mode='train'):

    # Load the config file. Print it for debugging.
    config = ConfigFile.load(config_path)
    print(config)

    # Set device to GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create model and datasets.
    model = BERTNewsClassifier(config=config, device=device)
    train_dr, val_dr, test_dr = create_datasets(config=config,
                                                device=device,
                                                output_dir_path=output_dir_path)

    # Train and test.
    if mode == 'train':
        # Save the config file of current run to the output directory.
        config.save_config(os.path.join(output_dir_path, 'config.yaml'))

        train(config, train_dr, val_dr, model, output_dir_path)
    elif mode == 'test':
        test(config, test_dr, model, output_dir_path)


def main(mode):

    if mode == 'train':
        output_dir_path = os.path.join('outputs', time.strftime("%Y%m%d-%H%M%S"))
        config_path = CFG_PATH

    elif mode == 'test':
        model_name = MODEL_PATH.rpartition('/')[-1].split('.pt')[0].replace('.', '_')
        model_output_dir = MODEL_PATH.rpartition('/')[0]

        # Create a new directory for the metrics.
        output_dir_path = f'{model_output_dir}/{model_name}_metrics'
        config_path = f'{model_output_dir}/config.yaml'

    else:
        raise ValueError(f'Unknown mode: {mode}')
    os.makedirs(output_dir_path, exist_ok=True)

    # Run the model in the specified mode (train or test) with the specified config file.
    run(mode=mode, config_path=config_path, output_dir_path=output_dir_path)


if __name__ == '__main__':
    main(mode=MODE)

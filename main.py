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

MODE = 'train'
CFG_PATH = 'configs\\config.yaml'
OUTPUT_DIR = 'outputs'


# TODO: Move this to a utils file?
def get_output_dir_path(config, mode=MODE, output_dir=OUTPUT_DIR):
    # If training, create a new directory for the run.
    if mode == 'train':
        output_dir_path = os.path.join(output_dir, time.strftime("%Y%m%d-%H%M%S"))

    # If testing, use the model's output directory.
    elif mode == 'test':
        model_name = config.test.model_path.rpartition('/')[-1].split('.pt')[0].replace('.', '_')
        model_output_dir = config.test.model_path.rpartition('/')[0]
        output_dir_path = f'{model_output_dir}/{model_name}_metrics'

    else:
        raise ValueError(f'Unknown mode: {config.general.mode}')

    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path


def main(mode, config_path=CFG_PATH):

    # Load the config file. Print it for debugging.
    config = ConfigFile.load(CFG_PATH)
    print(config)

    # Set device to GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Create output directory for the model and save the config file.
    output_dir_path = get_output_dir_path(config, mode)

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


if __name__ == '__main__':
    main(mode=MODE)

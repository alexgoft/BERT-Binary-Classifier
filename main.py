import torch

from config_file import ConfigFile
from data_loaders import create_datasets
from model import BERTNewsClassifier
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

CFG_PATH = 'configs/config.yaml'


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
    config_ = ConfigFile.load(CFG_PATH)
    print(config_)

    main(config_, mode=config_.general.mode)

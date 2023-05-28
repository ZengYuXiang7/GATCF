# coding : utf-8
# Author : yuxiang Zeng
import platform

from torch.utils.data import DataLoader
from datasets.dataset import *



def get_dataloaders(dataset, args):
    train, valid, test = dataset.get_tensor()
    train_set = QoSDataset(train, args)
    valid_set = QoSDataset(valid, args)
    test_set = QoSDataset(test, args)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        num_workers=8 if platform.system() == 'Linux' else 0,
        prefetch_factor=4 if platform.system() == 'Linux' else 2
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=8094,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=8 if platform.system() == 'Linux' else 0,
        prefetch_factor=4 if platform.system() == 'Linux' else 2
    )
    test_loader = DataLoader(
        test_set,
        batch_size=8094,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=8 if platform.system() == 'Linux' else 0,
        prefetch_factor=4 if platform.system() == 'Linux' else 2
    )

    return train_loader, valid_loader, test_loader

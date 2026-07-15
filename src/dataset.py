import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import *

from datasets import load_from_disk

def get_dataloader(train=True):
    path = str(PROCESSED_DATA_DIR / ('train' if train else 'test'))
    dataset = load_from_disk(path)
    dataset.set_format(type='torch')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)

    for batch in train_dataloader:
        for key, value in batch.items():
            print(key, '->', value.shape)
        break
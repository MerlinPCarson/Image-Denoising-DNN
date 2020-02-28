import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import h5py


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        super(Dataset, self).__init__()
        self.file_name = file_name
        with h5py.File(file_name, 'r') as data:
            self.keys = list(data.keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as data:
            example = np.array(data[self.keys[index]])
        return torch.Tensor(example)

    def shape(self):
        with h5py.File(self.file_name, 'r') as data:
            return np.array(data[self.keys[0]]).shape

        

def main():

    parser = argparse. ArgumentParser(description='Image Denoising Trainer')
    parser.add_argument('--train_set', type=str, default='train.h5', help='h5 file with training vectors')
    parser.add_argument('--val_set', type=str, default='val.h5', help='h5 file with validation vectors')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    args = parser.parse_args()

    assert os.path.exists(args.train_set), f'Cannot find training vectors file {args.train_set}'
    assert os.path.exists(args.val_set), f'Cannot find validation vectors file {args.train_set}'


    print('Loading datasets')

    train_data = Dataset(args.train_set)
    val_data = Dataset(args.val_set)

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(val_data)}')

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)

    # input shape for each example to network, NOTE: channels first
    num_channels, patch_height, patch_width = train_data.shape()

    print(f'Input shape to model forward will be: ({args.batch_size}, {num_channels}, {patch_height}, {patch_width})')

    return 0

if __name__ == '__main__':
    sys.exit(main())

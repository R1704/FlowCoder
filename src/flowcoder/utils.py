import torch
import torch.nn as nn
import math
import csv
import os


class PositionalEncoding(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout=0.1, max_len=5_000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def flatten(lst):
    return [element for sublist in lst for element in sublist]


# Define a function to append a new row to an existing CSV file
def append_to_csv(filename, row_data):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)


# Define a function to create a new CSV file with headers
def create_csv(filename, headers):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)


# Make a unique filename (so we don't overwrite when saving to files) or get the last file, depending on context
def get_checkpoint_filename(checkpoint_dir, find_last=False, base_name='gflownet', ext='pth'):
    idx, last_file = -1, None

    while True:
        idx += 1
        filename = f'{base_name}_{idx}.{ext}'
        filepath = os.path.join(checkpoint_dir, filename)

        if os.path.isfile(filepath):
            last_file = filename
        else:
            return os.path.join(checkpoint_dir, last_file) if find_last else filepath

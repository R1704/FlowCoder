import torch
import torch.nn as nn
import math
import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


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
    # List all checkpoint files in the directory
    checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.startswith(f'{base_name}_') and file.endswith(f'.{ext}')]

    if not checkpoint_files:
        # No checkpoint files found
        return None if find_last else os.path.join(checkpoint_dir, f'{base_name}_0.{ext}')

    # Sort the checkpoint files by index
    checkpoint_files.sort(key=lambda file: int(file.split('_')[1].split('.')[0]))

    if find_last:
        # Return the last checkpoint file
        return os.path.join(checkpoint_dir, checkpoint_files[-1])
    else:
        # Generate a new unique filename
        last_index = int(checkpoint_files[-1].split('_')[1].split('.')[0])
        new_index = last_index + 1
        return os.path.join(checkpoint_dir, f'{base_name}_{new_index}.{ext}')


def save_results(mode, depth, epoch, e_step, batch_program_names, programs, states, rewards, batch_size, max_reward, csv_file):
    for i in range(batch_size):
        epoch_data = [mode, depth, epoch, e_step, batch_program_names[i], programs[i], states[i], (rewards[i] == max_reward).item(), rewards[i].item()]
        append_to_csv(csv_file, epoch_data)


def plot_results(e_step_losses, m_step_losses, all_logZs, epoch, filepath):
    plt.figure(figsize=(15, 15))

    data = [(e_step_losses, 'E-step Losses Over Time', 'e loss'),
            (m_step_losses, 'M-step Losses Over Time', 'm loss'),
            (np.exp(all_logZs), 'Z Over Time', 'Z')]

    for i, (d, title, ylabel) in enumerate(data, start=1):
        plt.subplot(3, 1, i)
        plt.plot(d)
        plt.title(title)
        plt.xlabel('epochs')
        plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(os.path.join(filepath, f'epoch {epoch}'))
    plt.show()


# https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def add_unique_data(data_list: list, total_data: list, unique_solutions: set):
    for data_tuple in data_list:
        program, _, _, task_name = data_tuple
        if (program, task_name) not in unique_solutions:
            total_data.append(data_tuple)
            unique_solutions.add((program, task_name))


# Function to extract embeddings
def extract_embeddings(encoder, dataset):
    with torch.no_grad():
        embeddings = []
        for data in dataset:
            embedding = encoder(data)  # Assuming data can be directly passed to the encoder
            embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)


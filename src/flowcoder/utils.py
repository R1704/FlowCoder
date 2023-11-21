import torch
import torch.nn as nn
import math
import csv
import os
import time
import matplotlib.pyplot as plt
import numpy as np


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


def calculate_programs_per_second(start_time, program_counter):
    elapsed_time = time.time() - start_time
    avg_programs_per_sec = program_counter / elapsed_time
    return avg_programs_per_sec


def correct_programs(data):
    correct = [(prog[i], state[i], io[i], real_prog[i], rew[i])
               for (prog, state, io, real_prog, rew) in data
               for i in range(len(rew)) if rew[i] == 1]
    return correct


def save_results(mode, depth, epoch, e_step, batch_program_names, programs, rewards, batch_IOs, batch_size, max_reward, csv_file):
    for i in range(batch_size):
        if rewards[i] == max_reward:
            epoch_data = [mode, depth, epoch, e_step, batch_program_names[i], programs[i], batch_IOs[i]]
            append_to_csv(csv_file, epoch_data)


def print_stats(start_time, total_correct, total_data, batch_size):
    print(f'Solved {len(total_correct)} out of {len(total_data) * batch_size} tasks correctly')
    for i, (program, state, task, real_program, _) in enumerate(total_correct):
        print('=' * 50)
        print(f'program     : {program}\n')
        print(f'state       : {state}\n')
        print(f'task        : {task}\n')
        print(f'real_program: {real_program}\n')
        print('=' * 50)
    print(f'Average programs per second: {calculate_programs_per_second(start_time, len(total_data))}')


def plot_results(e_step_losses, m_step_losses, all_logZs, program_ratios, epoch, filepath):
    plt.figure(figsize=(15, 15))

    data = [(e_step_losses, 'E-step Losses Over Time', 'e loss'),
            (m_step_losses, 'M-step Losses Over Time', 'm loss'),
            (np.exp(all_logZs), 'Z Over Time', 'Z'),
            (program_ratios, 'Correct Program Ratios Over Time', 'ratio')]

    for i, (d, title, ylabel) in enumerate(data, start=1):
        plt.subplot(4, 1, i)
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
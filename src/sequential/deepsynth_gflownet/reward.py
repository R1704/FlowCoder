import torch.nn as nn
import torch
import math
import editdistance
import numpy as np
from src.sequential.deepsynth_gflownet.utils import *


class Reward(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=2, dropout=0.1, device='cpu'):
        super(Reward, self).__init__()
        self.device = device
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.cosim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, true, pred):
        if len(pred) == 0:
            return torch.tensor(0.0, device=self.device)
        reward = self.cosine_sim(true, pred)
        # reward = self.edit_distance(true, pred)
        return torch.tensor(reward, device=self.device)

    def naive(self, true, pred):
        return torch.tensor(true == pred, device=self.device).to(torch.int8)

    def cosine_sim(self, true, pred):
        latent_true = torch.mean(
        self.transformer_encoder(self.pos_encoder(self.embedding(torch.tensor(true).to(torch.int64)))), dim=0)
        latent_pred = torch.mean(
        self.transformer_encoder(self.pos_encoder(self.embedding(torch.tensor(pred).to(torch.int64)))), dim=0)

        cosim = self.cosim(latent_true, latent_pred)
        norm_cosim = (cosim + 1) / 2
        return norm_cosim

    def edit_distance(self, true, pred):
        def list_to_str(lst):
            return ''.join(map(str, lst))

        ed = editdistance.eval(list_to_str(true), list_to_str(pred))
        print(true, pred, ed, len(true) - ed)
        return float(ed / (len(true) + len(pred)))


    def mean_squared_error(self, y_true, y_pred):
        # Convert lists to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # # If one of the vectors is empty, return maximum loss
        # if y_true.size == 0 and y_pred.size == 0:
        #     return 1
        #
        # If one of the vectors is empty, return maximum loss
        if y_true.size == 0 or y_pred.size == 0:
            return 0

        # Pad the shorter vector with zeros
        if len(y_true) < len(y_pred):
            y_true = np.pad(y_true, (0, len(y_pred) - len(y_true)))
        elif len(y_pred) < len(y_true):
            y_pred = np.pad(y_pred, (0, len(y_true) - len(y_pred)))

        # Compute mean squared error
        mse = np.mean((y_true - y_pred)**2)
        reward = np.exp(-mse)
        # reward /= np.abs(y_true.shape[0] - y_pred.shape[0])
        return reward

    def naive_reward(self, y_true, y_pred):
        if len(y_pred) == 0:
            return 0
        return sum([x == y for x, y in zip(y_pred, y_true)]) / len(y_true)
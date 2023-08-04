import torch
from torch import nn
import math
import logging

from src.sequential.deepsynth_gflownet.utils import *

class GFlowNet(nn.Module):
    def __init__(self, cfg, io_encoder, state_encoder, d_model=512, num_heads=8, num_layers=2, dropout=0.1, device='cpu'):
        super(GFlowNet, self).__init__()
        self.device = device
        self.cfg = cfg
        self.io_encoder = io_encoder
        self.state_encoder = state_encoder
        self.positional_encoding = PositionalEncoding(d_model)


        # Defining the transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout
            )

        # MLPs for logits and logZ
        self.forward_logits = GFlowNet_Forward(d_model, len(state_encoder.rules))

        self.logZ = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
            )

    def forward(self, state, io):

        # Process IO
        io = self.io_encoder(io)
        io = self.positional_encoding(io)

        # Process state
        state = self.state_encoder(state)
        state = self.positional_encoding(state)

        # Pass through the transformer
        transformer_output = self.transformer(io, state)

        # Predict the forward logits and total flow logZ
        forward_logits = self.forward_logits(transformer_output)[-1]

        logZ = self.logZ(transformer_output)[-1]

        return forward_logits, logZ

    # TODO: Do i need this?
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

class GFlowNet_Forward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GFlowNet_Forward, self).__init__()
        self.forward_logits = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, output_dim)
                )

    def forward(self, x):
        return self.forward_logits(x)
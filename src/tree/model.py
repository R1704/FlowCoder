import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from grammar import *

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoder = PositionalEncoding(vocab_size, embedding_dim, primitives)

        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.next_node_head = nn.Linear(embedding_dim, 2)
        self.node_type_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, tree):
        x = self.embedding(x) + self.positional_encoder.encode(tree)

        x = self.transformer_encoder(x)

        next_node_logits = self.next_node_head(x)
        node_type_logits = self.node_type_head(x)

        return next_node_logits, node_type_logits

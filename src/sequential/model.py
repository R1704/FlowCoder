import torch
import torch.nn as nn
from torch import jit


class FlowModel(nn.Module):
    def __init__(self,
                 tokenizer,
                 device,
                 embedding_dim=128,
                 num_layers=3,
                 num_heads=8,
                 dropout_prob=0.1
                 ):
        super(FlowModel, self).__init__()

        # Define the vocabulary
        self.tokenizer = tokenizer

        # Define the device
        self.device = device

        # Define the embedding layer
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(len(tokenizer.vocab), embedding_dim)

        # Define the transformer
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout_prob)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define the linear layer for decoding the output token
        self.decoder = nn.Linear(embedding_dim, len(tokenizer.vocab))

        # Define the logZ parameter
        self.logZ = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Get the index of the input sequence
        input_seq_idx = torch.tensor(self.tokenizer.encode(x), dtype=torch.long).to(self.device)
        # Add a batch dimension
        input_seq_idx = input_seq_idx.unsqueeze(1)
        # Embed the input sequence
        embds = self.embeddings(input_seq_idx)
        # Encode the input sequence
        input_encoding = self.encoder(embds)
        # Decode the output sequence (we only care about the last token)
        output_encoding = self.decoder(input_encoding[-1])
        # Return the logits
        logits = output_encoding
        return logits



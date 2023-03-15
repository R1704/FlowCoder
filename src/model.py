import torch
import torch.nn as nn


class FlowModel(nn.Module):
    def __init__(self, vocab, device):
        super(FlowModel, self).__init__()

        # Define the vocabulary
        self.vocab = vocab

        self.device = device

        # Define the embedding layer
        embedding_dim = 128
        self.embeddings = nn.Embedding(len(vocab), embedding_dim)

        # Define the transformer
        num_layers = 3
        hidden_dim = 256
        num_heads = 8
        dropout_prob = 0.1

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout_prob)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define the linear layer for decoding the output token
        self.decoder = nn.Linear(embedding_dim, len(vocab))

        self.logZ = nn.Parameter(torch.ones(1))

    def forward(self, x):
        input_seq_idx = torch.tensor([self.vocab.index(token) for token in x], dtype=torch.long).to(self.device)
        input_seq_idx = input_seq_idx.unsqueeze(1)
        embds = self.embeddings(input_seq_idx)
        input_encoding = self.encoder(embds)
        output_encoding = self.decoder(input_encoding[-1])
        logits = output_encoding
        return logits

import torch
import torch.nn as nn


class FlowModel(nn.Module):
    def __init__(self,
                 tokenizer,
                 device,
                 d_model=512,
                 num_layers=6,
                 num_heads=8,
                 dropout_prob=0.1
                 ):
        super(FlowModel, self).__init__()

        # Define the vocabulary
        self.tokenizer = tokenizer

        # Define the device
        self.device = device

        # Define the embedding layer
        self.d_model = d_model
        self.embeddings = nn.Embedding(len(tokenizer.vocab), d_model)

        # Define the transformer
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout_prob)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define the linear layer for decoding the output token
        self.decoder = nn.Linear(d_model, len(tokenizer.vocab))

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

    def resize_token_embeddings(self, num_tokens):
        old_embeddings = self.embeddings
        self.embeddings = nn.Embedding(num_tokens, self.d_model).to(self.device)
        min_tokens = min(old_embeddings.weight.shape[0], num_tokens)
        self.embeddings.weight.data[:min_tokens, :] = old_embeddings.weight.data[:min_tokens, :]

    def resize_decoder_weights(self, num_tokens):
        old_decoder = self.decoder
        self.decoder = nn.Linear(self.d_model, num_tokens).to(self.device)
        min_tokens = min(old_decoder.weight.shape[0], num_tokens)
        self.decoder.weight.data[:min_tokens, :] = old_decoder.weight.data[:min_tokens, :]
        self.decoder.bias.data[:min_tokens] = old_decoder.bias.data[:min_tokens]


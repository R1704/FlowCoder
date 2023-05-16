import torch
import torch.nn as nn
from src.sequential.config import *
import math


class GFlowNet_Encoder(nn.Module):
    def __init__(self, num_tasks, embedding_dim):
        super(GFlowNet_Encoder, self).__init__()
        self.embedding = nn.Embedding(num_tasks, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

class GFlowNet_Z(nn.Module):
    def __init__(self):
        super(GFlowNet_Z, self).__init__()

    def forward(self, x):
        ...

class GFlowNet_Forward(nn.Module):
    def __init__(self):
        super(GFlowNet_Forward, self).__init__()

    def forward(self, x):
        ...



class GFlowNet(nn.Module):
    def __init__(self,
                 tokenizer,
                 device,
                 d_model=512,
                 num_decoder_layers=6,
                 num_heads=8,
                 dropout_prob=0.1
                 ):
        super(GFlowNet, self).__init__()



        # Define the vocabulary
        self.tokenizer = tokenizer

        # Define the device
        self.device = device

        # Define the embedding layer
        self.d_model = d_model
        self.embeddings = nn.Embedding(len(tokenizer.vocab), d_model)

        # Define the positional encoding, call to staticmethod
        self.pos_enc = self.positional_encoding(max_trajectory, d_model).to(device)

        # Define the transformer
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        # Add the transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout_prob)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Define the linear layer for decoding the output token
        self.output_layer = nn.Linear(d_model, len(tokenizer.vocab))

        # Define the logZ parameter
        self.logZ = nn.Parameter(torch.ones(1))

    def forward(self):
        # Initialize the target tensor with a single <START> token
        start_token_idx = self.tokenizer.token_to_idx['<START>']
        target = torch.full((1, 1), start_token_idx, dtype=torch.long).to(self.device)

        output_tokens = []
        log_probs = []

        # Create a dummy memory tensor
        memory = torch.zeros((1, 1, self.d_model)).to(self.device)

        for _ in range(max_trajectory):
            # Embed the target sequence and add positional encoding
            target_embds = self.embeddings(target) + self.pos_enc[:, :target.size(1), :]

            # Decode the output sequence using the transformer decoder
            output_encoding = self.decoder(target_embds, memory)

            # Decode the output encoding to logits using the linear output layer
            logits = self.output_layer(output_encoding[-1])

            # Compute the softmax probabilities and log probabilities
            probs = torch.softmax(logits, dim=-1)
            log_probs_tensor = torch.log(probs)

            # Sample the next token from the logits
            next_token_idx = torch.multinomial(probs, num_samples=1).item()

            # Get the log probability of the sampled token
            log_prob = log_probs_tensor[0, next_token_idx].item()

            # Break the loop if the <STOP> token is generated
            if next_token_idx == self.tokenizer.token_to_idx['<STOP>']:
                break

            # Add the token to the output tokens
            output_tokens.append(next_token_idx)

            # Add the log probability to the log_probs list
            log_probs.append(log_prob)

            # Append the token to the target tensor and continue decoding
            next_token_tensor = torch.tensor([[next_token_idx]], dtype=torch.long).to(self.device)
            target = torch.cat([target, next_token_tensor], dim=0)

        return output_tokens, torch.tensor(log_probs).sum()

    def resize_token_embeddings(self, num_tokens):
        old_embeddings = self.embeddings
        self.embeddings = nn.Embedding(num_tokens, self.d_model).to(self.device)
        min_tokens = min(old_embeddings.weight.shape[0], num_tokens)
        self.embeddings.weight.data[:min_tokens, :] = old_embeddings.weight.data[:min_tokens, :]

    def resize_decoder_weights(self, num_tokens):
        old_output_layer = self.output_layer
        self.output_layer = nn.Linear(self.d_model, num_tokens).to(self.device)
        min_tokens = min(old_output_layer.weight.shape[0], num_tokens)
        self.output_layer.weight.data[:min_tokens, :] = old_output_layer.weight.data[:min_tokens, :]
        self.output_layer.bias.data[:min_tokens] = old_output_layer.bias.data[:min_tokens]

    def reset(self):
        self.logZ = nn.Parameter(torch.ones(1))
        self.embeddings.reset_parameters()
        self.decoder.reset_parameters()

    @staticmethod
    def positional_encoding(position, d_model):
        pos_enc = torch.zeros(position, d_model)
        for pos in range(position):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        return pos_enc.unsqueeze(0)



# class GFlowNet_Z(nn.Module):
#     def __init__(self, d_model):
#         nn.Module.__init__(self)
#         self.to_flow = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             mup.MuReadout(d_model, 1, readout_zero_init=False),
#         )
#
#     def forward(self, x, pad_mask):
#         x = self.to_flow(x).squeeze(-1)
#         masked_x = (x.view(-1) * pad_mask.exp().view(-1)).view(x.size())
#         pooled_x = masked_x.sum(1)  # / pad_mask.exp().sum(dim=-1).view(-1)
#         return pooled_x


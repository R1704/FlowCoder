import torch
import torch.nn as nn
from src.sequential.config import *
import math


class FlowModel(nn.Module):
    def __init__(self,
                 tokenizer,
                 device,
                 d_model=512,
                 num_decoder_layers=6,
                 num_heads=8,
                 dropout_prob=0.1
                 ):
        super(FlowModel, self).__init__()

        # Define the vocabulary
        self.tokenizer = tokenizer

        # Define the device
        self.device = device

        # Masked softmax to rule out impossible actions
        self.masked_softmax = MaskedSoftmax(tokenizer, device)

        # Define the embedding layer
        self.d_model = d_model
        self.embeddings = nn.Embedding(len(tokenizer.vocab), d_model)

        # Define the positional encoding, call to staticmethod
        self.pos_enc = self.positional_encoding(max_trajectory, d_model).to(device) # FIXME: is the static method going to device :O

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

            # Convert output_tokens to tensor
            output_tokens_tensor = torch.tensor(output_tokens, dtype=torch.long).to(self.device)

            # Compute the softmax probabilities and log probabilities
            probs = self.masked_softmax(logits, output_tokens_tensor)
            log_probs_tensor = torch.log(probs)

            # Sample the next token from the logits
            next_token_idx = torch.multinomial(probs, num_samples=1).item()

            # Get the log probability of the sampled token
            log_prob = log_probs_tensor[0, next_token_idx].item()

            # Break the loop if the <STOP> token is generated
            if next_token_idx == self.tokenizer.token_to_idx['<STOP>']:
                break

            # Add the token index to the output tokens
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
                pos_enc[pos, i] = math.sin(pos / (10_000 ** ((2 * i) / d_model)))
                pos_enc[pos, i + 1] = math.cos(pos / (10_000 ** ((2 * (i + 1)) / d_model)))

        return pos_enc.unsqueeze(0)


class MaskGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.grammar = tokenizer.grammar

    def is_nonterminal(self, token):
        return token in self.grammar.nonterminals

    def generate(self, tokens):
        # Initialize mask with 0s
        mask = [0] * len(self.tokenizer.vocab)

        # Special handling for <START> and <STOP> tokens
        start_token_idx = self.tokenizer.token_to_idx['<START>']
        stop_token_idx = self.tokenizer.token_to_idx['<STOP>']

        # If the sequence is empty
        if tokens.numel() == 0:
            mask[start_token_idx] = 1
            return mask

        last_token = self.tokenizer.idx_to_token[tokens[-1].item()]
        if last_token == '<START>':
            for idx, token in enumerate(self.tokenizer.vocab):
                if token in self.grammar.terminals:
                    mask[self.tokenizer.token_to_idx[token]] = 1
        elif self.is_nonterminal(last_token):
            for idx, token in enumerate(self.tokenizer.vocab):
                if token in self.grammar.terminals:
                    mask[self.tokenizer.token_to_idx[token]] = 1
        else:
            for idx, token in enumerate(self.tokenizer.vocab):
                if self.is_nonterminal(token):
                    mask[self.tokenizer.token_to_idx[token]] = 1
            mask[stop_token_idx] = 1

        return mask


class MaskedSoftmax(nn.Module):
    def __init__(self, grammar, device):
        super(MaskedSoftmax, self).__init__()
        self.mask_generator = MaskGenerator(grammar)
        self.device = device

    def forward(self, logits, tokens):
        mask = self.mask_generator.generate(tokens)
        mask = torch.tensor(mask, device=self.device).float()  # Added .float() to ensure the mask is of the same type as logits
        max_val = torch.max(logits)
        logits -= max_val
        logits_exp = torch.exp(logits)
        logits_exp_masked = logits_exp * mask
        probs = logits_exp_masked / torch.sum(logits_exp_masked)

        return probs


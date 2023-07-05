import torch
from torch import nn
import math
import mup
from torch.nn.utils.rnn import pad_sequence


class GFlowNet(nn.Module):
    def __init__(self, device, cfg, d_model=512, num_heads=8, num_layers=2):
        super(GFlowNet, self).__init__()
        self.device = device
        self.cfg = cfg

        # Primitives
        self.primitives = list(set(p for _, P in cfg.rules.items() for p, _ in P.items()))
        self.primitive2idx = {p: i for i, p in enumerate(self.primitives)}
        self.primitive2idx['<start>'] = len(self.primitives)
        self.idx2primitive = {i: p for p, i in self.primitive2idx.items()}
        n_primitives = len(self.primitive2idx)
        self.primitives_embedding = nn.Embedding(num_embeddings=n_primitives, embedding_dim=d_model)

        self.positional_encoding = PositionalEncoding(d_model)

        io_embed_dim = 64
        shared_dim = d_model + io_embed_dim

        # Defining the transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=shared_dim, nhead=num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)

        # MLPs for logits and logZ
        self.forward_logits = nn.Sequential(
                    nn.LayerNorm(shared_dim),
                    nn.Linear(shared_dim, shared_dim),
                    nn.ReLU(),
                    nn.Linear(shared_dim, n_primitives)
                )

        self.logZ = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, 1)
        )


    def forward(self, state, S, task):

        # Process task (embed and encode and get latent representation)
        task = task[0]  # task is a vector of dim 64

        if len(state) == 0:
            state = self.primitives_embedding(torch.tensor([self.primitive2idx['<start>']], device=self.device))
        else:
            state = self.primitives_embedding(torch.tensor([self.primitive2idx[s] for s in state], device=self.device))
            state = self.positional_encoding(state)


        # Repeat task to have same sequence length as state
        task_repeated = task.unsqueeze(0).repeat(state.shape[0], 1)

        # Concatenate state and task along the feature dimension
        combined = torch.cat((state, task_repeated), dim=-1).to(self.device)

        # pass through the transformer
        transformer_output = self.transformer_encoder(combined)

        # predict the forward logits, backward logits, and total flow logZ
        forward_logits = self.forward_logits(transformer_output)[-1]
        logZ = self.logZ(transformer_output)[-1]

        # possible programs from the last derivation S
        candidate_programs = self.cfg.rules[S]

        mask = torch.tensor(
            [0 if p in list(candidate_programs) else 1 for p in list(self.primitive2idx)], device=self.device).bool()

        # apply mask on forward and backward logits
        # we use -100 since exp(-100) is tiny, but we don't want -inf (since we're predicting log-values)
        forward_logits.masked_fill_(mask, float(-100))
        # print('forward_logits.shape  ', forward_logits.shape, forward_logits)

        return forward_logits, logZ

    def ProgramEncoder(self, program):
        return program

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
        # pe = pe.unsqueeze(0).transpose(0, 1)  # i think this is for batching
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

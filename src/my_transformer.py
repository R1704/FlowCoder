import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_embed, n_heads):
        super(SelfAttention, self).__init__()
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads

        assert self.d_heads * n_heads == d_embed, "d_embed needs to be divisible by n_heads"

        self.values = nn.Linear(self.d_heads, self.d_heads, bias=False)
        self.keys = nn.Linear(self.d_heads, self.d_heads, bias=False)
        self.query = nn.Linear(self.d_heads, self.d_heads, bias=False)

        self.fc_out = nn.Linear(n_heads * self.d_heads, d_embed)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, keys_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split emedding into self.n_heads pieces
        values = values.reshape(N, value_len, self.n_heads, self.d_heads)
        keys = keys.reshape(N, keys_len, self.n_heads, self.d_heads)
        query = query.reshape(N, query_len, self.n_heads, self.d_heads)

        values = self.values(values)
        keys = self.keys(keys)
        query = self.query(query)

        # query shape: (N, query_len, n_heads, d_heads)
        # keys shape: (N, keys_len, n_heads, d_heads)
        # energy shape: (N, n_heads, query_len, keys_len)
        energy = torch.einsum('nqhd, nkhd -> nhqk', [query, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy / (self.d_embed ** (1 / 2)), dim=3)

        # attention shape: (N, n_heads, query_len, keys_len)
        # values shape: (N, value_len, n_heads, d_heads)
        # out shape after einsum: (N, query_len, n_heads, d_heads), then flatten last two dimensions
        out = torch.einsum('nhqk, nvhd -> nqhd', [attention, values]).reshape(N, query_len, self.n_heads * self.d_heads)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_embed, n_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(d_embed, n_heads)
        self.norm1 = nn.LayerNorm(d_embed)
        self.norm2 = nn.LayerNorm(d_embed)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_embed, forward_expansion * d_embed),
            nn.ReLU(),
            nn.Linear(forward_expansion * d_embed, d_embed)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_embed, n_layers, n_heads, forward_expansion, dropout, max_length, device):
        super(Encoder, self).__init__()
        self.d_embed = d_embed
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, d_embed)
        self.position_embedding = nn.Embedding(max_length, d_embed)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_embed, n_heads, dropout, forward_expansion) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, d_embed, n_heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(d_embed, n_heads)
        self.norm = nn.LayerNorm(d_embed)
        self.transformer_block = TransformerBlock(d_embed, n_heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, d_embed, n_layers, n_heads, forward_expansion, dropout, max_len, device):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, d_embed)
        self.position_embedding = nn.Embedding(max_len, d_embed)

        self.layers = nn.ModuleList(
            [DecoderBlock(d_embed, n_heads, forward_expansion, dropout, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(d_embed, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)  # added by Ron
        self.logZ = nn.Parameter(torch.ones(1))  # added by Ron

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        out = self.softmax(out)  # added by Ron
        return out


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 n_layers=6,
                 forward_expansion=4,
                 n_heads=8,
                 dropout=0,
                 device='cuda',
                 max_len=100
                 ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, n_layers, n_heads, forward_expansion, dropout, max_len,
                               device)
        self.decoder = Decoder(trg_vocab_size, embed_size, n_layers, n_heads, forward_expansion, dropout, max_len,
                               device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # (N, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = ['2', '3', '+', '-']

    vocab2index = {k: i for i, k in enumerate(vocab)}
    index2vocab = {i: k for i, k in enumerate(vocab)}
    seq = []
    x = torch.tensor([vocab2index['3']]).unsqueeze(0).to(device)
    trg = torch.tensor([vocab2index['3']]).unsqueeze(0).to(device)

    # x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    # trg = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    # x = torch.tensor([[5]])
    # trg = torch.tensor([[5]])
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = len(vocab)
    trg_vocab_size = len(vocab)

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg[:, :-1])
    print(out)


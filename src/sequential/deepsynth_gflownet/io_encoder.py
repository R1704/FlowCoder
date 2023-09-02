import torch
import torch.nn as nn


class IOEncoder(nn.Module):
    """
    nb_arguments_max: maximum number of inputs
    size_max: maximum number of elements in a list
    lexicon: list of symbols that can appear (for instance range(-10,10))
    """
    def __init__(self, n_examples_max, size_max, lexicon, d_model=512, device='cpu'):
        super(IOEncoder, self).__init__()
        self.n_examples_max = n_examples_max
        self.size_max = size_max
        self.lexicon = ['PAD', 'IN', 'OUT'] + lexicon[:]
        self.symbol2idx = {s: i for i, s in enumerate(self.lexicon)}
        self.io_embedding = nn.Embedding(num_embeddings=len(self.lexicon), embedding_dim=d_model).to(device)
        self.device = device

    def encode_IO(self, IO):
        '''
        IO comes in the form [[I1,I2, ..., Ik], O]
        We concatenate all IOs into one list with special IN and OUT tokens, and PAD up to the longest example
        '''
        encoded = []
        for inp, out in IO:
            encoded += [self.symbol2idx['IN']] + [self.symbol2idx[i] for i in inp[0]] + [self.symbol2idx['OUT']] + [self.symbol2idx[o] for o in out]

        # Pad until the maximum allowed size
        encoded += [self.symbol2idx['PAD']] * (self.n_examples_max * self.size_max * 2 - len(encoded))
        return torch.tensor(encoded, device=self.device)

    def forward(self, IOs):
        ios = []
        max_len = 0

        # First, encode all IO pairs and find the max length
        for IO in IOs:
            encoded = self.encode_IO(IO)
            max_len = max(max_len, len(encoded))
            ios.append(encoded)

        # Then pad all sequences to the maximum length
        for i in range(len(ios)):
            ios[i] = torch.cat(
                [ios[i], torch.full((max_len - len(ios[i]),), self.symbol2idx['PAD'], dtype=torch.long, device=self.device)])

        ios = torch.stack(ios)
        ios = self.io_embedding(ios)
        ios = ios.transpose(0, 1)
        return ios

    def attention_mask(self, IOs_encoded):
        '''
        creates attention mask for encoded IOs

        attention mask is a tensor with the same size as IOs_encoded
        where positions with 'PAD' token are 0 and all others are 1
        '''
        return (IOs_encoded != self.symbol2idx['PAD']).long().to(self.device)

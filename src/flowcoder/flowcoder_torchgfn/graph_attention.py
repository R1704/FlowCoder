import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        Wh = self.W(h)  # (N, out_features)
        N = Wh.size(0)
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1),
                             Wh.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        e = self.leakyrelu(self.a(a_input).squeeze(2))
        attention = F.softmax(e.masked_fill(adj == 0, float('-inf')), dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GraphAttentionNetwork(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0.6, alpha=0.2, nheads=1):
        super().__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
                                         for _ in range(nheads)])
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x

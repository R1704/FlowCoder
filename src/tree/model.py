import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, num_nodes, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model, nhead, num_layers, dim_feedforward=2 * d_model
        )
        self.output_next_node = nn.Linear(d_model, 1)
        self.output_node_type = nn.Linear(d_model, num_nodes)

    def forward(self, x):
        x = self.transformer(x)
        next_node = self.output_next_node(x)
        node_type = self.output_node_type(x)
        return next_node, node_type
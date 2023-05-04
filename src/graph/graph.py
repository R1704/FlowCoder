# class Graph:
#     def __init__(self):
#         self.nodes = []
#         self.edges = []
#
#     def add_node(self, node):
#         self.nodes.append(node)
#
#     def add_edge(self, parent, child):
#         self.edges.append((parent, child))
#
#     def __repr__(self):
#         return f'Nodes: {self.nodes}\nEdges: {self.edges}'
#
#     def ast_to_graph(self, ast_node):
#         def _traverse_and_build_graph(node):
#             self.add_node(node)
#
#             for child in node.children.values():
#                 self.add_edge(node, child)
#                 _traverse_and_build_graph(child)
#
#         _traverse_and_build_graph(ast_node)
#         return self


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

from src.tree.AST import Node
from src.tree.utils import *
from src.tree.grammar import *


def node_to_one_hot(node):
    mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
               '+': 10, '-': 11, '*': 12, '/': 13}
    one_hot = [0] * len(mapping)
    one_hot[mapping[node.symbol]] = 1
    return one_hot


def ast_to_graph_data(ast):
    nodes = []
    edges = []
    edge_indices = []

    def traverse_tree(node):
        nodes.append(node_to_one_hot(node))
        current_idx = len(nodes) - 1

        for child_idx, child in node.children.items():
            edges.append((current_idx, len(nodes)))
            edge_indices.append(child_idx)
            traverse_tree(child)

    traverse_tree(ast)
    nodes = torch.tensor(nodes, dtype=torch.float)
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_indices = torch.tensor(edge_indices, dtype=torch.float)

    return Data(x=nodes, edge_index=edges, edge_attr=edge_indices)


class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        weight = self.lin(edge_attr)
        return x_j * weight

    def update(self, aggr_out):
        return F.relu(aggr_out)


class ASTGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(ASTGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MPNNLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(MPNNLayer(hidden_channels, hidden_channels))
        self.output_layer = nn.Linear(hidden_channels, in_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return self.output_layer(x)




ast = Node.create_tree(primitives, depth=3)
ast.print()
print(ast.evaluate())

in_channels = 14  # The number of unique symbols in the primitives
hidden_channels = 64
num_layers = 3

model = ASTGNN(in_channels, hidden_channels, num_layers)
data = ast_to_graph_data(ast)
print(data)
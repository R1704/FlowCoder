# import torch
# import math
#
#
# class PositionalEncoding:
#     def __init__(self, num_nodes, embedding_dim, primitives):
#         self.num_nodes = num_nodes
#         self.embedding_dim = embedding_dim // 2
#         self.primitives = primitives
#
#     def get_node_index_to_position_mapping(self):
#         node_index_to_position_mapping = {}
#         for idx, node in enumerate(self.primitives):
#             node_index_to_position_mapping[node.idx] = idx
#         return node_index_to_position_mapping
#
#     def encode(self, tree):
#         self.positional_encoding = []
#         self._traverse(tree, 0, 0)
#         num_nodes = len(self.positional_encoding)
#
#         max_depth, max_sibling_order = max(self.positional_encoding, key=lambda x: (x[0], x[1]))
#         depth_encoding, sibling_order_encoding = self._custom_encoding(max_depth + 1, max_sibling_order + 1)
#         combined_encoding = torch.zeros((self.num_nodes, self.embedding_dim * 2))
#
#         for idx, (depth, sibling_order) in enumerate(self.positional_encoding):
#             combined_encoding[idx, :self.embedding_dim] = depth_encoding[depth % len(depth_encoding)]
#             combined_encoding[idx, self.embedding_dim:] = sibling_order_encoding[
#                 sibling_order % len(sibling_order_encoding)]
#
#         return combined_encoding
#
#     def _traverse(self, node, depth, sibling_order):
#         self.positional_encoding.append((depth, sibling_order))
#         if not node.is_terminal:
#             for idx, child in enumerate(node.children.values()):
#                 self._traverse(child, depth + 1, idx)
#
#     def _custom_encoding(self, max_depth, max_sibling_order):
#         position_depth = torch.arange(0, max_depth + 1).unsqueeze(1).float()
#         position_sibling_order = torch.arange(0, max_sibling_order + 1).unsqueeze(1).float()
#
#         div_term_depth = torch.exp(
#             torch.arange(0, self.embedding_dim, 1).float() * -(math.log(10000.0) / self.embedding_dim))
#         div_term_sibling_order = torch.exp(
#             torch.arange(0, self.embedding_dim, 1).float() * -(math.log(10000.0) / self.embedding_dim) * 2)
#
#         depth_encoding = (position_depth * div_term_depth).sin() + (position_depth * div_term_depth).cos()
#         sibling_order_encoding = (position_sibling_order * div_term_sibling_order).sin() + (
#                 position_sibling_order * div_term_sibling_order).cos()
#
#         return depth_encoding, sibling_order_encoding


import numpy as np


import numpy as np

class PositionalEncoding:
    def __init__(self, max_depth, max_arity, d_model):
        self.max_depth = max_depth
        self.max_arity = max_arity
        self.d_model = d_model

    def encode_position(self, node) -> dict:
        position_encoding = {}

        def encode_recursive(node):
            position = np.zeros(self.d_model)
            div_term = np.exp(np.arange(0, self.d_model // 6 * 2, 1) * -(np.log(10000.0) / (self.d_model // 3)))

            depth = node.depth
            child_idx = node.child_idx if node.child_idx is not None else 0
            arity = node.parent.arity if node.parent is not None else -1

            depth_enc = np.zeros(self.d_model // 3)
            child_idx_enc = np.zeros(self.d_model // 3)
            arity_enc = np.zeros(self.d_model // 3)

            depth_enc[0::2] = np.sin(depth * div_term)
            depth_enc[1::2] = np.cos(depth * div_term)

            child_idx_enc[0::2] = np.sin(child_idx * div_term)
            child_idx_enc[1::2] = np.cos(child_idx * div_term)

            arity_enc[0::2] = np.sin(arity * div_term)
            arity_enc[1::2] = np.cos(arity * div_term)

            position = np.concatenate((depth_enc, child_idx_enc, arity_enc))

            position_encoding[node] = position

            for child in node.children.values():
                encode_recursive(child)

        encode_recursive(node)
        return position_encoding

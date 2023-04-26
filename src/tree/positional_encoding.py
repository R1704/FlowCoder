import torch


class PositionalEncoding:
    def __init__(self, tree, num_nodes):
        self.tree = tree
        self.num_nodes = num_nodes
        self.positional_encoding = []

    def encode(self):
        self._traverse(self.tree, 0, 0)
        return torch.tensor(self.positional_encoding, dtype=torch.float32)

    def _traverse(self, node, depth, sibling_order):
        self.positional_encoding.append((depth, sibling_order))
        if not node.is_terminal:
            for idx, child in enumerate(node.children.values()):
                self._traverse(child, depth + 1, idx)

from AST import Node
import random
import torch


def prepare_train_data(tree, primitives):
    """
    Prepare training data for the transformer model.
    :param tree: root node of the tree
    :param primitives: list of primitives
    :return: list of tuples (parent, child)
    """
    train_data = []

    def _traverse(node):
        if node.is_terminal:
            return
        for child in node.children.values():
            train_data.append((primitives.index(node), primitives.index(child)))
            _traverse(child)

    _traverse(tree)
    return train_data


def encode_tree(tree, primitives):
    indices = []
    positions = []

    def _traverse(node, depth):
        indices.append(primitives.index(node))
        positions.append(depth)
        if node.is_terminal:
            return
        for child in node.children.values():
            _traverse(child, depth + 1)

    _traverse(tree, 0)
    return torch.tensor(indices, dtype=torch.long), torch.tensor(positions, dtype=torch.float32)

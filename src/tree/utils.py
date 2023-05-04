import torch


def prepare_train_data(tree, primitives):
    train_data = []
    input_indices, positions = encode_tree(tree, primitives)
    for i in range(len(input_indices) - 1):
        train_data.append((input_indices[i], input_indices[i + 1]))
    if len(train_data) == 0:
        train_data.append((input_indices[0], input_indices[0]))
    return train_data


def encode_tree(tree, primitives):
    """
    Encode a tree into a sequence of (index, depth) tuples.
    """
    indices = []
    positions = []

    # depth first traversal
    def _traverse(node, depth):
        # add index of node to indices
        indices.append(primitives.index(node))
        # add depth of node to positions
        positions.append(depth)
        if node.is_terminal:
            return
        # traverse children recursively
        for child in node.children.values():
            _traverse(child, depth + 1)

    _traverse(tree, 0)
    return torch.tensor(indices, dtype=torch.long), torch.tensor(positions, dtype=torch.float32)

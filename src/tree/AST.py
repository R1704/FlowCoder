import random
from PrettyPrint import PrettyPrintTree

# https://github.com/AharonSambol/PrettyPrintTree
pt = PrettyPrintTree(lambda x: x.children.values(), lambda x: x.symbol)


class Node:

    def __init__(self, symbol=None, parent=None, arity: int = 0, child_idx: int = None,
                 function: callable = None) -> None:
        self.symbol = symbol
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.child_idx = child_idx
        self.arity = arity
        self.function = function
        self.children = {}

    def attach(self, node, child_idx):
        assert child_idx < self.arity, \
            f'Cannot attach a child to a nonterminal with arity {self.arity} at index {child_idx}.'
        assert child_idx not in self.children, 'child already filled.'
        self.children[child_idx] = node
        node.parent = self
        node.child_idx = child_idx
        node.depth = self.depth + 1

    def evaluate(self):
        def evaluate_recursive(node):
            if node.is_terminal:
                return node.symbol
            else:
                args = [evaluate_recursive(child) for child in node.children.values()]
                if node.function is not None:
                    return node.function(*args)
                else:
                    return args[0]

        return evaluate_recursive(self)

    def get_free_children(self):
        return list(set(range(self.arity)) - set(self.children.keys()))

    @property
    def is_root(self):
        return self.parent is None

    @property
    def is_terminal(self) -> bool:
        return self.arity == 0

    @property
    def is_nonterminal(self):
        return self.arity > 0

    @property
    def is_full(self):
        return all(child is not None for child in self.children.values())

    @property
    def is_complete(self):
        return self.is_full and all(child.is_complete for child in self.children.values())

    def __eq__(self, other):
        return self.symbol == other.symbol and self.arity == other.arity and self.function == other.function

    def __repr__(self):
        return f'{self.symbol}'

    def print(self):
        pt(self)

    @staticmethod
    def create_tree(primitives, depth):
        """
        Create a random tree of a given depth using the given primitives.
        :param primitives: list of primitives
        :param depth: depth of the tree
        :return: root node which recursively contains the tree
        """

        def _create_tree(current_depth, parent, child_idx):
            if current_depth == depth:
                terminal_nodes = [node for node in primitives if node.is_terminal]
                selected_node = random.choice(terminal_nodes)
                return Node(symbol=selected_node.symbol,
                            parent=parent,
                            arity=0,
                            child_idx=child_idx,
                            function=None)
            else:
                nonterminal_nodes = [node for node in primitives if node.is_nonterminal]
                selected_node = random.choice(nonterminal_nodes)
                node = Node(symbol=selected_node.symbol,
                            parent=parent,
                            arity=selected_node.arity,
                            child_idx=child_idx,
                            function=selected_node.function)
                for i in range(selected_node.arity):
                    child_node = _create_tree(current_depth + 1, node, i)
                    node.attach(child_node, i)
                return node

        root = _create_tree(0, None, None)
        return root

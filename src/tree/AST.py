import random
import numpy as np
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

    def evaluate(self) -> any:
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

    def get_free_children(self) -> list:
        return list(set(range(self.arity)) - set(self.children.keys()))

    def count_nodes(self) -> int:
        count = 1
        if not self.is_terminal:
            for child in self.children.values():
                count += child.count_nodes()
        return count

    def traverse(self, node_list):
        node_list.append(self)
        for child in self.children.values():
            child.traverse(node_list)
        return node_list

    @classmethod
    def extend_tree(cls, current_tree, primitives, max_depth=5) -> 'Node':
        if current_tree is None or current_tree.is_terminal:
            return current_tree

        if current_tree.depth < max_depth:
            for i, child in enumerate(current_tree.children.values()):
                if child.is_terminal or child.depth < max_depth:
                    extended_child = cls.extend_tree(child, primitives, max_depth)
                    if extended_child is not None:
                        current_tree.children[i] = extended_child
                        break
        return current_tree

    @staticmethod
    def create_tree(primitives, depth) -> 'Node':
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

    @staticmethod
    def get_random_root(primitives):
        nonterminal_nodes = [node for node in primitives if node.is_nonterminal]
        selected_node = random.choice(nonterminal_nodes)
        return Node(symbol=selected_node.symbol,
                    parent=None,
                    arity=selected_node.arity,
                    child_idx=None,
                    function=selected_node.function)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_terminal(self) -> bool:
        return self.arity == 0

    @property
    def is_nonterminal(self) -> bool:
        return self.arity > 0

    @property
    def is_full(self) -> bool:
        return all(child is not None for child in self.children.values())

    @property
    def is_complete(self) -> bool:
        return self.is_full and all(child.is_complete for child in self.children.values())

    def is_equal_tree(self, other) -> bool:
        if self != other:
            return False

        if len(self.children) != len(other.children):
            return False

        for key in self.children.keys():
            if key not in other.children:
                return False

            self_child = self.children[key]
            other_child = other.children[key]

            if not self_child.is_equal_tree(other_child):
                return False

        return True

    def __eq__(self, other) -> bool:
        return self.symbol == other.symbol and self.arity == other.arity and self.function == other.function

    def __repr__(self) -> str:
        return f'{self.symbol}'

    def __hash__(self):
        return hash(self.symbol)

    def print(self):
        pt(self)

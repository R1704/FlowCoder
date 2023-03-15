import random
from typing import *
import operator


class Symbol:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def __str__(self):
        return self.symbol


class Terminal(Symbol):
    def __init__(self, symbol: str):
        super().__init__(symbol)


class Nonterminal(Symbol):
    def __init__(self, symbol: str, arity: int):
        super().__init__(symbol)
        self.arity = arity


class Node:
    def __init__(self, parent=None, depth=0, max_depth=3, idx=0) -> None:
        # self.terminals = [Terminal(x) for x in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        self.terminals = [Terminal(x) for x in ['2', '3', '5', '7']]
        self.nonterminals = [Nonterminal(*x) for x in [('+', 2), ('-', 2), ('*', 2)]]

        self.parent = parent
        self.depth = depth
        self.max_depth = max_depth
        self.idx = idx
        self.value = self.sample_value()
        self.children = self.make_children()

        self.ops: dict = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul
        }


class Root(Node):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return str(self.value).join([f'{c.__str__()}' if isinstance(c, Leaf) else f'({c.__str__()})' for c in
                                     self.children]) + f' = {self.evaluate()}'

    def sample_value(self) -> Symbol:
        return random.sample(self.nonterminals, 1)[0]

    def make_children(self) -> List[Node]:
        if self.depth < self.max_depth:
            args = (self, self.depth + 1)
            return [Branch(*args, self.idx + i + 1) if random.random() < 0.5 else Leaf(*args, self.idx + i + 1) for i in
                    range(self.value.arity)]

    def evaluate(self):
        e = [c.evaluate() for c in self.children]
        return self.ops[self.value.symbol](*e)


class Branch(Node):
    def __init__(self, parent, depth, idx) -> None:
        super().__init__(parent=parent, depth=depth, idx=idx)

    def __str__(self) -> str:
        return str(self.value).join(
            [f'{c.__str__()}' if isinstance(c, Leaf) else f'({c.__str__()})' for c in self.children])

    def sample_value(self) -> Symbol:
        return random.sample(self.nonterminals, 1)[0]

    def make_children(self) -> List[Node]:
        args = (self, self.depth + 1)
        if self.depth < self.max_depth:
            return [Branch(*args, self.idx + i + 1) if random.random() < 0.5 else Leaf(*args, self.idx + i + 1) for i in
                    range(self.value.arity)]
        else:
            return [Leaf(*args, self.idx + i + 1) for i in range(self.value.arity)]

    def evaluate(self):
        e = [c.evaluate() for c in self.children]
        return self.ops[self.value.symbol](*e)


class Leaf(Node):
    def __init__(self, parent, depth, idx) -> None:
        super().__init__(parent=parent, depth=depth, idx=idx)

    def __str__(self) -> str:
        return self.value

    def sample_value(self) -> Symbol:
        return random.sample(self.terminals, 1)[0]

    def make_children(self) -> List[Node]:
        ...

    def evaluate(self):
        return int(self.value.symbol)


root = Root()
print(root)
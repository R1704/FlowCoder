from src.env import *


class Symbol:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def __repr__(self):
        return self.symbol


# can be literal or variable?
class Terminal(Symbol):
    def __init__(self, symbol: str):
        super().__init__(symbol)


class Nonterminal(Symbol):
    def __init__(self, symbol: str, operator: callable, arity: int):
        super().__init__(symbol)
        self.arity = arity
        self.operator = operator


class Grammar:
    def __init__(self):


        # A dictionary that turns string representations into actual operators
        self.ops: dict = {}

        self.terminals = []
        self.nonterminals = []
        self.vocab = []

        self.max_arity = 0

    def add_terminals(self, terminals: list[Terminal]):
        for terminal in terminals:
            self.terminals.append(terminal)
            self.vocab.append(terminal)

    def add_nonterminals(self, nonterminals: list[Nonterminal]):
        for nonterminal in nonterminals:
            self.ops[nonterminal.symbol] = nonterminal.operator
            self.nonterminals.append(nonterminal)
            self.vocab.append(nonterminal)
            if nonterminal.arity > self.max_arity:
                self.max_arity = nonterminal.arity




grammar = Grammar(Primes(100))
terminals = [Terminal(x) for x in ['2', '3', '5', '7']]
grammar.add_terminals(terminals)
nonterminals = [Nonterminal(*x) for x in [
    ('+', lambda x, y: x + y, 2),
    ('-', lambda x, y: x - y, 2),
    ('*', lambda x, y: x * y, 2)]]
grammar.add_nonterminals(nonterminals)
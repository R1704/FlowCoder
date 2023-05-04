import re
from src.env import Environment
from config import *

class Grammar:
    def __init__(self, env: Environment):

        self.env = env

        # A dictionary that turns string representations into actual operators
        self.ops: dict = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y
        }

        self.terminals = ['2', '3']
        self.nonterminals = list(self.ops.keys())
        self.primitives = self.terminals + self.nonterminals

    def add_terminal(self, terminal: str):
        self.terminals.append(terminal)
        self.primitives.append(terminal)

    def add_nonterminal(self, nonterminal: str):
        self.nonterminals.append(nonterminal)
        self.primitives.append(nonterminal)

    def valid_function(self, func: list) -> bool:

        if '<START>' in func:
            return False

        if len(func) < 3:
            return False

        if not any([x in self.nonterminals for x in func]):
            return False

        # no two symbols of the same type should be adjacent
        for i in range(len(func) - 1):
            if func[i] in self.nonterminals and func[i + 1] in self.nonterminals or \
                    func[i] in self.terminals and func[i + 1] in self.terminals:
                return False

        # the string should start and end with a numeral
        if func[0] in self.nonterminals or func[-1] in self.nonterminals:
            return False
        return True

    # Evaluates a binary expression
    def eval_binary_expr(self, op1, oper, op2) -> int:
        op1, op2 = int(op1), int(op2)
        return self.ops[oper](op1, op2)

    # Evaluates a whole expression
    def evaluate(self, func: list) -> int:
        while len(func) >= 3:
            chunk = func[:3]
            eval_chunk = self.eval_binary_expr(*chunk)
            func = [str(eval_chunk)] + func[3:]
        return int(func[0])

    def reward(self, func: list) -> int:
        assert func[0] == '<START>', 'The function should start with <START>'
        func = func[1:]

        if func[-1] == '<STOP>':
            func = func[:-1].copy()

        # We want to check whether the created function is well-defined
        if not self.valid_function(func):
            # print('not a valid function')
            return 0

        # evaluate function
        result: int = self.evaluate(func)

        # if it is not a prime, we don't want it, but we give a small reward for being well-defined
        if result not in self.env.primes:
            # print('not a prime')
            return 1

        # we want a short function to avoid redundancy
        else:
            return max(3, 2 * max_trajectory - len(func) - 1)
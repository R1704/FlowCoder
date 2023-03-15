import operator
import re
from env import Environment


class Grammar:
    def __init__(self, env: Environment):

        self.env = env

        # A dictionary that turns string representations into actual operators
        self.ops: dict = {
            '+': operator.add,
            '-': operator.sub
        }

        # TODO: add variables (x, placeholders) we can also define functions and higher order functions, lambdas, etc.)
        self.terminals = ['2', '3']
        self.nonterminals = list(self.ops.keys())
        self.primitives = self.terminals + self.nonterminals

    def add_terminal(self, terminal: str):
        self.terminals.append(terminal)
        self.primitives.append(terminal)

    def add_nonterminal(self, nonterminal: str):
        self.nonterminals.append(nonterminal)
        self.primitives.append(nonterminal)

    def valid_function(self, func: str) -> bool:
        func = list(func)
        if len(func) < 3:
            return False

        if not any([x in self.nonterminals for x in func]):
            return False

        # no two symbols of the same type should be adjacent TODO: only at first, because we could have n-digit numerals
        for i in range(len(func) - 1):
            if func[i] in self.terminals and func[i + 1] in self.terminals \
                    or func[i] in self.nonterminals and func[i + 1] in self.nonterminals:
                return False
        # the string should start and end with a numeral
        if func[0] in self.nonterminals or func[-1] in self.nonterminals:
            return False
        return True

    # Evaluates a binary expression
    def eval_binary_expr(self, op1, oper, op2):
        op1, op2 = int(op1), int(op2)
        return self.ops[oper](op1, op2)

    # Evaluates a whole expression
    def evaluate(self, s: str) -> int:
        s = re.split(r'(\D)', s)
        while len(s) >= 3:
            chunk = s[:3]
            eval_chunk = self.eval_binary_expr(*chunk)
            s = s[3:]
            s.insert(0, str(eval_chunk))
        return int(s[0])


    def reward(self, func: str) -> int:

        # We want to check whether the created function is well-defined
        if not self.valid_function(func):
            # print('not a valid function')
            return 0

        # evaluate function
        result: int = self.evaluate(func)

        # if it is not a prime, we don't want it
        if result not in self.primes:
            # print('not a prime')
            return 1

        # we want a short function to avoid redundancy
        else:
            return 10 // len(func)

import pickle
from src.env import Environment
from src.sequential.config import *
import re


class Grammar:
    def __init__(self, env: Environment):

        self.env = env

        # A dictionary that turns string representations into actual operators
        self.ops: dict = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
        }

        self.terminals: set = {'2', '3'}
        self.nonterminals: set = {k for k in self.ops.keys()}
        self.primitives: set = self.terminals | self.nonterminals

    def add_terminal(self, terminal: str):
        self.terminals.update([terminal])
        self.primitives.update([terminal])

    def add_nonterminal(self, nonterminal: str):
        self.nonterminals.update([nonterminal])
        self.primitives.update([nonterminal])

    def valid_function(self, func: list) -> bool:

        # no additional starting tokens should be in the sequence
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

    def reward(self, func: list) -> int:
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
            print(func)
            # return max(3, 2 * max_trajectory - len(func))
            return 30

    def parse(self, func: list):
        stack = []
        output = []

        # Helper function to check if a token is an operator
        def is_operator(token):
            return token in self.ops

        # Helper function to check if a token is an operand (a number)
        def is_operand(token):
            return re.match(r'^\d+$', token)

        # Helper function to get the precedence of an operator
        def precedence(operator):
            if operator in {'+', '-'}:
                return 1
            elif operator in {'*', '/'}:
                return 2
            return 0

        for token in func:
            if is_operand(token):
                output.append(token)
            elif is_operator(token):
                while stack and is_operator(stack[-1]) and precedence(stack[-1]) >= precedence(token):
                    output.append(stack.pop())
                stack.append(token)

        while stack:
            output.append(stack.pop())

        return output

    def evaluate(self, func: list) -> int:
        parsed_func = self.parse(func)

        stack = []
        for token in parsed_func:
            if token.isdigit():
                stack.append(int(token))
            elif token in self.ops:
                op2 = stack.pop()
                op1 = stack.pop()
                result = self.ops[token](op1, op2)
                stack.append(result)

        return stack[0]

    def load_grammar(self):
        # Load the grammar from file if it exists
        if os.path.exists(grammar_path):
            with open(grammar_path, 'rb') as f:
                if f.peek():
                    terminals = pickle.load(f)
                    print('Loaded grammar from file')
                    return terminals

    @staticmethod
    def reset_grammar():
        # delete the grammar file
        if os.path.exists(grammar_path):
            os.remove(grammar_path)

add = lambda x, y: x + y
sub = lambda x, y: x - y
mul = lambda x, y: x * y
div = lambda x, y: x / y
functions: dict = {'+': add, '-': sub, '*': mul, '/': div}

# print(functions['+'](1, 2))

triple = lambda x, y, z: x + y + z
# print(triple(1, 2, 3))

curr = lambda x, y: x(y)
# print(curr(lambda x: x + 1, 2))

loop = lambda f, x: list(map(f, x))

# loop = lambda f, n: [f for _ in range(n)]

from src.tree.AST import Node

from src.tree.utils import *

primitives = [
    *[Node(symbol=i) for i in range(10)],
    Node(symbol='+', arity=2, function=lambda x, y: x + y),
    Node(symbol='-', arity=2, function=lambda x, y: x - y),
    Node(symbol='*', arity=2, function=lambda x, y: x * y),
    # Node(symbol='/', arity=2, function=lambda x, y: x / y),
]

max_depth = 5
d_model = 64
ast = Node.create_tree(primitives, depth=max_depth)
ast.print()
print(ast.evaluate())

from src.tree.positional_encoding import *
position_encoder = PositionalEncoding(d_model=d_model, max_depth=max_depth, max_arity=2)

position_encoding = position_encoder.encode_position(ast)

for node, encoding in position_encoding.items():
    print(node, encoding)


# encoded = encode_tree(tree, primitives)
# print(encoded)

# token_1 = Node(symbol='+', arity=2, function=lambda x, y: x + y)
# token_1.attach(Node(symbol='-', arity=2, function=lambda x, y: x - y), 0)
# token_1.attach(Node(symbol='+', arity=2, function=lambda x, y: x + y), 1)
#
# token_2 = Node(symbol='-', arity=2, function=lambda x, y: x - y)
# token_2.attach(Node(symbol='-', arity=2, function=lambda x, y: x - y), 0)
# token_2.attach(Node(symbol='+', arity=2, function=lambda x, y: x + y), 1)
#
# # token_1.print()
# # token_2.print()
# tokens = primitives + [token_1, token_2]
#
#
# tree = Node(symbol='+', arity=2, function=lambda x, y: x + y)
# tree.attach(Node(symbol='-', arity=2, function=lambda x, y: x - y), 0)
# tree.attach(Node(symbol='+', arity=2, function=lambda x, y: x + y), 1)
# tree.children[0].attach(Node(symbol='-', arity=2, function=lambda x, y: x - y), 0)
# tree.children[0].attach(Node(symbol='+', arity=2, function=lambda x, y: x + y), 1)
# # tree.print()
#
# from src.tree.ASTTokenizer import ASTTokenizer
# tokenizer = ASTTokenizer(tokens)
# tokenized_tree = tokenizer.tokenize(tree)
#
# for t in tokenized_tree:
#     t.print()
#     print('\n')
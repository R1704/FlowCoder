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

primitives = [
    *[Node(symbol=i) for i in range(10)],
    Node(symbol='+', arity=2, function=lambda x, y: x + y),
    Node(symbol='-', arity=2, function=lambda x, y: x - y),
    Node(symbol='*', arity=2, function=lambda x, y: x * y),
    Node(symbol='/', arity=2, function=lambda x, y: x / y),
]

tree = Node.create_tree(primitives, depth=5)
tree.print()
print(tree.evaluate())

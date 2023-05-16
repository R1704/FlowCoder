import numpy as np


class DSL:
    ...

class Arithmetic(DSL):
    def __init__(self):
        ...

    functions = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        'div': lambda x, y: x / y,
        'pow': lambda x, y: x ** y,
        'log': lambda x, y: np.log(x) / np.log(y),
        'sqrt': lambda x: np.sqrt(x),
        'sin': lambda x: np.sin(x),
        'cos': lambda x: np.cos(x),
        'tan': lambda x: np.tan(x),
        'exp': lambda x: np.exp(x),
        'neg': lambda x: -x,
        'abs': lambda x: np.abs(x),
        'max': lambda x, y: np.maximum(x, y),
        'min': lambda x, y: np.minimum(x, y),
        'eq': lambda x, y: x == y,
        'neq': lambda x, y: x != y,
        'gt': lambda x, y: x > y,
        'lt': lambda x, y: x < y,
        'geq': lambda x, y: x >= y,
        'leq': lambda x, y: x <= y,
        'and_': lambda x, y: x and y,
        'or_': lambda x, y: x or y,
        'not_': lambda x: not x,
        'if_': lambda x, y, z: y if x else z
    }

art = Arithmetic()
print(art.functions.items())
art.functions['foo'] = lambda x: art.functions['add'](art.functions['sub'], x)
print(art.functions['foo'](1))



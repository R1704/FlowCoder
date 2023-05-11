from grammar import *
from src.env import *


env = Primes(100)
grammar = Grammar(env)
func = ['2', '+', '11', '-', '2', '*', '5']
print(grammar.reward(func))
print(grammar.parse(func))
print(grammar.evaluate(func))
import random

import matplotlib.pyplot as plt
import sympy as sp


class Environment:

    def plot_rewards(self):
        pass

    def get_task(self, n: int):
        pass


class Primes(Environment):
    def __init__(self, n: int = 100):
        super().__init__()
        self.n = n
        self.xs = range(n)
        self.primes = [*sp.primerange(0, n)]
        self.primes_binary = [1 if x in self.primes else 0 for x in self.xs]

    def plot_rewards(self):
        figsize = (14, 7)
        _, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=figsize)
        ax.plot(self.xs, self.primes_binary)
        plt.show()

    def get_task(self, n: int = 1):
        return random.sample(self.primes, n)

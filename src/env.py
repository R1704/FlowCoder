import operator
import matplotlib.pyplot as plt
import sympy as sp


class Environment:
    def __init__(self):
        pass

    def plot_rewards(self):
        pass


class Primes(Environment):
    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.xs = range(N)
        self.primes = [*sp.primerange(0, N)]
        self.primes_binary = [1 if x in self.primes else 0 for x in self.xs]

    def plot_rewards(self):
        figsize = (14, 7)
        _, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=figsize)
        ax.plot(self.xs, self.primes_binary)
        plt.show()

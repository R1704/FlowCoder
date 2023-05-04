from src.sequential.train import Training
from src.env import Primes
from config import *

env = Primes(100)
training = Training(env, num_rounds, epochs, batch_size, max_trajectory, min_trajectory, device, display=display)
updated_grammar = training.train()

print("Updated grammar:", updated_grammar.terminals)

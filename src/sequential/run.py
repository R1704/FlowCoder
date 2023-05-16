from src.sequential.train import Training
from src.env import Primes
from config import *


env = Primes(100)
reset_grammar = True
from_checkpoint = False
# we need to make sure that the grammar and the model align.
training = Training(env, num_rounds, epochs, batch_size, max_trajectory, min_trajectory, display=display,
                    reset_grammar=reset_grammar, from_checkpoint=from_checkpoint)
updated_grammar = training.train()

print("Updated grammar:", updated_grammar.terminals)

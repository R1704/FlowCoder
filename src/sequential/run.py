import torch
from src.sequential.train import Training
from src.env import Primes


num_rounds = 5
epochs = 1000
batch_size = 8
max_trajectory = 10
min_trajectory = 5
display = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = Primes(100)
training = Training(env, num_rounds, epochs, batch_size, max_trajectory, min_trajectory, device, display=display)
updated_grammar = training.train()

print("Updated grammar:", updated_grammar.terminals)

import torch
import tqdm
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import numpy as np

from src.env import Primes
from src.sequential.grammar import Grammar
from model import FlowModel
from src.sequential.tokenizer import Tokenizer

from config import *

primes = Primes(100)
# primes.plot_rewards()
grammar = Grammar(primes)
tokenizer = Tokenizer(grammar)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

losses = []
sampled_functions = []
minibatch_loss = 0
update_freq = 2
logZs = []

tokenizer = Tokenizer(grammar)
model = FlowModel(tokenizer, device).to(device)
opt = torch.optim.Adam(model.parameters(), 3e-4)

for episode in tqdm.tqdm(range(epochs)):
    minibatch_loss = 0
    for _ in range(batch_size):
        state = ['<START>']
        PF = model(state)
        total_PF = 0
        t = 0
        rewrd = torch.tensor(0.0)
        while t < max_trajectory:

            # Here P_F is logits, so we want the Categorical to compute the softmax for us
            cat = Categorical(logits=PF)
            action = cat.sample()

            # "Go" to the next state
            new_state = state + tokenizer.decode([action.item()])

            # Accumulate the P_F sum
            total_PF += cat.log_prob(action)

            # Check if we've reached the minimum trajectory length
            if t >= min_trajectory - 1:
                # print(tokenizer.seq2tokens(''.join(new_state))) # DEBUG
                rewrd = torch.tensor(grammar.reward(tokenizer.seq2tokens(''.join(new_state)))).float()

            # Check if the action is the stop token and break if so
            if tokenizer.decode([action.item()])[0] == '<STOP>':
                break

            PF = model(new_state)
            state = new_state
            t += 1

        # trajectory balance loss
        loss = (model.logZ + total_PF - torch.log(rewrd).clip(-20)).pow(2)
        minibatch_loss += loss
        sampled_functions.append(tokenizer.seq2tokens(''.join(state)))
    losses.append(minibatch_loss.item() / batch_size)
    minibatch_loss.backward()
    opt.step()
    opt.zero_grad()
    logZs.append(model.logZ.item())


f, ax = plt.subplots(2, 1, figsize=(10, 6))
plt.sca(ax[0])
plt.plot(losses)
plt.yscale('log')
plt.ylabel('loss')
plt.sca(ax[1])
plt.plot(np.exp(logZs))
plt.ylabel('estimated Z')
plt.show()

print(model.logZ.exp())
print(sampled_functions[-30:])



count = 0
unique_functions = set(tuple(func) for func in sampled_functions)
new_primitives = set()
for func in unique_functions:
    func = list(func)
    if grammar.valid_function(func):
        result = grammar.evaluate(func)
        if result in grammar.env.primes:
            count += 1
            print(count, func, result)
            new_primitives.add(str(result))
print(new_primitives)

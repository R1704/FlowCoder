import torch
import tqdm
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import numpy as np

from src.env import Primes
from src.sequential.grammar import Grammar
from model import FlowModel
from src.sequential.tokenizer import Tokenizer

primes = Primes(100)
# primes.plot_rewards()
grammar = Grammar(primes)
tokenizer = Tokenizer()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


epochs = 2500
max_trajectory = 5
losses = []
sampled_functions = []
minibatch_loss = 0
update_freq = 2
logZs = []

# vocab = tokenizer.tokenize(''.join(grammar.primitives))
# for _ in range(max_trajectory):
#     vocab.insert(0, tokenizer.character_to_token('<BOS>'))


vocab = [''] + grammar.primitives
model = FlowModel(vocab, device).to(device)
opt = torch.optim.Adam(model.parameters(), 3e-4)

for episode in tqdm.tqdm(range(epochs)):
    state = ['']
    PF = model(state)
    total_PF = 0
    for t in range(max_trajectory):
        # Here P_F is logits, so we want the Categorical to compute the softmax for us
        cat = Categorical(logits=PF)
        action = cat.sample()
        # "Go" to the next state
        new_state = state + [vocab[action]]
        # Accumulate the P_F sum
        total_PF += cat.log_prob(action)

        if t == max_trajectory - 1:
            rewrd = torch.tensor(grammar.reward(''.join(new_state))).float()
        PF = model(new_state)
        state = new_state

    loss = (model.logZ + total_PF - torch.log(rewrd).clip(-20)).pow(2)
    minibatch_loss += loss

    sampled_functions.append(state)
    if episode % update_freq == 0:
        losses.append(minibatch_loss.item())
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0
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



# # for each string in sampled_functions, check whether it is a valid function and if so evaluate it
# def evaluate_functions(functions):
#     for sample in functions:
#         valid = valid_function(sample)
#         print(sample, valid)
#         if valid:
#             print(valid, sample, evaluate(sample))
#
#
#
# evaluate_functions(sampled_functions)
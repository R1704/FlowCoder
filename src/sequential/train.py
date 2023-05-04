import tqdm
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pickle

from src.sequential.grammar import Grammar
from model import FlowModel
from src.sequential.tokenizer import Tokenizer
from config import *


class Training:
    def __init__(self, env, num_rounds, epochs, batch_size, max_trajectory, min_trajectory, device, display=False):
        self.env = env
        self.num_rounds = num_rounds
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_trajectory = max_trajectory
        self.min_trajectory = min_trajectory
        self.device = device
        self.display = display
        self.grammar = Grammar(self.env)
        self.tokenizer = Tokenizer(self.grammar)

    def train_round(self):
        model = FlowModel(self.tokenizer, self.device).to(self.device)

        # Load the model weights if they exist
        if os.path.exists(model_weights_path):
            state_dict = torch.load(model_weights_path)
            num_tokens = len(self.tokenizer.vocab)
            model.resize_token_embeddings(num_tokens)
            model.resize_decoder_weights(num_tokens)

            # Update state_dict with the resized model's state_dict for the embeddings and decoder layers
            updated_state_dict = model.state_dict()
            state_dict["embeddings.weight"] = updated_state_dict["embeddings.weight"]
            state_dict["decoder.weight"] = updated_state_dict["decoder.weight"]
            state_dict["decoder.bias"] = updated_state_dict["decoder.bias"]

            model.load_state_dict(state_dict, strict=False)

        opt = optim.Adam(model.parameters(), 3e-4)

        sampled_functions = []
        losses = []
        logZs = []

        for _ in tqdm.tqdm(range(self.epochs)):
            rewards = torch.zeros(self.batch_size, dtype=torch.float).to(self.device)
            log_probs = torch.zeros(self.batch_size, dtype=torch.float).to(self.device)

            total_loss = 0
            for i in range(self.batch_size):
                state = ['<START>']
                PF = model(state)
                total_PF = 0
                t = 0
                while t < self.max_trajectory:

                    # Here P_F is logits, so we want the Categorical to compute the softmax for us
                    cat = Categorical(logits=PF)
                    action = cat.sample()

                    # "Go" to the next state
                    new_state = state + self.tokenizer.decode([action.item()])

                    # Accumulate the P_F sum
                    total_PF += cat.log_prob(action)

                    # # Check if we've reached the minimum trajectory length
                    # if t >= self.min_trajectory - 1:
                    #     reward = torch.tensor(self.grammar.reward(new_state)).float().to(self.device)
                    # else:
                    #     reward = torch.tensor(0.0).to(self.device)

                    if t == self.max_trajectory - 1:
                        reward = torch.tensor(self.grammar.reward(new_state)).float().to(self.device)
                    else:
                        reward = torch.tensor(0.0).to(self.device)

                    PF = model(new_state)
                    state = new_state
                    t += 1

                    # Check if the action is the stop token and break if so
                    if self.tokenizer.decode([action.item()])[0] == '<STOP>':
                        reward = torch.tensor(self.grammar.reward(new_state)).float().to(self.device)
                        break

                rewards[i] = reward
                log_probs[i] = total_PF
                sampled_functions.append(state)

            loss = (model.logZ + log_probs - torch.log(rewards).clip(-20)).pow(2).mean()
            total_loss += loss.item()
            losses.append(total_loss / self.batch_size)
            logZs.append(model.logZ.item())
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Save the model weights
        torch.save(model.state_dict(), model_weights_path)

        if self.display:
            f, ax = plt.subplots(2, 1, figsize=(10, 6))
            plt.sca(ax[0])
            plt.plot(losses)
            plt.yscale('log')
            plt.ylabel('loss')
            plt.sca(ax[1])
            plt.plot(np.exp(logZs))
            plt.ylabel('estimated Z')
            plt.show()
            print(sampled_functions[-20:])
        return sampled_functions

    def train(self):
        # Load the grammar from file if it exists
        if os.path.exists(grammar_path):
            with open(grammar_path, 'rb') as f:
                if f.peek():
                    self.grammar.terminals = pickle.load(f)
                    self.tokenizer = Tokenizer(self.grammar)
        for round_idx in range(self.num_rounds):
            sampled_functions = self.train_round()

            # Extract the unique functions generated during training
            unique_functions = set(tuple(func[1:-1])
                                   if func[-1] == '<STOP>'
                                   else tuple(func[1:])
                                   for func in sampled_functions)

            # Evaluate these functions and add the results to the list of primitives
            count = 0
            new_terminals = set(self.grammar.terminals)
            for func in unique_functions:
                func = list(func)
                if self.grammar.valid_function(func):
                    result = self.grammar.evaluate(func)
                    if result in self.env.primes:
                        count += 1
                        print(count, func, result)
                        new_terminals.add(str(result))
            print(new_terminals)

            # Update the grammar with the new primitives
            for new_terminal in new_terminals:
                if new_terminal not in self.grammar.terminals:
                    self.grammar.add_terminal(new_terminal)

            # Save the updated grammar to file
            with open(grammar_path, 'wb') as f:
                pickle.dump(self.grammar.terminals, f)

            self.tokenizer = Tokenizer(self.grammar)

            # Update the maximum trajectory length
            # self.max_trajectory += 2

        return self.grammar

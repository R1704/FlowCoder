import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pickle
from dataclasses import dataclass

from src.sequential.grammar import Grammar
from src.sequential.model import FlowModel
from src.sequential.tokenizer import Tokenizer
from src.env import Environment
from config import *


@dataclass
class Training:
    env: Environment
    num_rounds: int
    epochs: int
    batch_size: int
    max_trajectory: int
    min_trajectory: int
    display: bool
    reset_grammar: bool = False
    from_checkpoint: bool = False
    display: bool = False
    grammar: Grammar = None
    tokenizer: Tokenizer = None
    device: torch = None

    def __post_init__(self):
        if self.grammar is None:
            self.grammar = Grammar(self.env)
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(self.grammar)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train_round(self):
        model = FlowModel(self.tokenizer, self.device)

        if self.from_checkpoint:
            print('loading model from checkpoint')
            self.load_model(model)

        # if torch.cuda.device_count() > 1:
        #     print(f'training on {torch.cuda.device_count()} GPUs')
        #     model = nn.DataParallel(model)  # parallelize across GPUs

        model.to(self.device)

        opt = optim.Adam(model.parameters(), 3e-4)

        sampled_functions = []
        losses = []
        logZs = []

        for _ in tqdm.tqdm(range(self.epochs)):

            rewards = torch.zeros(self.batch_size, dtype=torch.float).to(self.device)
            log_probs = torch.zeros(self.batch_size, dtype=torch.float).to(self.device)

            total_loss = 0
            for i in range(self.batch_size):

                # Sample a trajectory
                output_seq, log_prob = model()
                output_seq = self.tokenizer.decode(output_seq)

                # Compute the reward
                reward = torch.tensor(self.grammar.reward(output_seq)).float()

                # Update the reward and log_prob
                rewards[i] = reward
                log_probs[i] = log_prob

                # Add the trajectory to the list of sampled functions
                sampled_functions.append(output_seq)

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
            self.plot_stats(logZs, losses)

        print(torch.cuda.memory_summary(device=self.device))

        return sampled_functions

    def plot_stats(self, logZs, losses):
        f, ax = plt.subplots(2, 1, figsize=(10, 6))
        plt.sca(ax[0])
        plt.plot(losses)
        plt.yscale('log')
        plt.ylabel('loss')
        plt.sca(ax[1])
        plt.plot(np.exp(logZs))
        plt.ylabel('estimated Z')
        plt.show()

    def load_model(self, model):
        # TODO: check whether this is correct and if all models are loaded.
        #  Check this out: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        #  Pay attention to DataParallel, loading on GPU/ CPU


        # Load the model weights if they exist
        if os.path.exists(model_weights_path):
            state_dict = torch.load(model_weights_path)
            num_tokens = len(self.tokenizer.vocab)
            model.resize_token_embeddings(num_tokens)
            model.resize_decoder_weights(num_tokens)

            # Update state_dict with the resized model's state_dict for the embeddings and decoder layers
            updated_state_dict = model.state_dict()
            state_dict["embeddings.weight"] = updated_state_dict["embeddings.weight"]
            state_dict["embeddings.bias"] = updated_state_dict["embeddings.bias"]
            state_dict["output_layer.weight"] = updated_state_dict["output_layer.weight"]
            state_dict["output_layer.bias"] = updated_state_dict["output_layer.bias"]
            state_dict["logZ.weight"] = updated_state_dict["logZ.weight"]
            state_dict["logZ.bias"] = updated_state_dict["logZ.bias"]

            model.load_state_dict(state_dict, strict=False)

    def train(self):

        if self.reset_grammar:
            # Reset the grammar
            self.grammar.reset_grammar()
        terminals = self.grammar.load_grammar()
        if terminals:
            self.grammar.terminals = terminals
        print(f'starting with terminals: {self.grammar.terminals}')
        self.tokenizer = Tokenizer(self.grammar)

        for round_idx in range(self.num_rounds):
            sampled_functions = self.train_round()
            self.extract_new_terminals(sampled_functions)
            self.save_new_terminals()
            self.tokenizer = Tokenizer(self.grammar)

            # Update the maximum trajectory length
            # self.max_trajectory += 2

        return self.grammar

    def save_new_terminals(self):
        # Save the updated grammar to file
        with open(grammar_path, 'wb') as f:
            pickle.dump(self.grammar.terminals, f)

    def extract_new_terminals(self, sampled_functions):

        # Extract the unique functions generated during training
        unique_functions = set(tuple(func[1:]) for func in sampled_functions)

        # Evaluate these functions and add the results to the list of primitives
        valid_evaluated = [(func, self.grammar.evaluate(list(func))) for func in unique_functions if
                           self.grammar.valid_function(list(func))]
        print(valid_evaluated)
        new_primes = [func for func in valid_evaluated if func[1] in self.env.primes]

        for func in new_primes:
            self.grammar.add_terminal(str(func[1]))

        # Print some statistics
        print(
            f'valid functions: {len(valid_evaluated)} from {len(unique_functions)}, that is {len(valid_evaluated) / (len(unique_functions) + 10e-20):.3f}')
        print(
            f'prime functions: {len(new_primes)} from {len(valid_evaluated)}, {len(new_primes) / (len(valid_evaluated) + 10e-20):.3f}')
        print(f'new_terminals: {self.grammar.terminals}')

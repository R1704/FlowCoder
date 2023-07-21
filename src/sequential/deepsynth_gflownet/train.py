import torch.nn

from src.sequential.deepsynth.dsl import *
from src.sequential.deepsynth.run_experiment import *
from src.sequential.deepsynth_gflownet.data import *

from torch.distributions.categorical import Categorical
from torch.optim import Adam

import matplotlib.pyplot as pp
from dataclasses import dataclass

import logging

@dataclass
class Training:
    n_epochs: int
    batch_size: int
    learning_rate: float
    model_path: str
    data: Data

    def __post_init__(self):
        assert self.n_epochs <= self.data.dataset_size, f'not enough data for {self.n_epochs} epochs'

    def train(self, model, cfg):
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        losses = []
        logZs = []
        rewards = 0

        for epoch in tqdm.tqdm(range(self.n_epochs), ncols=40):
            batch_IOs, batch_program, latent_batch_IOs = self.data.get_next_batch(self.batch_size)

            state = []  # start with an empty state
            total_forward = 0
            non_terminal = cfg.start  # start with the CFGs start symbol

            # keep sampling until we have a complete program
            frontier = deque()
            initial_non_terminals = deque()
            initial_non_terminals.append(non_terminal)
            frontier.append((None, initial_non_terminals))

            while len(frontier) != 0:
                partial_program, non_terminals = frontier.pop()

                # If we are finished with the trajectory/ have a constructed program
                if len(non_terminals) == 0:
                    program = reconstruct_from_compressed(partial_program, target_type=cfg.start[0])
                    reward = Reward(program, batch_program, batch_IOs, self.data.dsl)

                    # Compute loss and backpropagate
                    loss = (logZ + total_forward - torch.log(reward).clip(-20)).pow(2)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    losses.append(loss)
                    logZs.append(logZ)

                    if reward.item() == 2:
                        rewards += 1

                    if epoch % 99 == 0:
                        logging.info(
                            f'Epoch: {epoch}\n'
                            f'Loss: {loss.item()}\n'
                            f'LogZ: {logZ.item()}\n'
                            f'Forward: {total_forward}\n'
                            f'Reward: {torch.log(reward).clip(-20)}\n'
                            f'Total Rewards: {rewards} / {self.n_epochs} = {rewards / self.n_epochs}\n'
                            f'Forward logits: {forward_logits}, {forward_logits.shape}\n'
                            f'Program length: {len(state)}, {program}\n'
                            # f'{}\n'
                        )

                # Keep digging
                else:
                    non_terminal = non_terminals.pop()

                    forward_logits, logZ = model(state, non_terminal, latent_batch_IOs)
                    # print(forward_logits)
                    # forward_logits = torch.nn.Softmax(forward_logits)
                    # print(forward_logits)

                    cat = Categorical(logits=forward_logits)
                    action = cat.sample()  # returns idx

                    total_forward += cat.log_prob(action)

                    # use the forward logits to sample the next derivation
                    program = model.idx2primitive[action.item()]
                    state = state + [program]

                    program_args = cfg.rules[non_terminal][program]
                    new_partial_program = (program, partial_program)
                    new_non_terminals = non_terminals.copy()

                    for arg in program_args:
                        new_non_terminals.append(arg)
                    frontier.append((new_partial_program, new_non_terminals))

        self.plot_results(losses, logZs)
        torch.save(model.state_dict(), self.model_path)

    def plot_results(self, losses, logZs):
        f, ax = pp.subplots(2, 1, figsize=(10, 6))
        losses = [l.cpu().detach().numpy() for l in losses]
        logZs = [z.cpu().detach().numpy() for z in logZs]
        pp.sca(ax[0])
        pp.plot(losses)
        pp.yscale('log')
        pp.ylabel('loss')
        pp.sca(ax[1])
        pp.plot(logZs)
        pp.ylabel('estimated Z')
        pp.show()


def Reward(program: Program, batch_program, task, dsl):
    program_checker = make_program_checker(dsl, task[0])
    rewrd = torch.tensor(float(program_checker(program, True)))
    print(f'$$$$$$$$$$$$$$$$$$$$$$$', rewrd)
    logging.debug(f'found program: {program}')
    logging.debug(f'actual program: {batch_program[0]}')
    if rewrd.item() == 2:
        logging.info('-----found the correct program-----')
        logging.info(f'found program: {program}')
        logging.info(f'actual program: {batch_program[0]}')
        logging.info(f'reward: {rewrd.item()}')
    return rewrd

# import numpy as np
# def cosine_similarity(v1, v2):
#     # Convert lists to numpy arrays
#     v1 = np.array(v1)
#     v2 = np.array(v2)
#
#     # If both of the vectors are empty, return minimum loss
#     if v1.size == 0 and v2.size == 0:
#         return 1.0
#
#     # If one of the vectors is empty, return maximum loss
#     if v1.size == 0 or v2.size == 0:
#         return 0.0
#
#     # Pad the shorter vector with zeros
#     if len(v1) < len(v2):
#         v1 = np.pad(v1, (0, len(v2) - len(v1)))
#     elif len(v2) < len(v1):
#         v2 = np.pad(v2, (0, len(v1) - len(v2)))
#
#     # Compute cosine similarity
#     dot_product = np.dot(v1, v2)
#     norm_v1 = np.linalg.norm(v1)
#     norm_v2 = np.linalg.norm(v2)
#
#     # Add a small constant to the denominator to avoid division by zero
#     epsilon = 1e-10
#     similarity = dot_product / (norm_v1 * norm_v2 + epsilon)
#
#     # Return the cosine distance (1 - similarity) as the loss
#     return 1 - similarity

import numpy as np

def mean_squared_error(y_true, y_pred):
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # If one of the vectors is empty, return maximum loss
    if y_true.size == 0 and y_pred.size == 0:
        return 1

    # If one of the vectors is empty, return maximum loss
    if y_true.size == 0 or y_pred.size == 0:
        return 0

    # Pad the shorter vector with zeros
    if len(y_true) < len(y_pred):
        y_true = np.pad(y_true, (0, len(y_pred) - len(y_true)))
    elif len(y_pred) < len(y_true):
        y_pred = np.pad(y_pred, (0, len(y_true) - len(y_pred)))

    # Compute mean squared error
    mse = np.mean((y_true - y_pred)**2)
    reward = np.exp(-mse)
    return reward

def make_program_checker(dsl: DSL, examples) -> Callable[[Program, bool], int]:
    # TODO: Naive reward for now, obvs improve this. Could be parameterized.
    correct_program_rwd = 10
    is_program_rwd = 1
    none_rwd = 0
    def checker(prog: Program, use_cached_evaluator: bool) -> int:
        if use_cached_evaluator:
            for i, example in enumerate(examples):
                input, output = example
                my_out = prog.eval(dsl, input, i)
                logging.debug(f'\nMy out: {my_out}'
                              f'\nActual out: {output}')
                return mean_squared_error(my_out, output)
            #     if out is None or None in out:
            #         return none_rwd
            #     elif output != out:
            #         return is_program_rwd
            # return correct_program_rwd
        else:
            for example in examples:
                input, output = example
                my_out = prog.eval_naive(dsl, input)
            #     if out is None or None in out:
            #         return none_rwd
            #     elif output != out:
            #         return is_program_rwd
            # return correct_program_rwd
                return mean_squared_error(my_out, output)
    return checker

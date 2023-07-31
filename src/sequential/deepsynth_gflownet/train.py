import torch.nn

from src.sequential.deepsynth.dsl import *
from src.sequential.deepsynth.run_experiment import *
from src.sequential.deepsynth_gflownet.data import *
from src.sequential.deepsynth_gflownet.reward import *

from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn as nn

import matplotlib.pyplot as pp
from dataclasses import dataclass

import logging

import random


@dataclass
class Training:
    n_epochs: int
    batch_size: int
    learning_rate: float
    e_steps: int
    m_step_threshold: float
    m_steps: int
    model_path: str
    data: Data
    model: nn.Module
    reward: Reward

    def __post_init__(self):
        assert self.n_epochs <= self.data.dataset_size, f'not enough data for {self.n_epochs} epochs'

    def sample_program(self, latent_batch_IOs):

        state = []  # start with an empty state
        total_forward = 0
        non_terminal = self.data.cfg.start  # start with the CFGs start symbol

        # keep sampling until we have a complete program
        frontier = deque()
        initial_non_terminals = deque()
        initial_non_terminals.append(non_terminal)
        frontier.append((None, initial_non_terminals))

        while len(frontier) != 0:
            partial_program, non_terminals = frontier.pop()

            # # TODO: idx should be sampled by GFN
            # # Choose a random program from the frontier
            # idx = random.choice(range(len(frontier)))
            #
            # # Rotate deque by -idx, pop from right and rotate back
            # frontier.rotate(-idx)
            # partial_program, non_terminals = frontier.pop()
            # frontier.rotate(idx)

            # If we are finished with the trajectory/ have a constructed program, calculate loss, update GFN
            if len(non_terminals) == 0:
                program = reconstruct_from_compressed(partial_program, target_type=self.data.cfg.start[0])
                return program, forward_logits, logZ, total_forward

            # Keep digging
            else:
                non_terminal = non_terminals.pop()

                forward_logits, logZ = self.model(state, non_terminal, latent_batch_IOs)
                # print(forward_logits)
                # forward_logits = torch.nn.Softmax(forward_logits)
                # print(forward_logits)

                cat = Categorical(logits=forward_logits)
                action = cat.sample()  # returns idx

                total_forward += cat.log_prob(action)

                # use the forward logits to sample the next derivation
                program = self.model.idx2primitive[action.item()]
                state = state + [program]

                program_args = self.data.cfg.rules[non_terminal][program]
                new_partial_program = (program, partial_program)
                new_non_terminals = non_terminals.copy()

                for arg in program_args:
                    new_non_terminals.append(arg)
                frontier.append((new_partial_program, new_non_terminals))

    def e_step(self, optim_gfn, epoch):

        # keep track of losses and logZs
        losses = []
        logZs = []
        tries = 0
        reward = torch.tensor(0)


        # Sample task
        batch_IOs, batch_program, latent_batch_IOs = self.data.get_next_batch(self.batch_size)


        # until correct program is found or >= n_tries
        while reward.item() != 1.0 and tries < self.e_steps:
            program, forward_logits, logZ, total_forward = self.sample_program(latent_batch_IOs)
            tries += 1
            reward = self.reward_(program, batch_program, batch_IOs, self.data.dsl)

        # save program, task pairs for sleep phase

        # Compute loss and backpropagate
        loss = (logZ + total_forward - torch.log(reward).clip(-20)).pow(2)

        loss.backward()
        optim_gfn.step()
        optim_gfn.zero_grad()

        losses.append(loss)
        logZs.append(logZ)

        self.print_stats(epoch, loss, logZ, total_forward, reward, forward_logits, program)

        return losses, logZs


    def m_step(self, optim):
        losses = []
        for _ in range(self.m_steps):
            # Sample task
            batch_IOs, batch_program, latent_batch_IOs = self.data.get_next_batch(self.batch_size)
            program, forward_logits, logZ, total_forward = self.sample_program(latent_batch_IOs)
            reward = self.reward_(program, batch_program, batch_IOs, self.data.dsl)
            print(f'reward in m_step {reward}')
            loss = torch.tensor(-torch.log(reward).clip(-20), requires_grad=True)
            loss.backward()
            losses.append(loss)
            optim.step()
            optim.zero_grad()

        return losses



    def train(self):

        # Optimizers for generative model and GFN_Forward
        optim_gen = Adam(self.model.parameters(), lr=self.learning_rate)
        optim_gfn = Adam(self.model.forward_logits.parameters(), lr=self.learning_rate)
        correct = 0
        for epoch in tqdm.tqdm(range(self.n_epochs), ncols=40):

            # Optimize GFlowNet
            gfn_losses, logZs = self.e_step(optim_gfn, epoch)

            # Optimize Generative Model
            if gfn_losses[-1] < self.m_step_threshold:
                self.m_step(optim_gen)

        self.plot_results(gfn_losses, logZs)

        # Save model
        # TODO: Save models independently?
        torch.save(self.model.state_dict(), self.model_path)

    def sleep(self):
        ...
        def replays(self):
            ...

        def fantasies(self):
            ...

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


    def print_stats(self, epoch, loss, logZ, total_forward, reward, forward_logits, program):
        if epoch % 99 == 0:
            logging.info(
                f'Epoch: {epoch}\n'
                f'Loss: {loss.item()}\n'
                f'LogZ: {logZ.item()}\n'
                f'Forward: {total_forward}\n'
                f'Reward: {torch.log(reward).clip(-20)}\n'
                # f'Total Rewards: {rewards} / {self.n_epochs} = {rewards / self.n_epochs}\n'
                f'Forward logits: {forward_logits}, {forward_logits.shape}\n'
                # f'Program length: {len(state)}, {program}\n'
                # f'{}\n'
            )


    def make_program_checker(self, dsl: DSL, examples) -> Callable[[Program, bool], int]:
        def checker(prog: Program, use_cached_evaluator: bool) -> int:
            if use_cached_evaluator:
                for i, example in enumerate(examples):
                    # TODO: If a different reward is used,
                    #  note that there may be multiple examples, so account for that
                    input, output = example
                    pred_out = prog.eval(dsl, input, i)
                    logging.debug(f'\nexample nr. {i} :::::::: with io relation: \t {input} ---> {output} and prediction {pred_out}')
                    if output != pred_out:
                        return torch.tensor(0.0)
                return torch.tensor(1.0)
                # return self.reward(output, pred_out)
            else:
                for example in examples:
                    input, output = example
                    pred_out = prog.eval_naive(dsl, input)
                    if output != pred_out:
                        return torch.tensor(0.0)
                return torch.tensor(1.0)
                # return self.reward(output, pred_out)

        return checker

    def reward_(self, program: Program, batch_program, task, dsl):
        # program_checker = self.make_program_checker(dsl, task[0])
        # rewrd = program_checker(program, True)
        logging.debug(f'actual program: {batch_program[0]}')
        logging.debug(f'found program: {program}')
        # logging.debug(f'Same program: {batch_program[0] == program}')
        # TODO: Looking at the actual program for comparison is a little hacky,
        #  but otherwise we need to deal with variables, which makes the problem a lot harder.
        rewrd = torch.tensor(batch_program[0] == program).to(torch.int8)
        logging.debug(f'Reward: {rewrd}')

        # if rewrd.item() == 1:
        #     logging.info('-----found the correct program-----')
        #     logging.info(f'found program: {program}')
        #     logging.info(f'actual program: {batch_program[0]}')
        #     logging.info(f'reward: {rewrd.item()}')
        return rewrd


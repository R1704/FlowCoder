import torch.nn

from src.sequential.deepsynth.dsl import *
from src.sequential.deepsynth.run_experiment import *
from src.sequential.deepsynth_gflownet.data import *
from src.sequential.deepsynth_gflownet.reward import *
from src.sequential.deepsynth_gflownet.utils import *

from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn as nn

import matplotlib.pyplot as pp
from dataclasses import dataclass

import logging



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
    device: torch

    def __post_init__(self):
        assert self.n_epochs <= self.data.dataset_size, f'not enough data for {self.n_epochs} epochs'

    def get_next_rules(self, S):
        return [(S, p) for p in self.data.cfg.rules[S].keys()]

    def sample_program(self, batch_IOs):
        states = [['START']] * self.batch_size
        total_forward = torch.zeros(self.batch_size)
        non_terminals = [self.data.cfg.start] * self.batch_size  # start with the CFGs start symbol
        frontiers = [{nt: self.get_next_rules(nt)} for nt in non_terminals]
        final_programs = [[]] * self.batch_size

        while any(frontiers):

            forward_logits, logZs = self.model(states, batch_IOs)
            mask = [[1 if rule in flatten(fr.values()) else 0 for rule in self.model.state_encoder.rules] for fr in
                    frontiers]
            mask = torch.tensor(mask, device=self.device)  # Convert mask to tensor
            forward_logits = forward_logits - (1 - mask) * 100  # Subtract 100 from invalid actions
            actions = []
            for i in range(self.batch_size):
                cat = Categorical(logits=forward_logits[i])
                action = cat.sample()  # returns idx
                total_forward[i] += cat.log_prob(action).item()
                actions.append(action)

            for i in range(self.batch_size):
                if frontiers[i]:
                    rule = self.model.state_encoder.idx2rule[actions[i].item()]
                    nt, program = rule
                    final_programs[i].append(program)
                    frontiers[i].pop(nt)

                    states[i] = states[i] + [rule]
                    program_args = self.data.cfg.rules[nt][program]
                    frontiers[i].update({nt: self.get_next_rules(nt) for nt in program_args})

            if not any(frontiers):
                print('$$$$$$$$$$$$$ final_programs $$$$$$$$$$$$$')
                print(final_programs)
                return final_programs, forward_logits, logZs, total_forward

    def e_step(self, optim_gfn, epoch):

        # keep track of losses and logZs
        losses = []
        logZs = []
        tries = 0
        reward = torch.tensor(0)

        # Sample task
        batch_IOs, batch_program = self.data.get_next_batch(self.batch_size)

        # until correct program is found or >= n_tries
        while reward.item() != 1.0 and tries < self.e_steps:
            program, forward_logits, logZ, total_forward = self.sample_program(batch_IOs)
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
            batch_IOs, batch_program = self.data.get_next_batch(self.batch_size)

            # program, forward_logits, logZ, total_forward = self.sample_program(latent_batch_IOs)
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

            for i in range(self.data.dataset_size // self.batch_size):
                batch_IOs, batch_program = self.data.get_next_batch(self.batch_size)
                final_programs, forward_logits, logZs, total_forward = self.sample_program(batch_IOs)


                # Optimize GFlowNet
                # gfn_losses, logZs = self.e_step(optim_gfn, epoch)

                # Optimize Generative Model
                # if gfn_losses[-1] < self.m_step_threshold:
                #     self.m_step(optim_gen)

        # self.plot_results(gfn_losses, logZs)

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


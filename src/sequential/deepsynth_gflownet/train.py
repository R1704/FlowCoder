import torch.nn

from src.config import *

from src.sequential.deepsynth.dsl import *
from src.sequential.deepsynth.run_experiment import *
from src.sequential.deepsynth_gflownet.data import *
from src.sequential.deepsynth_gflownet.reward import *
from src.sequential.deepsynth_gflownet.utils import *

from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn as nn

import matplotlib.pyplot as plt

from dataclasses import dataclass

import logging

import time
import random


@dataclass
class Training:
    batch_size: int
    learning_rate_trn: float
    learning_rate_gfn: float
    e_steps: int
    m_step_threshold: float
    m_steps: int
    model_path: str
    data: Data
    model: nn.Module
    reward: Reward
    device: torch

    def __post_init__(self):
        self.epochs = self.data.dataset_size // (self.e_steps * self.m_steps * self.batch_size)
        self.optimizer_generative = Adam(self.model.parameters(), lr=self.learning_rate_trn)
        self.optimizer_policy = Adam(self.model.forward_logits.parameters(), lr=self.learning_rate_gfn)
        self.criterion = nn.NLLLoss()

    def get_next_rules(self, S):
        return [(S, p) for p in self.data.cfg.rules[S].keys()]

    def get_mask(self, rules: list):
        mask = [1 if rule in rules else 0 for rule in self.model.state_encoder.rules]
        mask = torch.tensor(mask, device=self.device)
        return mask

    def sample_program_dfs(self, batch_IOs, epsilon=0., beta=1.):
        """

        :param batch_IOs:
        :param epsilon:
        :param beta: beta should be < 1
        :return:
        """
        states = [['START'] for _ in range(len(batch_IOs))]

        # Initialise container for forward logits accumulation
        total_forwards = torch.zeros(self.batch_size, device=self.device)

        frontiers = [deque([self.data.cfg.start]) for _ in range(len(batch_IOs))]

        programs = [[] for _ in range(len(batch_IOs))]

        while any(frontiers):

            forward_logits, logZs = self.model(states, batch_IOs)

            # Tempering with the policy for exploration purposes
            if random.random() < epsilon:
                # Uniform sampling if random is smaller than epsilon
                forward_logits = torch.full(forward_logits.shape, fill_value=1.0 / forward_logits.shape[1],
                                            device=self.device)
            # Exponentiate by beta (we are multiplying since we are in log space)
            if beta != 1:
                forward_logits *= beta

            for i in range(len(batch_IOs)):
                if frontiers[i]:
                    next_nt = frontiers[i].pop()
                    rules = self.get_next_rules(next_nt)
                    mask = self.get_mask(rules)
                    forward_logits[i] = forward_logits[i] - (1 - mask) * 100
                    cat = Categorical(logits=forward_logits[i])
                    action = cat.sample()
                    total_forwards[i] += cat.log_prob(action).item()
                    rule = self.model.state_encoder.idx2rule[action.item()]
                    nt, program = rule
                    programs[i] = (program, programs[i])

                    states[i] = states[i] + [rule]
                    program_args = self.data.cfg.rules[nt][program]
                    frontiers[i].extend(program_args)

        if not any(frontiers):
            reconstructed = [reconstruct_from_compressed(program, target_type=self.data.cfg.start[0])
                             for program in programs]
            return logZs, total_forwards, reconstructed, states

    def sleep(self, correct_programs):
        """
        The sleep phase for refining the forward_logits using saved correct programs.
        :param correct_programs: List of tuples containing (program, batch_IOs, states) for each correct program.
        """
        print('SLEEP PHASE')

        # Prepare the batch from correct_programs
        batch_programs, batch_IOs, correct_states = zip(*correct_programs)
        correct_states = [state[1:] for state in correct_states]
        states = [['START'] for _ in range(len(batch_IOs))]

        # Initialise container for forward logits accumulation
        total_forwards = torch.zeros(self.batch_size, device=self.device)

        frontiers = [deque([self.data.cfg.start]) for _ in range(len(batch_IOs))]

        programs = [[] for _ in range(len(batch_IOs))]

        while any(frontiers):
            forward_logits, logZs = self.model(states, batch_IOs)
            for i in range(len(batch_IOs)):
                if frontiers[i]:
                    next_nt = frontiers[i].pop()
                    correct_rule_idx = self.model.state_encoder.rule2idx[
                        correct_states[i].pop(0)]  # pop the next correct rule from the list
                    total_forwards[i] += forward_logits[i][correct_rule_idx]
                    rule = self.model.state_encoder.idx2rule[correct_rule_idx]

                    nt, program = rule
                    programs[i] = (program, programs[i])

                    states[i] = states[i] + [rule]
                    program_args = self.data.cfg.rules[nt][program]
                    frontiers[i].extend(program_args)

        if not any(frontiers):
            # reconstructed = [reconstruct_from_compressed(program, target_type=self.data.cfg.start[0])
            #                  for program in programs]

            sleep_loss = -total_forwards.mean()
            sleep_loss.backward()

            self.optimizer_policy.step()  # (E-step)
            self.optimizer_policy.zero_grad()

        # def sleep(self, correct_programs):
    #     """
    #     The sleep phase for refining the forward_logits using saved correct programs.
    #     :param correct_programs: List of tuples containing (program, batch_IOs, states) for each correct program.
    #     """
    #     print('SLEEP PHASE')
    #
    #     # Prepare the batch from correct_programs
    #     batch_programs, batch_IOs, correct_states = zip(*correct_programs)
    #
    #     states = [['START'] for _ in range(len(batch_IOs))]
    #
    #     # Initialise container for forward logits accumulation
    #     total_forwards = torch.zeros(self.batch_size, device=self.device)
    #
    #     frontiers = [deque([self.data.cfg.start]) for _ in range(len(batch_IOs))]
    #
    #     programs = [[] for _ in range(len(batch_IOs))]
    #
    #
    #     while any(frontiers):
    #         forward_logits, logZs = self.model(states, batch_IOs)
    #         for i in range(len(batch_IOs)):
    #             j = 1
    #             if frontiers[i]:
    #                 next_nt = frontiers[i].pop()
    #                 print(next_nt)
    #                 mask = torch.zeros(len(self.model.state_encoder.rules), device=self.device)
    #                 print(self.model.state_encoder.rule2idx[correct_states[i][j]])
    #                 mask[self.model.state_encoder.rule2idx[correct_states[i][j]]] = 1
    #                 forward_logits[i] = forward_logits[i] - (1 - mask) * 100
    #                 cat = Categorical(logits=forward_logits[i])
    #                 action = cat.sample()
    #                 assert action == self.model.state_encoder.rule2idx[correct_states[i][j]]
    #                 total_forwards[i] += cat.log_prob(action).item()
    #                 rule = self.model.state_encoder.idx2rule[action.item()]
    #
    #                 nt, program = rule
    #                 programs[i] = (program, programs[i])
    #
    #                 states[i] = states[i] + [rule]
    #                 program_args = self.data.cfg.rules[nt][program]
    #                 frontiers[i].extend(program_args)
    #
    #     if not any(frontiers):
    #         reconstructed = [reconstruct_from_compressed(program, target_type=self.data.cfg.start[0])
    #                          for program in programs]
    #
    #         sleep_loss = -total_forwards.mean()
    #         sleep_loss.backward()
    #
    #
    #         self.optimizer_policy.step()  # (E-step)
    #         self.optimizer_policy.zero_grad()





        sleep_batch = []
        # for batch_idx, ios in enumerate(correct_program_ios):

    #         new_ios = []
    #         for ios_idx, io in enumerate(ios):
    #             i, o = io
    #             predicted_output = programs[batch_idx].eval(self.data.dsl, i, ios_idx)
    #             new_ios.append([i, predicted_output])
    #         sleep_batch.append(new_ios)
    #     for _ in range(5):
    #         logZs, total_forwards, programs = self.sample_program_dfs(sleep_batch)
    #         rewards = self.rewards(programs, batch_programs, batch_IOs, self.data.dsl)
    #
    #         # Compute loss and backpropagate
    #         e_loss = -torch.log(total_forwards)
    #         e_loss = e_loss.mean()
    #         e_loss.backward()
    #
    #         self.optimizer_policy.step()  # (E-step)
    #         self.optimizer_policy.zero_grad()
    #
    #     correct_programs = []
    #     correct_program_ios =

    def states_to_target_indices(self, trajectory):
        target_indices = []
        for state in trajectory:
            index_sequence = [self.model.state_encoder.rule2idx[rule] for rule in state]
            target_indices.append(index_sequence)

        return torch.tensor(target_indices, dtype=torch.long, device=self.device)

    def train(self):

        start_time = time.time()  # to store the starting time of the training
        program_counter = 0
        total_rewards = []

        # keep track of losses and logZs
        e_step_losses = []
        m_step_losses = []

        total_logZs = []

        correct_programs = []


        for epoch in tqdm.tqdm(range(self.epochs), ncols=40):
            correct_programs_per_epoch = 0
            nr_of_programs_per_epoch = 0
            print()
            logging.info(f'Computing E-step')
            # Optimize GFlowNet
            self.unfreeze_parameters(self.model.forward_logits)  # Unfreeze GFlowNet parameters
            self.freeze_parameters(self.model.transformer)  # Freeze transformer parameters
            for _ in range(self.e_steps):

                # Sample tasks
                batch_IOs, batch_programs = self.data.get_next_batch(self.batch_size)
                program_counter += self.batch_size

                # Predict programs
                logZs, total_forwards, programs, states = self.sample_program_dfs(batch_IOs, epsilon=0.05, beta=0.8)

                # Calculate rewards
                rewards = self.rewards(programs, batch_programs, batch_IOs, self.data.dsl)
                total_rewards.append(rewards.sum().item() / rewards.size(0))
                correct_programs_per_epoch += rewards.sum().item()
                nr_of_programs_per_epoch += rewards.size(0)

                # Compute loss and backpropagate
                # Trajectory balance
                e_loss = (logZs + total_forwards / len(max(states, key=len)) - torch.log(rewards).clip(-20)).pow(2)
                # TODO: Check that max(states, key=len) makes sense
                e_loss = e_loss.mean()
                e_loss.backward()

                self.optimizer_policy.step()  # (E-step)
                self.optimizer_policy.zero_grad()

                # Bookkeeping
                e_step_losses.append(e_loss.item())
                total_logZs.append(logZs.mean().item())

                # Collect good programs
                for i in range(self.batch_size):
                    if rewards[i] == 1.:
                        # print(f'found correct program {programs[i]}, {batch_programs[i]}, {len(correct_programs)}')
                        correct_programs.append((programs[i], batch_IOs[i], states[i]))

                # If we have enough for a batch, sleep
                if len(correct_programs) >= self.batch_size:
                    self.sleep(correct_programs[:self.batch_size])

                    # Reset
                    correct_programs = correct_programs[self.batch_size:]

            # Optimize Generative Model
            if True: #e_step_losses[-1] < self.m_step_threshold:
                self.unfreeze_parameters(self.model.transformer)  # Unfreeze transformer parameters
                self.freeze_parameters(self.model.forward_logits)  # Freeze GFlowNet parameters

                logging.info(f'Computing M-step')
                for _ in range(self.m_steps):
                    batch_IOs, batch_programs = self.data.get_next_batch(self.batch_size)
                    program_counter += self.batch_size
                    _, _, programs, _ = self.sample_program_dfs(batch_IOs)
                    rewards = self.rewards(programs, batch_programs, batch_IOs, self.data.dsl)
                    total_rewards.append(rewards.sum().item() / rewards.size(0))
                    m_loss = -torch.log(rewards).clip(-20).mean()
                    m_loss.backward()
                    m_step_losses.append(m_loss.item())
                    self.optimizer_generative.step()  # (M-step)
                    self.optimizer_generative.zero_grad()

                    correct_programs_per_epoch += rewards.sum().item()
                    nr_of_programs_per_epoch += rewards.size(0)

            if epoch % 5 == 0:
                avg_programs_per_sec = self.calculate_programs_per_second(start_time, program_counter)

                logging.info(
                    f'total rewards: {sum(total_rewards) / ((epoch + 1) * self.e_steps * self.m_steps * self.batch_size)}\n'
                    f'correct_programs_per_epoch: {correct_programs_per_epoch} out of {nr_of_programs_per_epoch} = {correct_programs_per_epoch/nr_of_programs_per_epoch}\n'
                    f'e_loss: {e_step_losses[-1]}\n'
                    f'm_loss: {m_step_losses[-1]}\n'
                    f'Z: {np.exp(total_logZs[-1])}\n'
                    f'Programs created per second on average: {avg_programs_per_sec:.2f}\n'
                    f'Total programs created: {program_counter}'
                    )

                # self.plot_results(e_step_losses, m_step_losses, total_logZs, total_rewards, epoch)
        self.plot_results(e_step_losses, m_step_losses, total_logZs, total_rewards, epoch)

        # Save model
        torch.save(self.model.state_dict(), self.model_path)

    def freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = True


    @staticmethod
    def plot_results(e_step_losses, m_step_losses, all_logZs, program_ratios, epoch):

        plt.figure(figsize=(15, 15))
        plt.subplot(4, 1, 1)
        plt.plot(e_step_losses)
        plt.title('E-step Losses Over Time')
        plt.xlabel('epochs')
        plt.ylabel('e loss')

        plt.subplot(4, 1, 2)
        plt.plot(m_step_losses)
        plt.title('M-step Losses Over Time')
        plt.xlabel('epochs')
        plt.ylabel('m loss')

        plt.subplot(4, 1, 3)
        plt.plot(np.exp(all_logZs))
        plt.title('Z Over Time')
        plt.xlabel('epochs')
        plt.ylabel('Z')

        plt.subplot(4, 1, 4)
        plt.plot(program_ratios)
        plt.title('Correct Program Ratios Over Time')
        plt.xlabel('epochs')
        plt.ylabel('ratio')

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'epoch {epoch}'))
        plt.show()

    @staticmethod
    def calculate_programs_per_second(start_time, program_counter):
        elapsed_time = time.time() - start_time
        avg_programs_per_sec = program_counter / elapsed_time
        return avg_programs_per_sec




    def get_edit_distance(self, program, ios, dsl):
        for i, io in enumerate(ios):
            input, output = io
            predicted_output = program.eval(dsl, input, i)
            edit_distance = self.reward.edit_distance(output, predicted_output)
            return np.exp(edit_distance)
    def compare_outputs(self, program, ios, dsl):
        for x, io in enumerate(ios):
            i, o = io
            predicted_output = program.eval(dsl, i, x)
            if o != predicted_output:
                return 0.0
            return 1.0





    def rewards(self, programs, batch_program, batch_ios, dsl):
        # program_checker = self.make_program_checker(dsl, task[0])
        # rewrd = program_checker(program, True)


        # TODO: Looking at the actual program for comparison is a little hacky,
        #  but otherwise we need to deal with variables, which makes the problem a lot harder.
        rewrd = torch.tensor([a == b for a, b in zip(programs, batch_program)], device=self.device).float()
        rewrd.requires_grad_()

        # rewrd = torch.tensor([self.check(dsl, ios, program) for program, ios in zip(programs, batch_ios)], requires_grad=True, device=self.device).exp()

        # rewrd = torch.tensor([self.compare_outputs(program, ios, dsl) for program, ios in zip(programs, batch_ios)], requires_grad=True, device=self.device)
        # print(rewrd)
        # print(list(zip(programs, batch_program)))
        # print(rewrd == torch.tensor([a == b for a, b in zip(programs, batch_program)], device=self.device).float())
        return rewrd



















 #
    #
    # def sample_program_top_down(self, batch_IOs, s=None):
    #
    #     # The top is either defined by s if it is given or by the start symbol of the cfg
    #     s = self.data.cfg.start if s is None else s
    #
    #     # Initialise trajectories
    #     states = [['START'] for _ in range(self.batch_size)]
    #
    #     # Initialise container for forward logits accumulation
    #     total_forwards = torch.zeros(self.batch_size, device=self.device)
    #
    #     # A frontier is a queue of lists of dictionaries. A dictionary consists of non-terminal: [rules(non-terminal)] pairs
    #     # frontiers = [deque([{s: self.get_next_rules(s)}]) for _ in range(self.batch_size)]
    #     # frontiers = [deque([[s]]) for _ in range(self.batch_size)]
    #
    #     # Actual program constructions
    #     # argument_stack = [[] for _ in range(self.batch_size)]
    #     # program_stack = [[] for _ in range(self.batch_size)]
    #
    #
    #
    #     while any(frontiers):
    #
    #         forward_logits, logZs = self.model(states, batch_IOs)
    #         mask = self.get_mask(frontiers)
    #         forward_logits = forward_logits - (1 - mask) * 100
    #         cat = Categorical(logits=forward_logits)
    #         actions = cat.sample()
    #         total_forwards += cat.log_prob(actions)
    #
    #         for i in range(self.batch_size):
    #             if frontiers[i]:
    #
    #                 front = frontiers[i].pop()
    #
    #                 print(f'front: {front}')
    #
    #                 candidate_rules = [self.get_next_rules(nt) for nt in front]
    #                 print(f'candidate_rules: {candidate_rules}')
    #
    #
    #
    #                 # If we popped from the frontier and we made a prediction
    #
    #
    #
    #     if not any(frontiers):
    #         print('$$$$$$$$$$$$$ final_programs $$$$$$$$$$$$$')
    #         # print(arguments)


    # def top_down_v2(self, nt, states, batch_IOs):
    #     next_rules = self.get_next_rules(nt)
    #     forward_logits, logZ = self.model(states, batch_IOs)
    #     mask = self.get_mask(next_rules)
    #     forward_logits = forward_logits - (1 - mask) * 100
    #     cat = Categorical(logits=forward_logits)
    #     action = cat.sample()
    #     rule = self.model.state_encoder.idx2rule[action.item()]
    #     nt, program = rule
    #
    #     frontier = self.data.cfg[nt][program]
    #     arguments = []



    # def top_down(self, arguments, frontiers, states, batch_IOs, total_forward):
    #     # Frontiers is a list of size batch_size.
    #     # Each element in that list is a queue which contains dictionaries of nt: [rules]
    #
    #     forward_logits, logZs = self.model(states, batch_IOs)
    #
    #     for i in range(self.batch_size):
    #         if frontiers[i]:
    #
    #             # Get the latest dictionary, containing the arguments of the parent function
    #             # We need to pick one rule for each nt
    #             nts_dict = frontiers[i].pop()
    #
    #             # Get the mask and mask the forward logits
    #             mask = self.get_mask(flatten(nts_dict.values()))
    #             forward_logits[i] = forward_logits[i] - (1 - mask) * 100
    #
    #             # Predict next rule
    #             cat = Categorical(logits=forward_logits[i])
    #             action = cat.sample()  # returns idx
    #             total_forward[i] += cat.log_prob(action).item()
    #             rule = self.model.state_encoder.idx2rule[action.item()]
    #             nt, program = rule
    #             print()
    #             print(nt, program, rule)
    #
    #             # update state
    #             states[i].append(rule)
    #
    #             # # Append program to arguments
    #             # arguments[i].append(rule)
    #
    #             # Remove nt from frontier
    #             nts_dict.pop(nt)
    #
    #             # If we still have nts in the remaining dict, restack the frontier
    #             if len(nts_dict) != 0:
    #                 frontiers[i].append(nts_dict)
    #
    #             # If the program can't be expanded further, return it
    #             if isinstance(program, Variable) or rule in self.model.state_encoder.terminal_rules:
    #                 print(f'arguments {arguments}')
    #                 arguments = sorted(arguments, key=lambda x: x[0][1][1])
    #                 arguments = [p for nt, p in arguments]
    #
    #             # get all args of the program that still need to be expanded
    #             # and put them in the frontier
    #             args = self.data.cfg.rules[nt][program]
    #             frontiers[i].append({nt: self.get_next_rules(nt) for nt in args})
    #
    #             arguments, frontiers, states, batch_IOs, total_forward = self.top_down(arguments, frontiers, states, batch_IOs, total_forward)
    #             # print(f'arguments {arguments}')
    #             # return the function
    #             # return [Function(program, [arg for arg in sorted(arguments, key=lambda x: x[0][1][1])])], frontiers, states, batch_IOs, total_forward
    #             return Function(program, arguments), frontiers, states, batch_IOs, total_forward



    # def sample_program_bottom_up(self, batch_IOs):
    #     states = [['START'] for _ in range(self.batch_size)]
    #     total_forward = torch.zeros(self.batch_size, device=self.device)
    #     frontiers = [deque({'START': self.model.state_encoder.terminal_rules}) for _ in range(self.batch_size)]
    #     programs = [[] for _ in range(self.batch_size)]
    #
    #     while any(frontiers):
    #         forward_logits, logZs = self.model(states, batch_IOs)
    #         mask = self.get_mask(frontiers)
    #         forward_logits = forward_logits - (1 - mask) * 100
    #
    #         cat = Categorical(logits=forward_logits)
    #         actions = cat.sample()
    #         total_forward += cat.log_prob(actions)
    #
    #         for i in range(self.batch_size):
    #             if frontiers[i]:
    #                 rule = self.model.state_encoder.idx2rule[actions[i].item()]
    #                 nt, program = rule
    #
    #                 # parents = self.model.state_encoder.get_parents(rule)
    #                 # # Finish program
    #                 # programs[i] = (program, programs[i])
    #                 frontiers[i].pop(nt)
    #                 states[i] = states[i] + [rule]
    #                 program_args = self.data.cfg.rules[nt][program]
    #                 frontiers[i].update({nt: self.get_next_rules(nt) for nt in program_args})
    #
    #         if not any(frontiers):
    #             print('$$$$$$$$$$$$$ final_programs $$$$$$$$$$$$$')
    #             print(programs)
    #             return programs, forward_logits, logZs, total_forward


    # def reconstruct_from_list(self, program_list, target_type):
    #     if len(program_list) == 1:
    #         return program_list.pop()
    #     else:
    #         p = program_list.pop()
    #         if isinstance(p, (New, BasicPrimitive)):
    #             list_arguments = p.type.ends_with(target_type)
    #             arguments = [None] * len(list_arguments)
    #             for i in range(len(list_arguments)):
    #                 arguments[len(list_arguments) - i - 1] = reconstruct_from_list(
    #                     program_list, list_arguments[len(list_arguments) - i - 1]
    #                     )
    #             return Function(p, arguments)
    #         if isinstance(p, Variable):
    #             return p
    #         assert False

    # def e_step(self, optim_gfn, epoch):
    #
    #     # keep track of losses and logZs
    #     losses = []
    #     logZs = []
    #     tries = 0
    #     reward = torch.tensor(0)
    #
    #     # Sample task
    #     batch_IOs, batch_program = self.data.get_next_batch(self.batch_size)
    #
    #     # until correct program is found or >= n_tries
    #     while reward.item() != 1.0 and tries < self.e_steps:
    #         forward_logits, logZs, total_forwards, programs = self.sample_program_dfs(batch_IOs)
    #         tries += 1
    #         reward = self.reward_(program, batch_program, batch_IOs, self.data.dsl)
    #
    #     # save program, task pairs for sleep phase
    #
    #     # Compute loss and backpropagate
    #     loss = (logZ + total_forward - torch.log(reward).clip(-20)).pow(2)
    #
    #     loss.backward()
    #     optim_gfn.step()
    #     optim_gfn.zero_grad()
    #
    #     losses.append(loss)
    #     logZs.append(logZ)
    #
    #     self.print_stats(epoch, loss, logZ, total_forward, reward, forward_logits, program)
    #
    #     return losses, logZs

    #
    # def m_step(self, optim):
    #     losses = []
    #     for _ in range(self.m_steps):
    #         # Sample task
    #         batch_IOs, batch_program = self.data.get_next_batch(self.batch_size)
    #
    #         # program, forward_logits, logZ, total_forward = self.sample_program(latent_batch_IOs)
    #         reward = self.reward_(program, batch_program, batch_IOs, self.data.dsl)
    #         print(f'reward in m_step {reward}')
    #         loss = torch.tensor(-torch.log(reward).clip(-20), requires_grad=True)
    #         loss.backward()
    #         losses.append(loss)
    #         optim.step()
    #         optim.zero_grad()
    #
    #     return losses

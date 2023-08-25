import difflib
import torch.nn
from rapidfuzz.distance import Levenshtein
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
from collections import Counter
import torch.nn.functional as F


@dataclass
class Training:
    batch_size: int
    learning_rate_trn: float
    learning_rate_gfn: float
    e_steps: int
    m_step_threshold_init: float
    m_steps: int
    replay_prob: float
    fantasy_prob: float
    model_path: str
    data: Data
    model: nn.Module
    reward: Reward
    device: torch

    def __post_init__(self):
        self.epochs = self.data.dataset_size // (self.e_steps * self.m_steps * self.batch_size)

        # Threshold decrement value for linearly decreasing the threshold over the training period; adjust as needed
        self.threshold_decrement = self.m_step_threshold_init / (self.e_steps * self.m_steps)

        # Extract parameters that are NOT part of forward_logits
        generative_params = (p for n, p in self.model.named_parameters() if "forward_logits" not in n)
        self.optimizer_generative = Adam(generative_params, lr=self.learning_rate_trn)
        self.optimizer_policy = Adam(self.model.forward_logits.parameters(), lr=self.learning_rate_gfn)

        self.max_reward = 1.

    def get_next_rules(self, S):
        return [(S, p) for p in self.data.cfg.rules[S].keys()]

    def get_mask(self, rules: list):
        mask = [1 if rule in rules else 0 for rule in self.model.state_encoder.rules]
        mask = torch.tensor(mask, device=self.device)
        return mask

    def sample_program_dfs(self, batch_IOs, epsilon=0., beta=1.):

        states = [['START'] for _ in range(len(batch_IOs))]

        # Initialise container for forward logits accumulation
        total_forwards = torch.zeros(self.batch_size, device=self.device)
        final_logZ = torch.zeros(self.batch_size, device=self.device)

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

                    final_logZ[i] = logZs[i]  # We do this to account for different trajectory lengths. I want the predicted partition function of each trajectory
                    rule = self.model.state_encoder.idx2rule[action.item()]
                    nt, program = rule
                    programs[i] = (program, programs[i])

                    states[i] = states[i] + [rule]
                    program_args = self.data.cfg.rules[nt][program]
                    frontiers[i].extend(program_args)

        if not any(frontiers):
            reconstructed = [reconstruct_from_compressed(program, target_type=self.data.cfg.start[0])
                             for program in programs]
            return final_logZ, total_forwards, reconstructed, states






    def replay(self, correct_programs):

        print('REPLAY')

        # Prepare the batch from correct_programs
        batch_programs, batch_IOs, correct_states_list = zip(*correct_programs)

        # Initialise container for forward logits accumulation
        total_forwards = torch.zeros(self.batch_size, device=self.device)

        # Initial state
        states = [['START'] for _ in range(len(batch_IOs))]

        # Maximum trajectory length
        max_traj_length = max([len(correct_states) for correct_states in correct_states_list])

        # Process each step of each trajectory
        for t in range(max_traj_length - 1):  # adjusted for -1 as we're skipping the first token for prediction

            forward_logits, _ = self.model(states, batch_IOs)

            for i, correct_states in enumerate(correct_states_list):
                if t + 1 < len(correct_states):  # using t + 1 to get the next state/token for prediction
                    rule = correct_states[t + 1]

                    # We know which rule was correct, so we don't need to sample
                    correct_rule_idx = self.model.state_encoder.rule2idx[rule]

                    # In the normal sampling,
                    # log softmax is done by Categorical. Here we need to do it manually.
                    total_forwards[i] += torch.log(F.softmax(forward_logits[i], dim=0)[correct_rule_idx])

                    # Update the state for the next prediction
                    states[i].append(rule)

        sleep_loss = -total_forwards.mean()  # TODO: Try weighting it (e.g. -10 * ...)
        self.optimizer_policy.zero_grad()
        sleep_loss.backward()
        self.optimizer_policy.step()  # (E-step sleep)


    def fantasy(self, correct_programs):

        print('FANTASY')

        # Prepare the batch from correct_programs
        batch_programs, batch_IOs, correct_states_list = zip(*correct_programs)

        all_fantasy_batches = []
        for program in batch_programs:
            fantasy_batch = []
            for batch_idx, ios in enumerate(batch_IOs):

                new_ios = []
                for ios_idx, io in enumerate(ios):
                    i, o = io
                    predicted_output = program.eval(self.data.dsl, i, ios_idx)
                    new_ios.append([i, predicted_output])
                fantasy_batch.append(new_ios)
            all_fantasy_batches.append(fantasy_batch)


        for prog_idx, fantasy_batch in enumerate(all_fantasy_batches):

            # Initialise container for forward logits accumulation
            total_forwards = torch.zeros(self.batch_size, device=self.device)

            # Initial state
            states = [['START'] for _ in range(len(fantasy_batch))]

            # Maximum trajectory length
            max_traj_length = len(correct_states_list[prog_idx])

            # Process each step of each trajectory
            for t in range(max_traj_length - 1):  # adjusted for -1 as we're skipping the first token for prediction

                forward_logits, _ = self.model(states, fantasy_batch)

                for i in range(self.batch_size):
                    rule = correct_states_list[prog_idx][t+1]
                    correct_rule_idx = self.model.state_encoder.rule2idx[rule]
                    total_forwards[i] += torch.log(F.softmax(forward_logits[i], dim=0)[correct_rule_idx])

                    # Update the state for the next prediction
                    states[i].append(rule)

            sleep_loss = -total_forwards.mean()
            self.optimizer_policy.zero_grad()
            sleep_loss.backward()
            self.optimizer_policy.step()  # (E-step sleep)







    def fantasy_all(self, all_fantasy_batches):


        print('FANTASY ALL')


        for i, (_, ios, correct_state) in enumerate(all_fantasy_batches):

            # Initialise container for forward logits accumulation
            total_forwards = torch.zeros(self.batch_size, device=self.device)

            # Initial state
            states = [['START'] for _ in range(self.batch_size)]

            # Maximum trajectory length
            max_traj_length = len(correct_state)

            # Process each step of each trajectory
            for t in range(max_traj_length - 1):  # adjusted for -1 as we're skipping the first token for prediction
                forward_logits, _ = self.model(states, ios)

                for i in range(self.batch_size):
                    rule = correct_state[t + 1]
                    correct_rule_idx = self.model.state_encoder.rule2idx[rule]

                    # Get the softmax probabilities
                    probabilities = F.softmax(forward_logits[i], dim=0)

                    # Accumulate the log probability of the correct action
                    total_forwards[i] += torch.log(probabilities[correct_rule_idx]).item()

                    # Update the state for the next prediction
                    states[i].append(rule)

            # Compute the sleep loss
            sleep_loss = -total_forwards.mean()
            self.optimizer_policy.zero_grad()
            sleep_loss.backward()
            self.optimizer_policy.step()  # E-step sleep

    # def fantasy_one(self, program, io_batch, state):
    #     print('FANTASY ONE')
    #
    #     # Initialise container for forward logits accumulation
    #     total_forwards = torch.zeros(self.batch_size, device=self.device)
    #
    #     # Initial state
    #     states = [['START'] for _ in range(len(state))]
    #
    #     # Maximum trajectory length
    #     max_traj_length = len(state)
    #
    #     # Process each step of each trajectory
    #     for t in range(max_traj_length - 1):  # adjusted for -1 as we're skipping the first token for prediction
    #
    #         forward_logits, _ = self.model(states, io_batch)
    #
    #         for i in range(self.batch_size):
    #             rule = state[t + 1]
    #             correct_rule_idx = self.model.state_encoder.rule2idx[rule]
    #             total_forwards[i] += forward_logits[i][correct_rule_idx]
    #
    #             # Update the state for the next prediction
    #             states[i].append(rule)
    #
    #     sleep_loss = -total_forwards.mean()
    #     self.optimizer_policy.zero_grad()
    #     sleep_loss.backward()
    #     self.optimizer_policy.step()  # (E-step sleep)

    def train(self):

        start_time = time.time()  # Store the start time of training for performance analysis
        program_counter = 0  # Counter for the number of programs processed
        program_ratios = []
        # Current threshold value, initialized to the initial threshold
        current_threshold = self.m_step_threshold_init

        # Moving average of the GFlowNet's training loss
        alpha = 0.3
        moving_average_loss = None

        # Lists to store losses during the E-step and M-step optimization
        e_step_losses = []
        m_step_losses = []

        total_logZs = []  # List to keep track of log partition functions

        correct_programs = []  # List to collect correct programs found during training
        all_fantasy_batches = []
        collected_programs = set()  # Set to keep track of collected programs
        unique_programs = set()  # Set to keep track of collected programs
        all_programs = []
        all_correct_programs = []
        all_real_programs = []


        for epoch in tqdm.tqdm(range(self.epochs), ncols=40):
            correct_programs_per_epoch = 0
            nr_of_programs_per_epoch = 0
            print()

            ##############################################
            ##############################################
            ##############################################
            ########### ***** E-STEP ***** ###############
            ##############################################
            ##############################################
            ##############################################

            logging.info(f'Computing E-step')

            # Unfreeze GFlowNet parameters and freeze transformer parameters for the E-step optimization
            self.unfreeze_parameters(self.model.forward_logits)
            self.freeze_parameters(self.model.transformer)
            for _ in range(self.e_steps):

                # Sample tasks for training
                batch_IOs, batch_programs = self.data.get_next_batch(self.batch_size)
                all_real_programs.extend(batch_programs)

                # Predict programs and calculate associated log partition functions and other parameters
                # epsilon and beta for exploration
                logZs, total_forwards, programs, states = self.sample_program_dfs(batch_IOs, epsilon=0.2, beta=0.8)
                # logZs, total_forwards, programs, states = self.sample_program_dfs(batch_IOs, epsilon=0., beta=1.)
                program_counter += self.batch_size

                # Calculate rewards for the predicted programs
                rewards = self.rewards(programs, batch_programs, batch_IOs, self.data.dsl)
                # for r, rp, pp, ios in zip(rewards, programs, batch_programs, batch_IOs):
                #     print(
                #         f'reward: {r}\n'
                #         f'real program: {rp}\n'
                #         f'predicted program: {pp}\n'
                #         f'ios: {ios}\n------------\n'
                #         )

                # Count correct programs in the batch
                correct_programs_per_epoch += (rewards == self.max_reward).sum().item()
                nr_of_programs_per_epoch += rewards.size(0)
                unique_programs.add(p for p in programs)
                all_programs.extend(programs)

                # steps = torch.tensor([len(s) - 1 for s in states], device=self.device)
                # Compute the loss and perform backpropagation
                e_loss = (logZs + total_forwards - torch.log(rewards).clip(-20)).pow(2)  # Trajectory Balance
                e_loss = e_loss.mean()

                # Update moving average of the GFlowNet's training loss
                moving_average_loss = e_loss.item() if moving_average_loss is None else alpha * e_loss.item() + (
                            1 - alpha) * moving_average_loss

                # Update policy parameters (E-step)
                self.optimizer_policy.zero_grad()
                e_loss.backward()
                self.optimizer_policy.step()

                # Record the loss and logZ
                e_step_losses.append(e_loss.item())
                total_logZs.append(logZs.mean().item())

                # Collect good programs
                for i in range(self.batch_size):
                    if rewards[i] == self.max_reward:
                        # print(f'found correct program {programs[i]}, {batch_programs[i]}, {len(correct_programs)}')
                        # if programs[i] not in collected_programs:
                        if True:
                            # print(f'adding {programs[i]}\n'
                            #       f'actual program {batch_programs[i]}\n'
                            #       f'n_programs collected {len(correct_programs)+1}')
                            collected_programs.add(programs[i])  # Add the program to the set of collected programs
                            correct_programs.append((programs[i], batch_IOs[i], states[i]))
                            all_correct_programs.append(programs[i])
                # If we have enough for a batch, sleep
                if len(correct_programs) >= self.batch_size:
                    rand = random.random()
                    # rand = 0
                    if rand < self.replay_prob:
                        self.replay(correct_programs[:self.batch_size])

                    if rand < self.fantasy_prob:
                        self.fantasy(correct_programs[:self.batch_size])

                    # Reset
                    correct_programs = correct_programs[self.batch_size:]
                    collected_programs = set()

                #
                #
                #
                #
                #
                #
                #
                #
                #
                #
                # # For each program in the batch
                # for idx, program in enumerate(programs):
                #
                #     # Create a list to store input output relations
                #     fantasy_batch = []
                #     include_program = True  # flag to check if all predicted outputs are valid
                #
                #     # We go through all inputs per program to get more data
                #     for batch_idx, ios in enumerate(batch_IOs):
                #
                #         # Create a list to store new tasks, given the predicted program
                #         new_ios = []
                #         for ios_idx, io in enumerate(ios):
                #             i, o = io
                #             predicted_output = program.eval(self.data.dsl, i, ios_idx)
                #
                #             # If we ever find a shitty program, we don't want to include it
                #             if predicted_output in [None, []] or None in predicted_output:
                #                 include_program = False  # mark program as invalid if any predicted output is None or []
                #                 break
                #
                #             new_ios.append([i, predicted_output])
                #         if not include_program:
                #             break
                #
                #         # If the io relation is good, include the program
                #         fantasy_batch.append(new_ios)
                #
                #     if include_program:  # only include the program if all predicted outputs are valid
                #         all_fantasy_batches.append((program, fantasy_batch, states[idx]))
                #         # print(f'{len(all_fantasy_batches)} all_fantasy_batches')
                #
                # # for x in all_fantasy_batches:
                # #     print(x)
                # # print(len(all_fantasy_batches))
                # # if random.random() < 0.3:
                # # if True:
                # #     if len(all_fantasy_batches) >= self.batch_size:
                # #         self.fantasy_all(all_fantasy_batches[:self.batch_size])
                # #         all_fantasy_batches = all_fantasy_batches[self.batch_size:]
                # #
                # #
                #
                #









            # print(f'self.threshold_decrement, moving_average_loss, current_threshold === {self.threshold_decrement, moving_average_loss, current_threshold}')



            ##############################################
            ##############################################
            ##############################################
            ########### ***** M-STEP ***** ###############
            ##############################################
            ##############################################
            ##############################################


            # Optimize Generative Model
            # Check whether the moving average of the GFlowNet's training loss is below the current threshold
            if moving_average_loss < current_threshold or epoch == self.epochs - 1:
                self.unfreeze_parameters(self.model.transformer)  # Unfreeze transformer parameters
                self.freeze_parameters(self.model.forward_logits)  # Freeze GFlowNet parameters

                logging.info(f'Computing M-step')
                # Iterate through the number of M-steps
                for _ in range(self.m_steps):
                    # Sample another batch of tasks for training
                    batch_IOs, batch_programs = self.data.get_next_batch(self.batch_size)
                    program_counter += self.batch_size
                    all_real_programs.extend(batch_programs)

                    # Predict programs without specific parameters epsilon and beta
                    _, _, programs, _ = self.sample_program_dfs(batch_IOs)

                    # Calculate rewards for the predicted programs
                    rewards = self.rewards(programs, batch_programs, batch_IOs, self.data.dsl)

                    for i in range(self.batch_size):
                        if rewards[i] == self.max_reward:
                            all_correct_programs.append(programs[i])

                    unique_programs.update(programs)
                    all_programs.extend(programs)

                    # Compute the M-step loss, which measures how well the generative model is performing
                    # The loss is based on the negative log of the rewards, and it's clipped to avoid numerical instability
                    m_loss = -torch.log(rewards).clip(-20).mean()

                    # Perform backpropagation to calculate the gradients
                    m_loss.backward()

                    # Record the loss for later analysis
                    m_step_losses.append(m_loss.item())

                    # Update the generative model's parameters using the calculated gradients
                    self.optimizer_generative.step()  # Update (M-step)

                    # Reset gradients for the next iteration
                    self.optimizer_generative.zero_grad()

                    # Keep track of correct programs found during the M-step
                    correct_programs_per_epoch += (rewards == self.max_reward).sum().item()
                    nr_of_programs_per_epoch += rewards.size(0)

                # Decrease the threshold value linearly over the epochs
                current_threshold -= self.threshold_decrement

            program_ratios.append(correct_programs_per_epoch/nr_of_programs_per_epoch)

            if epoch % 5 == 0:
                avg_programs_per_sec = self.calculate_programs_per_second(start_time, program_counter)

                logging.info(

                    f'correct_programs_per_epoch            : {correct_programs_per_epoch} out of {nr_of_programs_per_epoch} = {correct_programs_per_epoch/nr_of_programs_per_epoch}\n'
                    f'e_loss                                : {e_step_losses[-1]}\n'
                    f'm_loss                                : {m_step_losses[-1] if m_step_losses else None}\n'
                    f'Z                                     : {np.exp(total_logZs[-1])}\n'
                    f'Programs created per second on average: {avg_programs_per_sec:.2f}\n'
                    f'Total programs created                : {program_counter}\n'
                    f'Unique programs created               : {len(unique_programs)}\n'
                    f'Nr of unique correct programs         : {len(set(all_correct_programs))}\n'
                    f'Unique correctly predicted programs   : {set(all_correct_programs)}\n'
                    f'Nr of unique real programs            : {len(set(all_real_programs))}\n'
                    # f'Unique real programs                  : {set(all_real_programs)}\n'
                    # f'All programs created: {Counter(all_programs)}\n'
                    )

        self.plot_results(e_step_losses, m_step_losses, total_logZs, program_ratios, epoch)

        # Save model
        torch.save(self.model.state_dict(), self.model_path)








    ##############################################
    ##############################################
    ##############################################
    ########### ***** REWARD ***** ###############
    ##############################################
    ##############################################
    ##############################################



    def get_edit_distance(self, program, ios, dsl):
        # FIXME: we're looping but returning directly. is this intended?
        for i, io in enumerate(ios):
            input, output = io
            predicted_output = program.eval(dsl, input, i)
            edit_distance = self.reward.edit_distance(output, predicted_output)
            return np.exp(edit_distance)



    def compare_outputs(self, program, ios, dsl):
        # For any of the examples,
        # if the real output isn't as predicted, return 0, else 1
        for io in ios:
            i, o = io
            predicted_output = program.eval_naive(dsl, i)
            if o != predicted_output:
                return 0.0
        return 1.0

    import difflib

    def normalized_edit_distance(self, seq1, seq2):
        sm = difflib.SequenceMatcher(None, seq1, seq2)
        max_len = max(len(seq1), len(seq2))

        # Calculating the sum of lengths of the edited portions
        # Using max to account for 'replace' which affects both sequences.
        edit_distance = sum([max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"])

        normalized_distance = edit_distance / max_len
        return 1 - normalized_distance

    import Levenshtein

    def normalized_similarity(self, seq1, seq2):
        # print(seq1, seq2)
        # Levenshtein.normalized_distance computes the normalized edit distance
        return 1 - Levenshtein.distance(seq1, seq2) / max(len(seq1), len(seq2))

    def rewards(self, programs, batch_program, batch_ios, dsl):
        # program_checker = self.make_program_checker(dsl, task[0])
        # rewrd = program_checker(program, True)


        # TODO: Looking at the actual program for comparison is a little hacky,
        #  but otherwise we need to deal with variables, which makes the problem a lot harder.
        # rewrd = torch.tensor([a == b for a, b in zip(programs, batch_program)], device=self.device).float()
        # rewrd.requires_grad_()

        # rewrd = torch.tensor([self.get_edit_distance(program, ios, dsl) for program, ios in zip(programs, batch_ios)], requires_grad=True, device=self.device).exp()
        # print(rewrd)
        # rewrd = rewrd - rewrd.min()
        # rewrd = rewrd / rewrd.max()





        # rewrd = []
        # for x, ios in enumerate(batch_ios):
        #     rs = []
        #     if programs[x] == batch_program[x]:
        #         rewrd.append(self.max_reward)
        #         # print("Program:", programs[x])
        #         # print("Batch Program:", batch_program[x])
        #     else:
        #         for i, io in enumerate(ios):
        #             input, output = io
        #             predicted_output = programs[x].eval(dsl, input, i)
        #             # print(programs[x], batch_program[x])
        #             # print(output, predicted_output)
        #             if predicted_output is None:
        #                 rs.append(0.)
        #                 continue
        #             if isinstance(output, int) or isinstance(predicted_output, int):
        #                 print(output, predicted_output)
        #                 rs.append(0.)
        #                 continue
        #             r = self.normalized_similarity(output, predicted_output)
        #             # if r == 1:
        #             #     print(programs[x], batch_program[x], output, predicted_output, r)
        #             rs.append(r)
        #         rewrd.append(np.mean(rs))
        #         # print(predicted_output, output, rewrd[-1])
        # rewrd = torch.tensor(rewrd, requires_grad=True, device=self.device)




        rewrd = []
        for i, program, ios in zip(range(len(programs)), programs, batch_ios):
            res = self.compare_outputs(program, ios, dsl)

            # if res == 1. or program == batch_program[i]:
            #     for x, io in enumerate(ios):
            #         inp, out = io
            #         predicted_output = program.eval_naive(dsl, inp)
            #         print(f'reward          : {res}')
            #         print(f'input           : {inp}')
            #         print(f'real output     : {out}')
            #         print(f'predicted_output: {predicted_output}')
            #         print(f'real program    : {batch_program[i]}')
            #         print(f'pred program    : {program}')
            #         print('---------------------------------')


            rewrd.append(res)
        return torch.tensor(rewrd, requires_grad=True, device=self.device)

        # rewrd = torch.tensor([self.compare_outputs(program, ios, dsl) for program, ios in zip(programs, batch_ios)], requires_grad=True, device=self.device)
        # print(rewrd)
        # print(list(zip(programs, batch_program)))
        # print(rewrd == torch.tensor([a == b for a, b in zip(programs, batch_program)], device=self.device).float())
        # rewrd = torch.tensor([self.naive_reward(ios, prog) for prog, ios in zip(programs, batch_program)], requires_grad=True, device=self.device).float()
        return rewrd

    @staticmethod
    def plot_results(e_step_losses, m_step_losses, all_logZs, program_ratios, epoch):
        plt.figure(figsize=(15, 15))

        data = [(e_step_losses, 'E-step Losses Over Time', 'e loss'),
                (m_step_losses, 'M-step Losses Over Time', 'm loss'),
                (np.exp(all_logZs), 'Z Over Time', 'Z'),
                (program_ratios, 'Correct Program Ratios Over Time', 'ratio')]

        for i, (d, title, ylabel) in enumerate(data, start=1):
            plt.subplot(4, 1, i)
            plt.plot(d)
            plt.title(title)
            plt.xlabel('epochs')
            plt.ylabel(ylabel)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'epoch {epoch}'))
        plt.show()

    @staticmethod
    def calculate_programs_per_second(start_time, program_counter):
        elapsed_time = time.time() - start_time
        avg_programs_per_sec = program_counter / elapsed_time
        return avg_programs_per_sec

    def freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = True
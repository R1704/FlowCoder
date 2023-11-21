# Time
import datetime


# debugging
import logging

# Math
import random

# List stuff
from itertools import chain
# import numpy as np
# from collections import deque

# Torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Style
from dataclasses import dataclass
import tqdm

# Deepsynth
from deepsynth.dsl import *
from deepsynth.run_experiment import *
from deepsynth.experiment_helper import *
from deepsynth.program import Program

# FlowCoder
from flowcoder.data import Data
from flowcoder.utils import *
from flowcoder.config import *
from flowcoder.reward import *
from flowcoder.model import freeze_parameters, unfreeze_parameters



# TODO: show that Z approximates total flow
# TODO: fantasy batch in replay
# TODO: check all TODOs
# TODO: ablation sleep
# TODO: EM vs normal
# TODO: learning rate scheduler
# TODO: hyperparameter search




@dataclass
class Training:
    min_program_depth: int
    max_program_depth: int
    n_tasks: int
    epochs: int
    batch_size: int
    learning_rate_gen: float
    learning_rate_pol: float
    e_steps: int
    m_step_threshold_init: float
    m_steps: int
    inference_steps: int
    alpha: float
    beta: float
    epsilon: float
    replay_prob: float
    fantasy_prob: float
    data: Data
    model: nn.Module
    save_checkpoint: bool

    def __post_init__(self):
        # self.epochs = self.data.dataset_size // (self.e_steps * self.m_steps * self.batch_size)
        # self.epochs = self.data.dataset_size // self.batch_size

        # Threshold decrement value for linearly decreasing the threshold over the training period; adjust as needed
        self.threshold_decrement = self.m_step_threshold_init / (self.e_steps * self.m_steps)
        self.moving_average_loss = None

        # Extract parameters that are NOT part of forward_logits
        generative_params = (p for n, p in self.model.named_parameters() if "forward_logits")

        # Instantiate optimizers
        self.optimizer_generative = Adam(generative_params, lr=self.learning_rate_gen)
        self.optimizer_policy = Adam(self.model.forward_logits.parameters(), lr=self.learning_rate_pol)

        # Set max reward in case we want to amplify the reward
        self.max_reward = 10.

        # Saving data for analysis
        self.csv_file = os.path.join(RESULTS, f'stats_{datetime.datetime.now()}.csv')
        headers = ['Mode', 'Depth', 'Epoch', 'Steps', 'Task Name', 'Program', 'Examples']
        create_csv(self.csv_file, headers)

        # Create a dictionary of CFGs of different depths, so we can learn it gradually
        self.cfgs = {depth: self.data.dsl.DSL_to_CFG(self.data.type_request, max_program_depth=depth)
                     for depth in range(self.min_program_depth, self.max_program_depth + 1)}

    def get_next_rules(self, S, depth):
        return [(S, p) for p in self.cfgs[depth].rules[S].keys()]

    def get_mask(self, rules: list):
        mask = [1 if rule in rules else 0 for rule in self.model.state_encoder.rules]
        mask = torch.tensor(mask, device=device)
        return mask

    def sample_program_dfs(self, batch_IOs, depth, epsilon=0., beta=1.):

        # Initialize the state of each program in the batch to 'START'
        states = [['START'] for _ in range(len(batch_IOs))]

        # Initialize tensors to store cumulative logits and partition functions (logZs) for each program in the batch
        total_forwards = torch.zeros(self.batch_size, device=device)
        final_logZs = torch.zeros(self.batch_size, device=device)

        # Initialize a queue (frontier) for each program to store the next non-terminals to be expanded
        frontiers = [deque([self.data.cfg.start]) for _ in range(len(batch_IOs))]

        # Initialize an empty list to store the generated program for each program in the batch
        programs = [[] for _ in range(len(batch_IOs))]

        # Loop continues until all frontiers are empty, i.e., all programs are fully generated
        while any(frontiers):

            # Get logits and partition functions (logZs) for the current states and IOs
            forward_logits, logZs = self.model(states, batch_IOs)

            # Exploration: With probability epsilon, use uniform sampling instead of model's logits
            if random.random() < epsilon:
                forward_logits = torch.full(forward_logits.shape, fill_value=1.0 / forward_logits.shape[1],
                                            device=device)

            # Apply tempering by multiplying logits with beta (in log-space, this is equivalent to exponentiation)
            if beta != 1:
                forward_logits *= beta

            # Loop over each program in the batch
            for i in range(len(batch_IOs)):

                # Check if there are still non-terminals to be expanded in the frontier
                if frontiers[i]:
                    next_nt = frontiers[i].pop()

                    # Get the possible next rules for the current non-terminal
                    rules = self.get_next_rules(next_nt, depth)

                    # Create a mask to block invalid actions
                    mask = self.get_mask(rules)

                    # Apply the mask to logits
                    forward_logits[i] = forward_logits[i] - (1 - mask) * 100

                    # Sample an action (rule) based on the masked logits
                    cat = Categorical(logits=forward_logits[i])
                    action = cat.sample()

                    # Update cumulative log-probabilities and partition functions
                    total_forwards[i] += cat.log_prob(action).item()
                    final_logZs[i] = logZs[i]  # we do this to account for trajectory length,
                    # I want the last prediction of logZ

                    # Update the program and state based on the sampled action
                    rule = self.model.state_encoder.idx2rule[action.item()]
                    nt, program = rule
                    programs[i] = (program, programs[i])
                    states[i] = states[i] + [rule]

                    # Add the arguments of the chosen rule to the frontier for future expansion
                    program_args = self.data.cfg.rules[nt][program]
                    frontiers[i].extend(program_args)

        # Reconstruct the final programs from their compressed representations
        if not any(frontiers):
            reconstructed = [reconstruct_from_compressed(program, target_type=self.data.cfg.start[0])
                             for program in programs]
            return final_logZs, total_forwards, reconstructed, states












    ################################
    ################################
    ########## Replay ##############
    ################################
    ################################
    def replay(self, correct_programs):
        """
        We guide the model towards the correct trajectory,
        given that we have programs that solved the tasks correctly.
        """

        logging.info(f'\nReplay')

        # Prepare the batch from correct_programs
        batch_programs, correct_states_list, batch_IOs, _, _ = zip(*correct_programs)

        # TODO: make fantasy batch with correct program

        # Initialise container for forward logits accumulation
        total_forwards = torch.zeros(len(correct_programs), device=device)

        # Initial state
        states = [['START'] for _ in range(len(correct_programs))]

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

        replay_loss = -10 * total_forwards.mean()  # TODO: multiply with replay weight *10
        self.optimizer_policy.zero_grad()
        replay_loss.backward()
        self.optimizer_policy.step()  # (E-step sleep)













    ########################################
    ########################################
    ################ Fantasy ###############
    ########################################
    ########################################
    def fantasy(self, data, depth):

        logging.info(f'\nFantasy')

        # Prepare the batch from
        batch_programs, batch_states, batch_IOs, _, _ = data

        # Create a batch of inputs from empirical data and outputs from imagined programs
        batch_programs, batch_states, fantasy_batch = self.make_fantasy_batch(
            batch_programs, batch_states, batch_IOs, depth)

        # Initialise container for forward logits accumulation
        total_forwards = torch.zeros(self.batch_size, device=device)

        # Initial state
        states = [['START'] for _ in range(self.batch_size)]

        # Maximum trajectory length
        max_traj_length = max(len(traj) for traj in batch_states)

        # Process each step of each trajectory
        for t in range(max_traj_length - 1):  # adjusted for -1 as we're skipping the first token for prediction

            forward_logits, _ = self.model(states, fantasy_batch)

            # Create trajectories
            for i in range(self.batch_size):
                if len(batch_states[i]) - 1 >= t + 1:
                    rule = batch_states[i][t + 1]
                    correct_rule_idx = self.model.state_encoder.rule2idx[rule]
                    total_forwards[i] += torch.log(F.softmax(forward_logits[i], dim=0)[correct_rule_idx])

                    # Update the state for the next prediction
                    states[i].append(rule)

        # Update model
        fantasy_loss = -10 * total_forwards.mean()  # TODO: Try weighting it (e.g. -10 * ...)
        self.optimizer_policy.zero_grad()
        fantasy_loss.backward()
        self.optimizer_policy.step()  # (E-step sleep)

    def make_fantasy_batch(self, batch_programs, batch_states, batch_IOs, depth):
        new_batch_programs, new_batch_states, fantasy_batch = [], [], []

        def process_programs(programs, states, inputs):
            nonlocal new_batch_programs, new_batch_states, fantasy_batch

            for program, input_set, state in zip(programs, inputs, states):
                predicted_output = [program.eval_naive(self.data.dsl, [inp]) for inp in input_set]

                # NOTE: If the program is constant it means it doesn't depend on the input.
                # We are also checking for Nones, don't want any programs that produce them
                # Also, the output should be in the lexicon. This is not strictly necessary, the model would learn to
                # extrapolate without this constraint, but this complicates the model.
                if \
                        not program.is_constant() \
                        and predicted_output is not None \
                        and None not in predicted_output \
                        and not None in chain.from_iterable(predicted_output)\
                        and all(out in self.data.lexicon for sublist in predicted_output for out in sublist):

                    new_batch_programs.append(program)
                    new_batch_states.append(state)

                    IOs = [([I], O) for I, O in zip(input_set, predicted_output)]
                    fantasy_batch.append(IOs)

        # Use existing programs for initial batch
        sampled_inputs = [self.data.sample_input() for _ in range(self.batch_size)]
        process_programs(batch_programs, batch_states, sampled_inputs)

        # Resample for indices with None outputs until all are valid
        while (program_counter := self.batch_size - len(new_batch_programs)) != 0:
            inputs_for_resample = [self.data.sample_input() for _ in range(program_counter)]

            with torch.no_grad():
                _, _, reconstructed, new_states = self.sample_program_dfs(batch_IOs[:len(inputs_for_resample)], depth)

            process_programs(reconstructed, new_states, inputs_for_resample)

        return new_batch_programs, new_batch_states, fantasy_batch
























    #########################################
    #########################################
    ############## E-STEP ###################
    #########################################
    #########################################
    def e_step(self, epoch, depth, batch_IOs, task_names):
        logging.info(f'\nComputing E-step')
        solved = False
        e_step_losses = []
        e_step_logZs = []

        e_step_data = []

        # Unfreeze GFlowNet parameters and freeze transformer parameters for the E-step optimization
        # unfreeze_parameters(self.model.forward_logits)
        # freeze_parameters(self.model.transformer)

        e_step_tqdm = tqdm.tqdm(range(self.e_steps), position=0, desc='e_step', leave=False, colour='blue', ncols=80)
        for e_step in e_step_tqdm:

            # Predict programs and calculate associated log partition functions and other parameters
            # epsilon and beta for exploration
            logZs, total_forwards, programs, states = self.sample_program_dfs(batch_IOs, depth,
                                                                              epsilon=self.epsilon,
                                                                              beta=self.beta)

            # Calculate rewards for the predicted programs
            rewards_ = rewards(programs, batch_IOs, self.data.dsl, self.data.lexicon, self.max_reward)

            for program, reward in zip(programs, rewards_):
                e_step_tqdm.write(f'reward: {reward}, program: {program}')

            # Collect data for stats and training
            e_step_data.append((programs, states, batch_IOs, task_names, rewards_))

            # Compute the loss and perform backpropagation
            e_loss = (logZs + total_forwards - torch.log(rewards_).clip(-20)).pow(2)  # Trajectory Balance Loss
            e_loss = e_loss.mean()
            # TODO: Add e_step to loss, to prefer programs of MDL?

            # Update moving average of the GFlowNet's training loss
            self.moving_average_loss = e_loss.item() if self.moving_average_loss is None \
                else self.alpha * e_loss.item() + (1 - self.alpha) * self.moving_average_loss

            # Update policy parameters (E-step)
            self.optimizer_policy.zero_grad()
            e_loss.backward()
            self.optimizer_policy.step()

            # Record the loss and logZ
            e_step_losses.append(e_loss.item())
            e_step_logZs.append(logZs.mean().item())
            e_step_tqdm.set_postfix({'e_step loss': e_loss.item(), 'Z': logZs.mean().exp().item()})

            # Save results to csv
            if self.max_reward in rewards_:
                logging.info(f'Task: {task_names[0]} solved.')
                solved = True
                save_results('e-step', depth, epoch, e_step, task_names, programs, rewards_, batch_IOs, self.batch_size, self.max_reward, self.csv_file)

                replay_data = []
                for i in range(self.batch_size):
                    if rewards_[i] == self.max_reward:
                        replay_data.append((programs[i], states[i], batch_IOs[i], task_names[i], rewards_[i]))
                self.replay(replay_data)

                # If we solved it, continue with the next task.
                # Comment this return out if you want the chance for multiple solutions.
                # return e_step_data, e_step_losses, e_step_logZs, solved

            if random.random() < self.fantasy_prob:
                self.fantasy(e_step_data[-1], depth)

        # if not solved:
        #     logging.info(f'Task: {task_names[0]} not solved. :-(')
        #     epoch_data = [depth, epoch, '', task_names[0], '', batch_IOs[0]]
        #     append_to_csv(self.csv_file, epoch_data)``


        logging.debug(f'Finished e-step')
        return e_step_data, e_step_losses, e_step_logZs, solved






    #########################################
    #########################################
    ############## M-STEP ###################
    #########################################
    #########################################
    def m_step(self, epoch, depth, batch_IOs, task_names):
        solved = False
        m_step_losses = []
        m_step_data = []

        # unfreeze_parameters(self.model.transformer)  # Unfreeze transformer parameters
        # freeze_parameters(self.model.forward_logits)  # Freeze GFlowNet parameters


        m_step_tqdm = tqdm.tqdm(range(self.m_steps), leave=False)
        for m_step in m_step_tqdm:

            # Predict programs without specific parameters epsilon and beta
            _, _, programs, states = self.sample_program_dfs(batch_IOs, depth)

            # Calculate rewards for the predicted programs
            rewards_ = rewards(programs, batch_IOs, self.data.dsl, self.data.lexicon, self.max_reward)

            for program, reward in zip(programs, rewards_):
                m_step_tqdm.write(f'reward: {reward}, program: {program}')

            if self.max_reward in rewards_:
                logging.info(f'Task: {task_names[0]} solved.')
                solved = True
                save_results('m-step', depth, epoch, m_step, task_names, programs, rewards_, batch_IOs, self.batch_size, self.max_reward, self.csv_file)

            m_step_data.append((programs, states, batch_IOs, task_names, rewards_))

            # Compute the M-step loss, which measures how well the generative model is performing
            # The loss is based on the negative log of the rewards, and it's clipped to avoid numerical instability
            m_loss = -torch.log(rewards_).clip(-20).mean()

            # Perform backpropagation to calculate the gradients
            m_loss.backward()

            # Record the loss for later analysis
            m_step_losses.append(m_loss.item())

            # Update the generative model's parameters using the calculated gradients
            self.optimizer_generative.step()  # Update (M-step)

            # Reset gradients for the next iteration
            self.optimizer_generative.zero_grad()

            m_step_tqdm.set_postfix({'m_step loss': m_loss.item()})

        # if not solved:
        #     logging.info(f'Task: {task_names[0]} not solved. :-(')
        #     epoch_data = [depth, epoch, '', task_names[0], '', batch_IOs[0]]
        #     append_to_csv(self.csv_file, epoch_data)

        return m_step_data, m_step_losses









    #########################################
    #########################################
    ############### Train ###################
    #########################################
    #########################################
    def train(self):
        total_solved = 0
        start_time = time.time()  # Store the start time of training for performance analysis

        # Current threshold value, initialized to the initial threshold
        current_threshold = self.m_step_threshold_init

        total_data = []
        total_correct = []

        total_e_losses = []
        total_m_losses = []
        total_logZs = []

        for depth in tqdm.tqdm(range(self.min_program_depth, self.max_program_depth + 1), position=0, desc='depth',
                          leave=False, colour='green', ncols=80):
            for _ in tqdm.tqdm(range(self.n_tasks), position=0, desc='epoch', leave=False, colour='green', ncols=80):

                # Sample tasks from the real distribution and try to solve them (wake phase)
                batch_IOs, task_names = self.data.get_next_batch(self.batch_size)
                logging.info(f'Working on task: {task_names[0]}')
                logging.info(f'IOs: {batch_IOs[0]}')

                epoch_tqdm = tqdm.tqdm(range(self.epochs), position=0, desc='epoch', leave=False, colour='green', ncols=80)
                for epoch in epoch_tqdm:


                    # E-step
                    e_step_data, e_step_losses, e_step_logZs, solved = self.e_step(epoch, depth, batch_IOs, task_names)
                    total_solved += solved
                    total_e_losses.extend(e_step_losses)
                    total_logZs.extend(e_step_logZs)
                    e_step_correct = correct_programs(e_step_data)

                    total_data.extend(e_step_data)
                    total_correct.extend(e_step_correct)

                    # Replay (training on all correct task-program pairs)
                    if rand := random.random() <= self.replay_prob:
                        if total_correct:
                            for b in batch(total_correct, self.batch_size):
                                self.replay(b)

                    # Fantasy
                    if rand <= self.fantasy_prob:
                        self.fantasy(total_data[-1], depth)

                    # M-step
                    # Optimize Generative Model
                    # Check whether the moving average of the GFlowNet's training loss is below the current threshold
                    if self.moving_average_loss < current_threshold or epoch == self.epochs - 1:
                        m_step_data, m_step_losses = self.m_step(epoch, depth, batch_IOs, task_names)
                        total_m_losses.extend(m_step_losses)
                        total_correct.extend(correct_programs(m_step_data))
                        total_data.extend(m_step_data)
                    # Decrease the threshold value linearly over the epochs
                    current_threshold -= self.threshold_decrement

                    print_stats(start_time, total_correct, total_data, self.batch_size)
                    epoch_tqdm.set_postfix({'Solved': total_solved})

                    if self.save_checkpoint:
                        torch.save(self.model, TO_CHECKPOINT_PATH)

            plot_results(total_e_losses, total_m_losses, total_logZs, [], epoch, RESULTS)


    def inference(self):

        self.data.reset_task_generator()

        total_solved = 0
        start_time = time.time()  # Store the start time of training for performance analysis

        inference_data = []

        self.model.eval()

        for depth in tqdm.tqdm(range(self.min_program_depth, self.max_program_depth + 1), position=0, desc='depth',
                          leave=False, colour='green', ncols=80):
            task_tqdm = tqdm.tqdm(range(self.n_tasks), position=0, desc='epoch', leave=False, colour='green', ncols=80)
            for _ in task_tqdm:
                solved = False

                # Sample tasks from the real distribution and try to solve them (wake phase)
                batch_IOs, task_names = self.data.get_next_batch(self.batch_size)
                logging.info(f'Working on task: {task_names[0]}')
                logging.info(f'IOs: {batch_IOs[0]}')
                inference_tqdm = tqdm.tqdm(range(self.n_tasks), position=0, desc='inference steps', leave=False, colour='magenta', ncols=80)
                for step in inference_tqdm:

                    # Predict programs without specific parameters epsilon and beta
                    _, _, programs, states = self.sample_program_dfs(batch_IOs, depth)

                    # Calculate rewards for the predicted programs
                    rewards_ = rewards(programs, batch_IOs, self.data.dsl, self.data.lexicon, self.max_reward)

                    inference_data.append((programs, states, batch_IOs, task_names, rewards_))

                    if self.max_reward in rewards_:
                        logging.info(f'Task: {task_names[0]} solved.')
                        solved = True
                        save_results('inference', depth, '', step, task_names, programs, rewards_, batch_IOs, self.batch_size,
                                     self.max_reward, self.csv_file)

                if not solved:
                    logging.info(f'Task: {task_names[0]} not solved. :-(')
                    save_results('inference', depth, '', step, task_names, programs, rewards_, batch_IOs,
                                 self.batch_size, self.max_reward, self.csv_file)

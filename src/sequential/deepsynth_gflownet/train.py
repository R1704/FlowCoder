from src.sequential.deepsynth.dsl import *
from src.sequential.deepsynth.run_experiment import *
from src.sequential.deepsynth_gflownet.data import *

from torch.distributions.categorical import Categorical
from torch.optim import Adam

import matplotlib.pyplot as pp
from dataclasses import dataclass

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True


@dataclass
class Training:
    n_epochs: int
    batch_size: int
    learning_rate: float
    model_path: str
    data: Data

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
                            f'Total Rewards: {rewards} / {self.n_epochs} = {rewards / self.n_epochs}'
                        )

                # Keep digging
                else:
                    non_terminal = non_terminals.pop()

                    forward_logits, logZ = model(state, non_terminal, latent_batch_IOs)

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
    if rewrd.item() == 2:
        logging.info('-----found the correct program-----')
        logging.info(f'found program: {program}')
        logging.info(f'actual program: {batch_program[0]}')
        logging.info(f'reward: {rewrd.item()}')
    return rewrd

def make_program_checker(dsl: DSL, examples) -> Callable[[Program, bool], int]:
    correct_program_rwd = 10
    is_program_rwd = 1
    none_rwd = 0
    def checker(prog: Program, use_cached_evaluator: bool) -> int:
        if use_cached_evaluator:
            for i, example in enumerate(examples):
                input, output = example
                out = prog.eval(dsl, input, i)
                if out is None or None in out:
                    return none_rwd
                elif output != out:
                    return is_program_rwd
            return correct_program_rwd
        else:
            for example in examples:
                input, output = example
                out = prog.eval_naive(dsl, input)
                if out is None or None in out:
                    return none_rwd
                elif output != out:
                    return is_program_rwd
            return correct_program_rwd
    return checker

from src.sequential.deepsynth_gflownet.data import Data
from src.sequential.deepsynth.pcfg import PCFG
from src.sequential.deepsynth.run_experiment import *
from src.sequential.deepsynth_gflownet.data import *
from src.sequential.deepsynth_gflownet.state_encoder import *
from collections import deque
import torch

from src.sequential.deepsynth_gflownet.io_encoder import *


data = Data(
     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
     dataset_size=10_000,
     nb_examples_max=2,
     max_program_depth=2,
     nb_arguments_max=3,
     lexicon=[0, 1], # [x for x in range(-2, 2)], #[x for x in range(-30, 30)],
     size_max=3 # 10,
     )

def dfs(G : PCFG):
    '''
    A generator that enumerates all programs using a DFS.
    '''

    # We need to reverse the rules:
    new_rules = {}
    for S in G.rules:
        new_rules[S] = {}
        sorted_derivation_list = sorted(
            G.rules[S], key=lambda P: G.rules[S][P][1]
        )
        for P in sorted_derivation_list:
            new_rules[S][P] = G.rules[S][P]
    G = PCFG(start = G.start,
        rules = new_rules,
        max_program_depth = G.max_program_depth)

    frontier = deque()
    initial_non_terminals = deque()
    initial_non_terminals.append(G.start)
    frontier.append((None, initial_non_terminals))
    # A frontier is a queue of pairs (partial_program, non_terminals) describing a partial program:
    # partial_program is the list of primitives and variables describing the leftmost derivation, and
    # non_terminals is the queue of non-terminals appearing from left to right

    while len(frontier) != 0:
        partial_program, non_terminals = frontier.pop()
        print(f'partial program in dfs: {partial_program}')
        print(f'non_terminals in dfs: {non_terminals}')
        if len(non_terminals) == 0:
            yield partial_program
        else:
            S = non_terminals.pop()
            for P in G.rules[S]:
                args_P, w = G.rules[S][P]
                new_partial_program = (P, partial_program)
                new_non_terminals = non_terminals.copy()
                for arg in args_P:
                    new_non_terminals.append(arg)
                frontier.append((new_partial_program, new_non_terminals))


print(data.cfg)
rules = data.cfg.rules
S = data.cfg.start

first = rules[S]

print('-------Rules--------')
for rule, v in rules.items():
    print(rule, v)
print()

print('-------start and its programs--------')
print(S, first, '\n')

pcfg = data.cfg.CFG_to_Uniform_PCFG()
program_generator = dfs(pcfg)
partial_program = next(program_generator)
print(f'partial program: {partial_program}')

program = reconstruct_from_compressed(partial_program, target_type=data.cfg.start[0])
print(f'program: {program}')

print('-----------------------------')


state_encoder = StateEncoder(cfg=data.cfg)
print(state_encoder.rules)

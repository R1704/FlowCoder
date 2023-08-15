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
     max_program_depth=3,
     nb_arguments_max=3,
     lexicon=[0, 1], # [x for x in range(-2, 2)], #[x for x in range(-30, 30)],
     size_max=3 # 10,
     )


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
program_generator_dfs = dfs(pcfg)
program_generator_bfs = bfs(pcfg)
partial_program = next(program_generator_dfs)
print(f'partial program: {partial_program}')

program = reconstruct_from_compressed(partial_program, target_type=data.cfg.start[0])
print(f'program: {program}')
print(type(program))
print([type(a) for a in program.arguments])

print('-----------------------------')


state_encoder = RuleEncoder(cfg=data.cfg)
print(state_encoder.rules)
print('-----------------------------')

def get_neighbors(rule):
     neighbors = []
     (my_typ, (my_p, my_arg_idx), my_depth), my_prog = rule
     for r in state_encoder.rules[2:]:
          if r[0][1] != None:
               (typ, (p, arg_idx), depth), prog = r
               if my_p == p and my_depth == depth:
                    neighbors.append(r)
     return neighbors




r = state_encoder.rules[-3]
print(f'r: {r}')

# neighbors = get_neighbors(r)
# print(f'neighbors: {neighbors}')

parents = state_encoder.get_parent_rule(r)
print(f'parents: {parents}')
parent_rule = parents[0]
print(f'parent_rule: {parent_rule}')
parent_args = state_encoder.get_parent_args(parent_rule)
print(f'parent_args: {parent_args}')
parents = state_encoder.get_parent_rule(parent_rule)
print(f'parents: {parents}')

ls = [[1,2,3], [12,3], [2,3,4,5 ]]
print(len(max(ls, key=len)))
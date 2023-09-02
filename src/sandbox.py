from src.sequential.deepsynth_gflownet.data import Data
from src.sequential.deepsynth.pcfg import PCFG
from src.sequential.deepsynth.run_experiment import *
from src.sequential.deepsynth_gflownet.data import *
from src.sequential.deepsynth_gflownet.state_encoder import *
from src.sequential.deepsynth_gflownet.reward import *
from src.sequential.deepsynth_gflownet.config import *
from src.sequential.deepsynth.dreamcoder_dataset_loader import *
from collections import deque
import torch
from src.sequential.deepsynth.list_dataset import *
import pickle


from src.sequential.deepsynth_gflownet.io_encoder import *




data = Data(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

names, examples = data.get_next_batch(8)
# print(names)
# print()
# print(examples)

#
#
# print(data.cfg)
# rules = data.cfg.rules
# S = data.cfg.start
#
# first = rules[S]
#
# print('-------Rules--------')
# for rule, v in rules.items():
#     print(rule, v)
# print()
#
# print('-------start and its programs--------')
# print(S, first, '\n')
#
# pcfg = data.cfg.CFG_to_Uniform_PCFG()
# program_generator_dfs = dfs(pcfg)
# program_generator_bfs = bfs(pcfg)
# partial_program = next(program_generator_dfs)
# print(f'partial program: {partial_program}')
#
# program = reconstruct_from_compressed(partial_program, target_type=data.cfg.start[0])
# print(f'program: {program}')
# print(type(program))
# print([type(a) for a in program.arguments])
#
# print('-----------------------------')
#
#
# # state_encoder = RuleEncoder(cfg=data.cfg)
# # print(state_encoder.rules)
# print('-----------------------------')


# batch_IOs, batch_program = data.get_next_batch(1)
# print(batch_IOs, batch_program)



# folder = '/vol/tensusers4/rhommelsheim/master_thesis/src/sequential/deepsynth/list_dataset'
# tasks = load_tasks(folder)
#
# for task in tasks:
#      name, example = task
#      print(name)
#      for i, o in example:
#           print(i, o)
from typing import Tuple
import torch
import torch.nn as nn

from gfn.gflownet import SubTBGFlowNet
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.samplers import Sampler
from gfn.utils.modules import MLP
from gfn.env import DiscreteEnv
from gfn.actions import Actions   # Updated import for actions
from gfn.states import States     # Updated import for states
import random


import deepsynth.dsl as dsl
from deepsynth.DSL.list import semantics, primitive_types
from deepsynth.model_loader import build_dreamcoder_intlist_model
from deepsynth.dreamcoder_dataset_loader import load_tasks, filter_tasks_for_model
from deepsynth.type_system import Arrow, List, INT

from flowcoder.config import *
from flowcoder.flowcoder_torchgfn.graph_attention import GraphAttentionNetwork  # New import for GAT


def load_dreamcoder_tasks():
    # Load tasks
    tasks = load_tasks(DREAMCODER_DATASET_PATH)
    print("Loaded", len(tasks), "tasks")

    # Filter tasks
    _, _, rules_predictor = build_dreamcoder_intlist_model(max_program_depth=max_program_depth)
    tasks = filter_tasks_for_model(tasks, rules_predictor)
    with open("src/task_names.txt", "r") as file:
        task_names = file.read().splitlines()
    # Create a dictionary to map task names to their corresponding tasks
    task_dict = {name: task for name, task in tasks}
    # Filter the tasks based on whether they are in the task_names list and maintain the order
    tasks = [(name, task_dict[name]) for name in task_names if name in task_dict]
    print("Remaining tasks after filter:", len(tasks), "tasks")

    # Format tasks array
    dataset_size = len(tasks)
    all_tasks = []
    for name, examples in tasks:
        ex = [([i[0]], o) for i, o in examples]
        all_tasks.append((name, ex))
    return all_tasks, dataset_size


max_program_depth = 6
dsl_obj = dsl.DSL(semantics, primitive_types)

print(dsl_obj)

cfg = dsl_obj.DSL_to_CFG(Arrow(List(INT), List(INT)), max_program_depth=max_program_depth)

# create random programs
pcfg = cfg.CFG_to_Random_PCFG(alpha=0.5)
for program in pcfg.sampling():
    print(program)
    break


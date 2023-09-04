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
from src.sequential.deepsynth.cons_list import tuple2constlist



dsl = dsl.DSL(semantics, primitive_types)
type_request = Arrow(List(INT), List(INT))

cfg = dsl.DSL_to_CFG(type_request, max_program_depth=4)
dataset = Dataset(
    size=10,
    dsl=dsl,
    pcfg_dict={type_request: cfg.CFG_to_Uniform_PCFG()},
    nb_examples_max=15,
    arguments={type_request: type_request.arguments()},
    ProgramEncoder=lambda x: x,
    size_max=10,
    lexicon=[x for x in range(-30, 30)],
    for_flashfill=False
    )

# print(dataset.allowed_types)
# print(type_request)
# print(dataset.arguments)

nb_IOs = random.randint(1, dataset.nb_examples_max)
inputs = [[dataset.input_sampler.sample(type_) for type_ in dataset.arguments[type_request]] for _ in
                      range(nb_IOs)]

print(inputs)

outputs = []
for input_ in inputs:
    environment = tuple2constlist(input_)
    print(environment)
    # output = program.eval_naive(self.dsl, environment)
    # if self.__output_validation__(output, rtype):
    #     outputs.append(output)
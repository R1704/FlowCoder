from src.sequential.deepsynth.dsl import *
from src.sequential.deepsynth.DSL.list import *
from src.sequential.deepsynth.run_experiment import *
from src.sequential.deepsynth.experiment_helper import *
from src.sequential.deepsynth.type_system import *
from src.sequential.deepsynth.pcfg import *
from src.sequential.deepsynth.Predictions.dataset_sampler import Dataset
from src.sequential.deepsynth.Predictions.IOencodings import FixedSizeEncoding
from src.sequential.deepsynth.model_loader import __buildintlist_model
from src.sequential.deepsynth_gflownet.model import *

import collections

# DSL
dsl = DSL(semantics=semantics, primitive_types=primitive_types)
# print(dsl)


# CFG
# type_request = Arrow(List(INT), INT)
# cfg = dsl.DSL_to_CFG(type_request,  max_program_depth=4, min_variable_depth=1, upper_bound_type_size=10, n_gram=2)

# print(cfg.start)

# state = cfg.start
# print(state)
# actions = cfg.rules[state]
# print(actions)

dataset_size: int = 10_000
nb_examples_max = 2
max_program_depth = 4
nb_arguments_max = 1
lexicon = [x for x in range(-30, 30)]  # all elements of a list must be from lexicon
size_max = 10  # maximum number of elements in a list (input or output)
embedding_output_dimension = 10
# only useful for RNNEmbedding
number_layers_RNN = 1
size_hidden = 64

cfg, model_dummy = __buildintlist_model(
    dsl,
    max_program_depth,
    nb_arguments_max,
    lexicon,
    size_max,
    size_hidden,
    embedding_output_dimension,
    number_layers_RNN
)
type_request = Arrow(List(INT), List(INT))

model = GFlowNet(
    cfg=cfg,
    IOEncoder=model_dummy.IOEncoder,
    IOEmbedder=model_dummy.IOEmbedder,
    latent_encoder=model_dummy.latent_encoder,
    primitive_types=primitive_types,
    d_model=512,
    num_heads=8,
    num_layers=2
)

dataset = Dataset(
    size=dataset_size,
    dsl=dsl,
    pcfg_dict={type_request: cfg.CFG_to_Uniform_PCFG()},
    nb_examples_max=nb_examples_max,
    arguments={type_request: type_request.arguments()},
    ProgramEncoder=model.ProgramEncoder,
    size_max=model.IOEncoder.size_max,
    lexicon=model.IOEncoder.lexicon[:-2],
    for_flashfill=False
)
# gen = dataset.__iter__()
# state = []
#
# io, prog, _, req = next(gen)
#
# S = cfg.start  # state is a non-terminal
# state.append(S)
#
#
# # calculate the forward and backward logits
# forward_logits, backward_logits = model(state, [io])
#

# S = cfg.start
# Ps = cfg.rules[S]
# print(Ps)
# P = random.sample(list(Ps.keys()), 1)[0]
# print(P)
# args_P = cfg.rules[S][P]
# print(args_P)
# arguments = []
# for arg in args_P:
#     print(cfg.rules[arg])
#     arguments.append(arg)
# print(arguments)



S = cfg.start  # state is a non-terminal
def dfs():
    state = []
    S = cfg.start  # state is a non-terminal
    # keep sampling until we have a complete program
    frontier = deque()
    initial_non_terminals = deque()
    initial_non_terminals.append(S)
    frontier.append((None, initial_non_terminals))
    # A frontier is a queue of pairs (partial_program, non_terminals) describing a partial program:
    # partial_program is the list of primitives and variables describing the leftmost derivation, and
    # non_terminals is the queue of non-terminals appearing from left to right

    while len(frontier) != 0:
        partial_program, non_terminals = frontier.pop()
        if len(non_terminals) == 0:
            print('state: ', state)
            yield partial_program
        else:
            S = non_terminals.pop()
            # here we call the model to ask for the action
            P = random.sample(list(cfg.rules[S].keys()), 1)[0]
            state = state + [P]
            args_P = cfg.rules[S][P]
            new_partial_program = (P, partial_program)
            # print(new_partial_program)
            new_non_terminals = non_terminals.copy()
            for arg in args_P:
                new_non_terminals.append(arg)
            frontier.append((new_partial_program, new_non_terminals))

def reconstruct_from_compressed(program, target_type):
    program_as_list = []
    list_from_compressed(program, program_as_list)
    program_as_list.reverse()
    return reconstruct_from_list(program_as_list, target_type)

def list_from_compressed(program, program_as_list=None):
    (P, sub_program) = program
    if sub_program:
        list_from_compressed(sub_program, program_as_list)
    program_as_list.append(P)

def reconstruct_from_list(program_as_list, target_type):
    if len(program_as_list) == 1:
        return program_as_list.pop()
    else:
        P = program_as_list.pop()
        if isinstance(P, (New, BasicPrimitive)):
            list_arguments = P.type.ends_with(target_type)
            arguments = [None] * len(list_arguments)
            for i in range(len(list_arguments)):
                arguments[len(list_arguments) - i - 1] = reconstruct_from_list(
                    program_as_list, list_arguments[len(list_arguments) - i - 1]
                )
            return Function(P, arguments)
        if isinstance(P, Variable):
            return P
        assert False


gen = dataset.__iter__()

io, prog, _, req = next(gen)

dfs_gen = dfs()
prog1 = next(dfs_gen)

print(prog1)
rec = reconstruct_from_compressed(prog1, target_type=cfg.start[0])
print(rec)


# program_checker = make_program_checker(dsl, io)
# for example in io:
#     input, output = example
#     out = prog.eval_naive(dsl, input)
#     print(out)
# res = program_checker(rec, True)
# print(res)

# ev = rec.eval_naive(dsl, 4)
# print(ev)


# print(cfg.rules[cfg.start])

# list_derivations = {}
# for S in cfg.rules:
#     list_derivations[S] = sorted(cfg.rules[S], key=lambda P: cfg.rules[S][P])
#
# for k, v in list_derivations.items():
#     print(k, v)

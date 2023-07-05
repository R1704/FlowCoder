import src.sequential.deepsynth.dsl as dsl
from src.sequential.deepsynth.dsl import *
from src.sequential.deepsynth.DSL.list import *
from src.sequential.deepsynth.DSL import list as list_
from src.sequential.deepsynth.dsl import *
from src.sequential.deepsynth.run_experiment import *
from src.sequential.deepsynth.pcfg import *
from src.sequential.deepsynth.Predictions.dataset_sampler import Dataset
from src.sequential.deepsynth.model_loader import __buildintlist_model
from src.sequential.deepsynth_gflownet.model import *
from src.sequential.deepsynth.experiment_helper import *
from src.sequential.deepsynth.type_system import *
from src.sequential.deepsynth.Predictions.embeddings import RNNEmbedding
from src.sequential.deepsynth.Predictions.IOencodings import FixedSizeEncoding
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

    def train(self, model, cfg, batch_generator, device):
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        losses = []
        logZs = []

        for epoch in tqdm.tqdm(range(self.n_epochs), ncols=40):
            batch_IOs, batch_program, latent_batch_IOs = get_next_batch(batch_generator, self.batch_size)

            state = []  # start with an empty state
            total_forward = 0
            non_terminal = cfg.start  # start with the CFGs start symbol

            # keep sampling until we have a complete program
            frontier = deque()
            initial_non_terminals = deque()
            initial_non_terminals.append(non_terminal)
            frontier.append((None, initial_non_terminals))
            # A frontier is a queue of pairs (partial_program, non_terminals) describing a partial program:
            # partial_program is the list of primitives and variables describing the leftmost derivation, and
            # non_terminals is the queue of non-terminals appearing from left to right

            while len(frontier) != 0:
                partial_program, non_terminals = frontier.pop()

                # If we are finished with the trajectory/ have a constructed program
                if len(non_terminals) == 0:
                    program = reconstruct_from_compressed(partial_program, target_type=cfg.start[0])
                    reward = Reward(program, batch_program, batch_IOs)

                    # Compute loss and backpropagate
                    loss = (logZ + total_forward - torch.log(reward).clip(-20)).pow(2)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    losses.append(loss)
                    logZs.append(logZ)

                    if epoch % 100 == 0:
                        logging.info(
                            f'Epoch: {epoch}\nLoss: {loss}\nLogZ: {logZ}\nforward: {total_forward}\nreward: {torch.log(reward).clip(-20)}')

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




def get_next_batch(gen, batch_size):
    batch_IOs, batch_program, batch_requests = [], [], []
    for _ in range(batch_size):
        io, prog, _, req = next(gen)
        batch_IOs.append(io)
        batch_program.append(prog)
        batch_requests.append(req)

    embedded = IOEmbedder.forward(batch_IOs)
    latent = latent_encoder(embedded)
    return batch_IOs, batch_program, latent.to(device)

def Reward(program: Program, batch_program, task):
    program_checker = make_program_checker(dsl, task[0])
    # logging.debug(f'found program: {program}')
    # logging.debug(f'actual program: {batch_program[0]}')

    rewrd = torch.tensor(float(program_checker(program, True)))
    logging.debug(f'rew: {torch.log(rewrd)}')
    return rewrd




dsl = dsl.DSL(semantics, primitive_types)
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

IOEncoder = FixedSizeEncoding(
    nb_arguments_max=nb_arguments_max,
    lexicon=lexicon,
    size_max=size_max,
)

IOEmbedder = RNNEmbedding(
    IOEncoder=IOEncoder,
    output_dimension=embedding_output_dimension,
    size_hidden=size_hidden,
    number_layers_RNN=number_layers_RNN,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def __block__(input_dim, output_dimension, activation):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dimension),
        activation,
    )

latent_encoder = torch.nn.Sequential(
        __block__(IOEncoder.output_dimension * IOEmbedder.output_dimension, size_hidden, torch.nn.Sigmoid()),
        __block__(size_hidden, size_hidden, torch.nn.Sigmoid()),
    )

model = GFlowNet(
    device=device,
    cfg=cfg,
    d_model=512 + 64,
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
    size_max=IOEncoder.size_max,
    lexicon=IOEncoder.lexicon[:-2],
    for_flashfill=False
)

batch_generator = dataset.__iter__()
model_path = 'gflownet.pth'
model.to(device)
# Save model
torch.save(model.state_dict(), model_path)

model.load_state_dict(torch.load(model_path))

# For inference
# model.eval()


training = Training(n_epochs=2000, batch_size=1, learning_rate=3e-4, model_path=model_path)
training.train(model, cfg, batch_generator, device)
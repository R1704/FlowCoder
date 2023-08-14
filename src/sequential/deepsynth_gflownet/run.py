from src.sequential.deepsynth_gflownet.model import GFlowNet
from src.sequential.deepsynth_gflownet.data import Data
from src.sequential.deepsynth_gflownet.train import Training
from src.sequential.deepsynth_gflownet.reward import Reward

import torch

import logging
from src.sequential.deepsynth_gflownet.io_encoder import *
from src.sequential.deepsynth_gflownet.state_encoder import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'gflownet.pth'

from_checkpoint = False
save_checkpoint = False
train = True
inference = False

data = Data(
     device=device,
     dataset_size=1_000_000,
     nb_examples_max=2,
     max_program_depth=2,
     # max_program_depth=4,
     nb_arguments_max=3,
     # lexicon=[0, 1], # [x for x in range(-2, 2)], #[x for x in range(-30, 30)],
     lexicon=[x for x in range(-30, 30)],
     size_max=3,
     # size_max=10,
     )

io_encoder = IOEncoder(
    n_examples_max=data.nb_examples_max,
    size_max=data.size_max,
    lexicon=data.lexicon,
    d_model=512,  # TODO: try different dimensions
    device=device
    )

state_encoder = RuleEncoder(
    cfg=data.cfg,
    d_model=512,  # TODO: try different dimensions
    device=device
    )

model = GFlowNet(
    cfg=data.cfg,
    io_encoder=io_encoder,
    state_encoder=state_encoder,
    d_model=512,
    num_heads=8,
    num_layers=2,
    dropout=0.1,
    device=device
    )

if from_checkpoint:
    print('Model parameters loaded from checkpoint.')
    model.load_state_dict(torch.load(model_path))

model.to(device)

reward = Reward(
    vocab_size=10*len(data.lexicon),
    d_model=512,  # TODO: try different dimensions
    num_heads=8,
    num_layers=2,
    dropout=0.1
    )

if train:
    training = Training(
        batch_size=32,
        learning_rate_trn=1e-4, #1e-3,
        learning_rate_gfn=1e-4,
        e_steps=10,
        m_step_threshold=10,
        m_steps=10,
        model_path=model_path,
        data=data,
        model=model,
        reward=reward,
        device=device
    )
    training.train()

if save_checkpoint:
    print('Model parameters saved to checkpoint.')
    torch.save(model.state_dict(), model_path)

if inference:
    model.eval()
    batch_IOs, batch_program, latent_batch_IOs = data.get_next_batch(model.batch_size)
    model()

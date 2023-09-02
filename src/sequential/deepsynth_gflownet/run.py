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

lexicon_range = 30
d_model = 512

# TODO: Try different model sizes

data = Data(device=device, max_program_depth=4)

io_encoder = IOEncoder(
    n_examples_max=data.nb_examples_max,
    size_max=10,
    lexicon=[x for x in range(-lexicon_range*10, lexicon_range*10)],  # For leverage in the fantasy phase
    d_model=d_model,
    device=device
    )

state_encoder = RuleEncoder(
    cfg=data.cfg,
    d_model=d_model,
    device=device
    )

model = GFlowNet(
    cfg=data.cfg,
    io_encoder=io_encoder,
    state_encoder=state_encoder,
    d_model=d_model,
    num_heads=8,
    num_layers=2,
    dropout=0.1,
    device=device
    )

if from_checkpoint:
    print('Model parameters loaded from checkpoint.')
    model.load_state_dict(torch.load(model_path))

model.to(device)


if train:
    training = Training(
        min_program_depth=3,
        epochs=145,
        batch_size=4,
        learning_rate_trn=1e-4,
        learning_rate_gfn=1e-4,
        e_steps=500,
        m_step_threshold_init=150,
        m_steps=150,
        alpha=0.3,
        replay_prob=0.3,
        fantasy_prob=0.3,
        model_path=model_path,
        data=data,
        model=model,
        device=device
    )
    training.train()

if save_checkpoint:
    print('Model parameters saved to checkpoint.')
    torch.save(model.state_dict(), model_path)

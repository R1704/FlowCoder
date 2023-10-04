import sys
sys.path.append('/vol/tensusers4/rhommelsheim/master_thesis/src')

from flowcoder.model import GFlowNet
from flowcoder.data import Data
from flowcoder.train import Training
from flowcoder.io_encoder import IOEncoder
from flowcoder.state_encoder import RuleEncoder
from flowcoder.config import *

import torch
from torch.optim import Adam
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

from_checkpoint = True
save_checkpoint = False
train = True
inference = True

d_model = 512

# TODO: Try different model sizes

data = Data(max_program_depth=6)

io_encoder = IOEncoder(
    n_examples_max=data.nb_examples_max,
    size_max=10,
    lexicon=data.lexicon,
    d_model=d_model
    )

state_encoder = RuleEncoder(
    cfg=data.cfg,
    d_model=d_model
    )

model = GFlowNet(
    cfg=data.cfg,
    io_encoder=io_encoder,
    state_encoder=state_encoder,
    d_model=d_model,
    num_heads=8,
    num_layers=2,
    dropout=0.1
    )
learning_rate = 0.001
optimizer = Adam(model.parameters(), lr=learning_rate)

if from_checkpoint:
    checkpoint = torch.load(FROM_CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f'Model parameters loaded from checkpoint in {FROM_CHECKPOINT_PATH}.')

model.to(device)
if inference:
    model.eval()

training = Training(
    min_program_depth=5,
    max_program_depth=data.max_program_depth,
    epochs=145,
    batch_size=4,
    learning_rate_trn=1e-4,
    learning_rate_gfn=1e-4,
    e_steps=500,
    m_step_threshold_init=150,
    m_steps=150,
    alpha=0.3,
    beta=0.7,
    epsilon=0.3,
    replay_prob=0.3,
    fantasy_prob=0.005,
    data=data,
    model=model,
    optimizer=optimizer,
    save_checkpoint=save_checkpoint,
    )

training.train()



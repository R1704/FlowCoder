import sys
sys.path.append('/vol/tensusers4/rhommelsheim/master_thesis/src')

from flowcoder.model import GFlowNet
from flowcoder.data import Data
from flowcoder.train import Training
from flowcoder.io_encoder import IOEncoder
from flowcoder.state_encoder import RuleEncoder
from flowcoder.config import *

import torch
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True

from_checkpoint = False
save_checkpoint = True
train = True
inference = True


d_model = 512

data = Data(max_program_depth=3, shuffle=False)

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

if from_checkpoint:
    model = torch.load(FROM_CHECKPOINT_PATH)
    logging.info(f'Model parameters loaded from checkpoint in {FROM_CHECKPOINT_PATH}.')

model.to(device)

training = Training(
    min_program_depth=3,
    max_program_depth=data.max_program_depth,
    n_tasks=min(1, 145), # This is the amount of tasks we want to solve from the dreamcoder dataset (max is 145)
    epochs=2,
    batch_size=4,
    learning_rate_gen=1e-4,
    learning_rate_pol=1e-4,
    e_steps=500,
    m_step_threshold_init=150,
    m_steps=500,
    inference_steps=2500,
    alpha=0.3,
    beta=0.7,
    epsilon=0.3,
    replay_prob=1, #0.2,
    fantasy_prob=1, #0.2, # TODO: maybe a bit less of that, innit?!
    data=data,
    model=model,
    save_checkpoint=save_checkpoint,
    )

if train:
    training.train()
if inference:
    training.inference()

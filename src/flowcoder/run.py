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

from_checkpoint = True
save_checkpoint = False
train = False
inference = True


d_model = 512

data = Data(
    max_program_depth=3,
    shuffle_tasks=False,
    n_tasks=95,  # if variable_batch is true, make sure you have enough tasks for the batch_size
    variable_batch=False,  # if False, all tasks in the batch will be the same
    train_ratio=0.5,
    seed=42
    )

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
    min_program_depth=data.max_program_depth,  # change this for gradual learning
    max_program_depth=data.max_program_depth,
    epochs=5,
    batch_size=4,  # if data.variable_batch is True, this should be a divisor of data.n_tasks
    learning_rate_gen=1e-4,
    learning_rate_pol=1e-4,
    e_steps=2000,
    m_step_threshold_init=150,
    m_steps=2000,
    inference_steps=100,
    alpha=0.3,
    beta=0.7,
    gamma=10.,
    epsilon=0.3,
    replay_prob=0.3,
    fantasy_prob=1,
    data=data,
    model=model,
    save_checkpoint=save_checkpoint,
    )

if train:
    model.train()
    training.train()

if inference:
    model.eval()
    training.inference()

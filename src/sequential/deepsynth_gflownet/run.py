from src.sequential.deepsynth_gflownet.model import GFlowNet
from src.sequential.deepsynth_gflownet.data import Data
from src.sequential.deepsynth_gflownet.train import Training
from src.sequential.deepsynth_gflownet.reward import Reward

import torch

import logging
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
     dataset_size=10_000,
     nb_examples_max=2,
     max_program_depth=2, #4,
     nb_arguments_max=1,
     lexicon=[0, 1], # [x for x in range(-2, 2)], #[x for x in range(-30, 30)],
     size_max=3, # 10,
     embedding_output_dimension=10,
     number_layers_RNN=1,
     size_hidden=64
     )

model = GFlowNet(
    device=device,
    cfg=data.cfg,
    d_model=512,
    io_dim=data.size_hidden,
    num_heads=8,
    num_layers=2
)

if from_checkpoint:
    print('Model parameters loaded from checkpoint.')
    model.load_state_dict(torch.load(model_path))

model.to(device)

reward = Reward(
    vocab_size=10*len(data.lexicon),
    d_model=512,
    num_heads=8,
    num_layers=2,
    dropout=0.1
)

if train:
    training = Training(
        n_epochs=min(1, data.dataset_size),
        batch_size=2,
        learning_rate=3e-4,
        e_steps=10,
        m_step_threshold=2,
        m_steps=1,
        model_path=model_path,
        data=data,
        model=model,
        reward=reward
    )
    training.train()

if save_checkpoint:
    print('Model parameters saved to checkpoint.')
    torch.save(model.state_dict(), model_path)

if inference:
    model.eval()
    batch_IOs, batch_program, latent_batch_IOs = data.get_next_batch(model.batch_size)
    model()

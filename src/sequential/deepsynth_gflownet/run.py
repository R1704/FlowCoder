from src.sequential.deepsynth_gflownet.model import GFlowNet
from src.sequential.deepsynth_gflownet.dataset import Data
from src.sequential.deepsynth_gflownet.train import Training

import torch


model_path = 'gflownet.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data = Data(
     device=device,
     dataset_size=10000,
     nb_examples_max=2,
     max_program_depth=4,
     nb_arguments_max=1,
     lexicon=[x for x in range(-30, 30)],
     size_max=10,
     embedding_output_dimension=10,
     number_layers_RNN=1,
     size_hidden=64
     )

model = GFlowNet(
    device=device,
    cfg=data.cfg,
    d_model=512 + 64,
    num_heads=8,
    num_layers=2
)
model.to(device)


training = Training(
    n_epochs=2000,
    batch_size=1,
    learning_rate=3e-4,
    model_path=model_path,
    data=data
)
training.train(model, data.cfg)



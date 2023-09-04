import os
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DREAMCODER_DATASET_PATH = '/vol/tensusers4/rhommelsheim/master_thesis/src/sequential/deepsynth/list_dataset'
RESULTS = '/vol/tensusers4/rhommelsheim/master_thesis/src/results'
CHECKPOINT = '/vol/tensusers4/rhommelsheim/master_thesis/src/checkpoints'
MODEL_PATH = os.path.join(CHECKPOINT, f'gflownet.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from flowcoder.utils import *
import os
import torch


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DREAMCODER_DATASET_PATH = '/vol/tensusers4/rhommelsheim/master_thesis/src/deepsynth/list_dataset'
RESULTS = '/vol/tensusers4/rhommelsheim/master_thesis/results'
CHECKPOINT = '/vol/tensusers4/rhommelsheim/master_thesis/checkpoints'
FROM_CHECKPOINT_PATH = os.path.join(CHECKPOINT, 'depth_3_all.pth')
    # get_checkpoint_filename(CHECKPOINT, find_last=True)
# FROM_CHECKPOINT_PATH = get_checkpoint_filename(CHECKPOINT, find_last=True)
TO_CHECKPOINT_PATH = FROM_CHECKPOINT_PATH
# TO_CHECKPOINT_PATH = get_checkpoint_filename(CHECKPOINT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

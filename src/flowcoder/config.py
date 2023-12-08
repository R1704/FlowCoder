import os
import torch
import datetime


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DREAMCODER_DATASET_PATH = '/vol/tensusers4/rhommelsheim/master_thesis/src/deepsynth/list_dataset'
RESULTS = '/vol/tensusers4/rhommelsheim/master_thesis/results'
CHECKPOINT = '/vol/tensusers4/rhommelsheim/master_thesis/checkpoints'

EXPERIMENT_NAME = 'depth_3_48_tasks'
EXPERIMENT_NAME += str(datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
FROM_CHECKPOINT_PATH = os.path.join(CHECKPOINT, EXPERIMENT_NAME+'.pth')
TO_CHECKPOINT_PATH = FROM_CHECKPOINT_PATH
CSV_FILENAME = os.path.join(RESULTS, EXPERIMENT_NAME+'.csv')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

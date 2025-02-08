import os
import torch
import datetime


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DREAMCODER_DATASET_PATH = 'src/deepsynth/list_dataset'
RESULTS = 'results'
CHECKPOINT = 'checkpoints'

EXPERIMENT_NAME = ''
FROM_CHECKPOINT_PATH = os.path.join(CHECKPOINT, EXPERIMENT_NAME+'.pth')
# EXPERIMENT_NAME += str(datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
TO_CHECKPOINT_PATH = FROM_CHECKPOINT_PATH
CSV_FILENAME = os.path.join(RESULTS, EXPERIMENT_NAME+'_inference.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

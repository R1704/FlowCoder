import os
import torch

# Define the path to save the model weights
current_directory = os.path.dirname(os.path.abspath(__file__))
checkpoints_path = os.path.join(current_directory, 'checkpoints')
model_weights_path = os.path.join(checkpoints_path, 'model_weights.pth')

grammar_path = os.path.join(current_directory, 'grammar.pkl')

num_rounds = 6
epochs = 5000
batch_size = 8
max_trajectory = 7
min_trajectory = 2
display = True
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


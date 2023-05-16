import os
import torch

# Define the path to save the model weights
current_directory = os.path.dirname(os.path.abspath(__file__))
checkpoints_path = os.path.join(current_directory, 'checkpoints')
model_weights_path = os.path.join(checkpoints_path, 'model_weights.pth')

grammar_path = os.path.join(current_directory, 'grammar.pkl')

num_rounds = 5
epochs = 100
batch_size = 8
max_trajectory = 8
min_trajectory = 2
display = True


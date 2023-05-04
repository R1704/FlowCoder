import os

# Define the path to save the model weights
current_directory = os.path.dirname(os.path.abspath(__file__))
checkpoints_path = os.path.join(current_directory, 'checkpoints')
model_weights_path = os.path.join(checkpoints_path, 'model_weights.pth')

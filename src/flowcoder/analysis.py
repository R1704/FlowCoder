from flowcoder.data import Data
from flowcoder.config import *
from flowcoder.io_encoder import IOEncoder

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE



from_checkpoint = False



# Get the task data
data = Data(
    max_program_depth=3,
    shuffle_tasks=False,
    n_tasks=145,  # if variable_batch is true, make sure you have enough tasks for the batch_size
    variable_batch=False,  # if False, all tasks in the batch will be the same
    train_ratio=1,
    seed=3
    )

# We only want to embed solved task embeddings
df = pd.read_csv('/vol/tensusers4/rhommelsheim/master_thesis/results/stats_2023-12-04 19:37:08.204531.csv')
solved_tasks = df[df['Solved'] == True]['Task Name'].unique()

# We need this to preserve order
task_names = []

if from_checkpoint:
    checkpoint = '/vol/tensusers4/rhommelsheim/master_thesis/checkpoints/gflownet_0.pth'
    model = torch.load(checkpoint)
    io_encoder = model.io_encoder
else:
    io_encoder = IOEncoder(
        n_examples_max=data.nb_examples_max,
        size_max=10,
        lexicon=data.lexicon,
        d_model=512
        )
io_encoder.to(device)

embeddings_list = []

# Extract and flatten embeddings for each task
with torch.no_grad():
    for i in range(data.n_tasks):
        task, task_name = data.get_io_batch(1, all_tasks=True)
        if task_name in solved_tasks:
            task_names.append(task_name)
            embedding = io_encoder(task)

            # Flatten the embedding to 1D
            embedding_flat = embedding.view(embedding.size(0), -1)

            embeddings_list.append(embedding_flat)

# Stack all flattened embeddings into a single tensor
all_embeddings = torch.vstack(embeddings_list)  # Use vstack to handle different shapes

# Convert to NumPy
all_embeddings_np = all_embeddings.cpu().numpy()

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=50, n_iter=1000, random_state=0)
embeddings_2d = tsne.fit_transform(all_embeddings_np)

# Plot
plt.figure(figsize=(10, 6))

# This is assuming `solved_tasks` is in the same order as `embeddings_2d`
for i, task_name in enumerate(task_names):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
    plt.annotate(task_name, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
plt.axis('off')
plt.title("t-SNE Visualization of Solved Task Embeddings")
plt.show()

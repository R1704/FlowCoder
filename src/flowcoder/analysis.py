from flowcoder.data import Data
from flowcoder.config import *
from flowcoder.io_encoder import IOEncoder
from flowcoder.state_encoder import RuleEncoder

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from sklearn.manifold import TSNE
import re
from collections import defaultdict


from_checkpoint = True

# Get the task data
data = Data(
    max_program_depth=3,
    shuffle_tasks=True,
    n_tasks=95,  # if variable_batch is true, make sure you have enough tasks for the batch_size
    variable_batch=False,  # if False, all tasks in the batch will be the same
    train_ratio=0.5,
    seed=42
    )

with open("task_names.txt", "r") as file:
    task_names = file.read().splitlines()

def extract_base_task_name(task_name):
    # Remove numerical values and additional spaces
    return re.sub(r'\d+', '', task_name).replace(' with ', '').strip()


# dictionary with (base name: list of tasks belonging to the group) pairs
task_groups = defaultdict(list)
for task_name in task_names:
    base_name = extract_base_task_name(task_name)
    task_groups[base_name].append(task_name)

# Generate a list of distinct colors
colors = list(mcolors.TABLEAU_COLORS.values())

basename2color = {}
for i, basename in enumerate(task_groups.keys()):
    basename2color[basename] = colors[i % len(colors)]

# Map each task to a color based on its group
group_colors = {}
basename2color = {}
for group_id, (group_name, tasks) in enumerate(task_groups.items()):
    color = colors[group_id % len(colors)]  # Cycle through colors if there are more groups than colors
    basename2color[group_name] = color
    for task in tasks:
        group_colors[task] = color

# We only want to embed solved task embeddings
df = pd.read_csv('/vol/tensusers4/rhommelsheim/master_thesis/results/depth_3_48_tasks2023-12-07 22:24:45.csv')
solved_tasks = df[df['Solved']]['Task Name'].unique()
print('Solved tasks:')
for st in solved_tasks:
    print('\t', st)

if from_checkpoint:
    checkpoint = '/vol/tensusers4/rhommelsheim/master_thesis/checkpoints/depth_3_48_tasks2023-12-07 22:24:45.pth'
    model = torch.load(checkpoint)
    io_encoder = model.io_encoder
    state_encoder = model.state_encoder
else:
    io_encoder = IOEncoder(
        n_examples_max=data.nb_examples_max,
        size_max=10,
        lexicon=data.lexicon,
        d_model=512
        )
    state_encoder = RuleEncoder(
        cfg=data.cfg,
        d_model=512
        )

io_encoder.to(device)
state_encoder.to(device)


###########################
###########################
##### STATE EMBEDDINGS ####
###########################
###########################
str2rule = {str(rule): rule for rule in state_encoder.rules}

# Very convoluted way to get states back from the strings we saved to CSV
def extract_state_from_str(state_str):
    # Initialize variables
    rules = []
    temp_rule = ''
    parentheses_count = 0
    first_rule_handled = False

    # Iterate through each character in the string
    for char in state_str:
        if char == '[' and not temp_rule and not first_rule_handled:
            continue
        if char == ']' and parentheses_count == 0:
            if temp_rule:
                rules.append(temp_rule.strip().strip("'"))
            break
        if char == ',' and parentheses_count == 0:
            if temp_rule:
                rules.append(temp_rule.strip().strip("'"))
                temp_rule = ''
            continue
        if char == '(':
            parentheses_count += 1
        if char == ')':
            parentheses_count -= 1
        temp_rule += char

        # Special handling for the first rule 'START'
        if not first_rule_handled and temp_rule.strip() == "'START'":
            rules.append("START")
            temp_rule = ''
            first_rule_handled = True
    state = [str2rule[r] for r in rules]
    return [state]

# TODO: pad the embeddings
# # Initialize list to store state embeddings and corresponding task names
# state_embeddings_list = []
# task_labels = []
#
# with torch.no_grad():
#     for task_name in solved_tasks[:5]:
#
#         # Extract states associated with the current task
#         states_df = df[(df['Task Name'] == task_name) & (df['Solved'])]
#
#         # Loop through each solved instance of the task
#         for _, row in states_df.iterrows():
#             state_str = row['State']
#             state = extract_state_from_str(state_str)
#             # Generate state embedding
#             state_embedding = state_encoder(state)
#             state_embedding_flat = state_embedding.view(state_embedding.size(0), -1)
#             state_embeddings_list.append(state_embedding_flat.cpu())
#             task_labels.append(task_name)  # Store the task name
#
# print(len(state_embeddings_list))
# # Stack all flattened state embeddings into a single tensor
# # all_state_embeddings = torch.cat(state_embeddings_list, dim=0)
# all_state_embeddings = torch.cat(state_embeddings_list, dim=0)
# all_state_embeddings_np = all_state_embeddings.numpy()
# print(len(all_state_embeddings_np), all_state_embeddings_np.shape, len(task_labels))
# # # Apply t-SNE to state embeddings
# # tsne_state = TSNE(n_components=2, perplexity=50, n_iter=1000, random_state=0)
# # state_embeddings_2d = tsne_state.fit_transform(all_state_embeddings_np)
# #
# # # Plotting state embeddings with colors
# # plt.figure(figsize=(10, 6))
# # for i, state_embedding in enumerate(state_embeddings_2d):
# #     task_name = task_labels[i]
# #     color = task_colors.get(task_name, 'black')  # Default color is black if task not in dict
# #     plt.scatter(state_embedding[0], state_embedding[1], color=color)
# #     # Optionally add annotations
# #     # plt.annotate(task_name, (state_embedding[0], state_embedding[1]))
# # plt.axis('off')
# # plt.title("t-SNE Visualization of Solved State Embeddings")
# # plt.show()
#
#




###########################
###########################
##### TASK EMBEDDINGS #####
###########################
###########################
def get_task_embeddings():
    # Task Embeddings
    embeddings_list = []
    embedding_names = []

    # Extract and flatten embeddings for each task
    with torch.no_grad():
        for i in range(data.n_tasks):
            task, task_name = data.get_io_batch(1, all_tasks=True)
            if task_name[0] in solved_tasks:
                embedding_names.append(task_name[0])
                embedding = io_encoder(task)

                # Flatten the embedding to 1D
                embedding_flat = embedding.view(embedding.size(0), -1)

                embeddings_list.append(embedding_flat)

    # Stack all flattened embeddings into a single tensor
    all_embeddings = torch.vstack(embeddings_list)  # Use vstack to handle different shapes

    # Convert to NumPy
    all_embeddings_np = all_embeddings.cpu().numpy()

    return all_embeddings_np, embedding_names



def plot_tsne(embeddings, task_names, group_colors, param_list):
    """
    Plots t-SNE visualizations with different parameters for comparison, including a simplified legend.

    :param embeddings: NumPy array of embeddings
    :param task_names: List of task names corresponding to each embedding
    :param group_colors: Dictionary mapping each task to a group color
    :param param_list: List of dictionaries with t-SNE parameters
    """
    num_params = len(param_list)
    plt.figure(figsize=(10, 6 * num_params))

    # Create unique scatter plots and labels for the legend
    unique_scatter_plots = {}

    for group_name, color in basename2color.items():
        scatter_plot = plt.scatter([], [], color=color)  # Create a scatter plot with no data
        unique_scatter_plots[group_name] = scatter_plot

    for i, params in enumerate(param_list):
        plt.subplot(num_params, 1, i + 1)

        tsne = TSNE(**params)
        embeddings_2d = tsne.fit_transform(embeddings)

        for j, task_name in enumerate(task_names):
            color = group_colors[task_name]
            plt.scatter(embeddings_2d[j, 0], embeddings_2d[j, 1], color=color)
            # plt.annotate(task_name, (embeddings_2d[j, 0], embeddings_2d[j, 1]))

        plt.title(f"t-SNE with params: {params}")
        plt.axis('off')

        # Add the legend outside of the plot
        plt.legend(unique_scatter_plots.values(), unique_scatter_plots.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

# Example usage
param_list = [
    {'n_components': 2, 'perplexity': 30, 'n_iter': 10000, 'random_state': 1704},
    {'n_components': 2, 'perplexity': 50, 'n_iter': 10000, 'random_state': 1704},
    {'n_components': 2, 'perplexity': 70, 'n_iter': 10000, 'random_state': 1704},
    ]


all_embeddings_np, embedding_names = get_task_embeddings()
plot_tsne(all_embeddings_np, embedding_names, group_colors, param_list)


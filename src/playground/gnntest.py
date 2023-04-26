import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Define the vocabulary
vocab = [0, 1, "+", "-"]

# Create a custom dataset
# For the sake of the example, we'll create a dataset with random trees
def create_random_tree_dataset(num_graphs=100, max_nodes=10):
    dataset = []

    for _ in range(num_graphs):
        num_nodes = torch.randint(2, max_nodes + 1, (1,)).item()
        x = torch.randint(0, len(vocab), (num_nodes, 1), dtype=torch.float)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes - 1), dtype=torch.long)
        tree = Data(x=x, edge_index=edge_index)
        dataset.append(tree)

    return dataset

train_dataset = create_random_tree_dataset()
test_dataset = create_random_tree_dataset()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Define a GNN model
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, len(vocab))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        x = global_mean_pool(x, torch.zeros(data.num_nodes, dtype=torch.long))
        x = self.fc(x)

        return F.log_softmax(x, dim=-1)

# Train the model
def train(model, loader, optimizer):
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        target = data.x.squeeze().long()

        # Compute loss for the single output value
        loss = F.nll_loss(out, target[:1])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# Test the model
def test(model, loader):
    model.eval()
    correct = 0

    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.x.squeeze().long()).sum().item()

    return correct / sum([data.num_nodes for data in loader.dataset])

# Instantiate and train the model
model = GNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(1, 101):
#     train_loss = train(model, train_loader, optimizer)
#     test_acc = test(model, test_loader)
#     print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}')

# Make predictions and construct the abstract syntax tree
def predict_next_node(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        probs = F.softmax(out, dim=-1)
        next_node_idx = torch.multinomial(probs, 1).item()
    return vocab[next_node_idx]


def construct_tree(model, max_depth=5):
    tree = Data(x=torch.tensor([[0.0]]), edge_index=torch.empty((2, 0), dtype=torch.long))
    nodes_to_process = [(0, 0)]  # (node_index, depth)

    while nodes_to_process:
        parent_node_idx, depth = nodes_to_process.pop(0)
        parent_data = Data(x=tree.x[parent_node_idx].view(1, -1), edge_index=torch.empty((2, 0), dtype=torch.long))

        if depth < max_depth:
            next_node = predict_next_node(model, parent_data)
            next_node_idx = tree.num_nodes
            tree.x = torch.cat([tree.x, torch.tensor([[vocab.index(next_node)]], dtype=torch.float)], dim=0)
            tree.edge_index = torch.cat(
                [tree.edge_index, torch.tensor([[parent_node_idx], [next_node_idx]], dtype=torch.long)], dim=1)
            nodes_to_process.append((next_node_idx, depth + 1))

    return tree


# Construct an abstract syntax tree using the trained model
ast = construct_tree(model)
print(ast)

# def visualize_ast(ast, labels=None):
#     # Convert the PyTorch Geometric Data object to a NetworkX graph
#     G = nx.DiGraph()
#     G.add_edges_from(ast.edge_index.T.tolist())
#
#     # Set node labels if not provided
#     if labels is None:
#         labels = {i: vocab[int(ast.x[i, 0])] for i in range(ast.num_nodes)}
#     else:
#         labels = {i: labels[i] for i in range(ast.num_nodes)}
#
#     # Draw the graph using a shell layout
#     pos = nx.shell_layout(G)
#     nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, font_size=12, font_weight='bold', node_color='cyan', edgecolors='black')
#
#     # Show the plot
#     plt.show()

def visualize_ast(ast, labels=None):
    # Convert the PyTorch Geometric Data object to a NetworkX graph
    G = nx.DiGraph()
    G.add_edges_from(ast.edge_index.T.tolist())

    # Set node labels if not provided
    if labels is None:
        labels = {i: vocab[int(ast.x[i, 0])] for i in range(ast.num_nodes)}
    else:
        labels = {i: labels[i] for i in range(ast.num_nodes)}

    # Draw the graph using a spring layout
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, font_size=12, font_weight='bold', node_color='cyan', edgecolors='black')

    # Show the plot
    plt.show()


# Construct an abstract syntax tree using the trained model
ast = construct_tree(model)

# Visualize the AST
visualize_ast(ast)

for x in train_dataset:
    visualize_ast(x)

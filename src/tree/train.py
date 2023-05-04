from embedding import *
from positional_encoding import *
from model import *
from utils import *
from grammar import *


def encode_tree(tree, embedding_model, positional_encoder):
    # Get the embeddings of all nodes in the tree
    node_embeddings = {}
    for node in tree.traverse([]):
        node_idx = primitives.index(node)
        node_embedding = embedding_model(torch.tensor([node_idx]))
        node_embeddings[node] = node_embedding

    # Get the positional encodings of all nodes in the tree
    positional_encodings = positional_encoder.encode_position(tree)

    # Combine embeddings and positional encodings
    combined_encodings = {}
    for node in tree.traverse():
        combined_encoding = node_embeddings[node] + positional_encodings[node]
        combined_encodings[node] = combined_encoding

    return combined_encodings


num_nodes = len(primitives)
max_depth = 5
max_arity = 2
embedding_dim = 256

num_epochs = 10
d_model = 256
nhead = 4
num_layers = 2
batch_size = 1


# Instantiate the Transformer model
transformer_model = TransformerModel(num_nodes, d_model, nhead, num_layers)
euclidean_model = EuclideanEmbedding(num_nodes, embedding_dim)
euclidean_optimizer = torch.optim.Adam(euclidean_model.parameters(), lr=0.001)

# Set the learning rate and optimizer
lr = 0.001
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=lr)

# Loss function
criterion = nn.CrossEntropyLoss()

# Start with a single-node tree (the root node)
current_tree = Node.get_random_root(primitives)
current_tree.print()

for epoch in range(num_epochs):
    epoch_loss = 0





    # Euclidean Embedding training
    train_data = prepare_train_data(current_tree, primitives)
    print(train_data)
    train_embeddings(euclidean_model, euclidean_optimizer, train_data, EuclideanEmbedding.euclidean_loss, max_norm=5.0)

    # Positional Encoding computation
    positional_encoder = PositionalEncoding(max_depth, max_arity, d_model)

    combined_encodings = encode_tree(current_tree, euclidean_model, positional_encoder)
    print(combined_encodings)
    #
    # for batch_idx in range(0, len(train_data), batch_size):
    #     optimizer.zero_grad()
    #
    #     # Get the input and target sequences
    #     input_indices, positions = encode_tree(current_tree, primitives)
    #     input_indices = input_indices[:-1]  # Remove the last node
    #     positions = positions[:-1]
    #
    #     # Get the embeddings and add position encodings
    #     input_embeddings = euclidean_model(input_indices)
    #     node_index_to_position_mapping = pos_enc.get_node_index_to_position_mapping()
    #     positional_indices = torch.tensor([node_index_to_position_mapping[idx.item()] for idx in input_indices])
    #     input_embeddings += positional_encoding[positional_indices]
    #     input_embeddings = input_embeddings.unsqueeze(1)  # Add the sequence dimension
    #
    #     # Get the target node and type
    #     target_next_node = positions[1:]  # Shifted by 1 to the right
    #     target_node_type = input_indices[1:]
    #
    #     # Forward pass
    #     next_node_logits, node_type_logits = transformer_model(input_embeddings, current_tree)
    #     next_node_logits = next_node_logits.squeeze(1)
    #     node_type_logits = node_type_logits.squeeze(1)
    #
    #     # Compute the loss and update the model
    #     loss1 = criterion(next_node_logits, target_next_node.long())
    #     loss2 = criterion(node_type_logits, target_node_type)
    #
    #     loss = loss1 + loss2
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     epoch_loss += loss.item()
    #
    # print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_data)}")

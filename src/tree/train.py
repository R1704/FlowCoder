from embedding import *
from positional_encoding import *
from model import *
from utils import *


primitives = [
    *[Node(symbol=f'{i}') for i in range(10)],
    Node(symbol='+', arity=2, function=lambda x, y: x + y),
    Node(symbol='-', arity=2, function=lambda x, y: x - y),
    Node(symbol='*', arity=2, function=lambda x, y: x * y),
    Node(symbol='/', arity=2, function=lambda x, y: x / y),
]

tree = Node.create_tree(primitives, depth=5)
tree.print()


##########################
##########################
### Poincare Embedding ###
##########################
##########################

# Create a PoincareEmbedding instance
num_nodes = len(primitives)
embedding_dim = 256
model = PoincareEmbedding(num_nodes, embedding_dim)

# Prepare your training data
train_data = prepare_train_data(tree, primitives)

# Set up the RSGD optimizer
optimizer = RSGD(model.parameters(), lr=1e-4)

# Train the embeddings
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for u_idx, v_idx in train_data:
        optimizer.zero_grad()
        u = model(torch.tensor([u_idx]))
        v = model(torch.tensor([v_idx]))
        loss = PoincareEmbedding.poincare_loss(u, v)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}')




###########################
###########################
### Euclidean Embedding ###
###########################
###########################

# Create an EuclideanEmbedding instance
model = EuclideanEmbedding(num_nodes, embedding_dim)
# Set up the Adam optimizer for EuclideanEmbedding
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Train the embeddings
num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0
    for u_idx, v_idx in train_data:
        optimizer.zero_grad()
        u = model(torch.tensor([u_idx]))
        v = model(torch.tensor([v_idx]))
        loss = EuclideanEmbedding.euclidean_loss(u, v)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}')


# Positional Encoding

pos_enc = PositionalEncoding(tree, num_nodes)
positional_encoding = pos_enc.encode()

def add_positional_encoding(embeddings, positional_encodings):
    assert len(embeddings) == len(positional_encodings), "Mismatch in length of embeddings and positional encodings."
    return embeddings + positional_encodings


# Combining the embeddings and positional encodings
# tree_indices, _ = encode_tree(tree, primitives)
# node_embeddings = model(tree_indices)
# node_embeddings_with_position = add_positional_encoding(node_embeddings, positional_encoding)




num_epochs = 100
d_model = 256
nhead = 4
num_layers = 2
batch_size = 1

# Instantiate the Transformer model
transformer_model = TransformerModel(num_nodes, d_model, nhead, num_layers)

# Set the learning rate and optimizer
lr = 0.001
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=lr)

# Loss function
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx in range(0, len(train_data), batch_size):
        optimizer.zero_grad()

        # Get the input and target sequences
        input_indices, positions = encode_tree(tree, primitives)
        input_indices = input_indices[:-1]  # Remove the last node
        positions = positions[:-1]

        # Get the embeddings and add position encodings

        input_embeddings = model(input_indices) + pos_enc.encode()
        input_embeddings = input_embeddings.unsqueeze(1)  # Add the sequence dimension

        # Get the target node and type
        target_next_node = positions[1:]  # Shifted by 1 to the right
        target_node_type = input_indices[1:]

        # Forward pass
        next_node_logits, node_type_logits = transformer_model(input_embeddings)
        next_node_logits = next_node_logits.squeeze(1)
        node_type_logits = node_type_logits.squeeze(1)

        # Compute the loss and update the model
        loss1 = criterion(next_node_logits, target_next_node.long())
        loss2 = criterion(node_type_logits, target_node_type)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_data)}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from flowcoder.flowcoder_torchgfn.program_states import ProgramStates

class ProgramGNN(nn.Module):
    """Graph Attention Network for processing program ASTs."""
    
    def __init__(
        self, 
        node_features: int, 
        hidden_dim: int, 
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Linear(node_features, hidden_dim)
        
        # Use GATv2 layers (improved version of GAT)
        self.convs = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial embedding
        x = self.embedding(x)
        
        # Graph attention layers with residual connections and layer norm
        for conv, norm in zip(self.convs, self.layer_norms):
            x_residual = x
            x = conv(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
            x = x + x_residual  # Residual connection
            x = norm(x)
        
        # Global attention pooling
        x = global_mean_pool(x, batch)
        
        return x

class PolicyNetwork(nn.Module):
    """Policy network for the GFlowNet."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        self.gnn = ProgramGNN(
            node_features=4,  # From one-hot encoding in ProgramState
            hidden_dim=hidden_dim,
            num_layers=3
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, states: ProgramStates) -> torch.Tensor:
        # Process the program graphs
        graph_embeddings = self.gnn(states.program_graphs)
        
        # Get action logits
        logits = self.policy_head(graph_embeddings)
        
        # Mask invalid actions
        logits = logits.masked_fill(~states.forward_masks, float('-inf'))
        
        return logits

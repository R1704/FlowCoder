import pytest
import torch
from flowcoder.flowcoder_torchgfn.sandbox import gnn

def test_gnn_index_error():
    graph = torch.empty((0, 2), dtype=torch.long)  # Empty graph to trigger IndexError
    with pytest.raises(IndexError):
        lambda_embedding = gnn(graph)
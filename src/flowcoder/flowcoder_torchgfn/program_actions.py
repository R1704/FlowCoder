from typing import List, Optional, Tuple
import torch
from gfn.actions import Actions
from deepsynth.program import Program

class ProgramActions(Actions):
    """Actions for program synthesis in GFlowNets."""
    
    # Class variables required by Actions base class
    action_shape = (1,)  # Each action is a single integer index
    dummy_action = torch.tensor([-1])  # Used for padding
    exit_action = torch.tensor([-2])   # Used to signal end of program
    
    def __init__(self, tensor: torch.Tensor):
        super().__init__(tensor)
        self.applied_rules: List[Tuple[Program, List[int]]] = []  # Track applied CFG rules
        
    def __repr__(self):
        return f"ProgramActions(tensor={self.tensor}, batch_shape={self.batch_shape})"
    
    @classmethod
    def from_cfg_rule(cls, rule_index: int) -> 'ProgramActions':
        """Create an action from a CFG rule index."""
        return cls(torch.tensor([[rule_index]], dtype=torch.long))

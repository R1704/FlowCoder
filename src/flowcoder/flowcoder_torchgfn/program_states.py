from typing import Dict, List, Optional, Tuple, Union
from copy import deepcopy

import torch
from torch_geometric.data import Data, Batch

from gfn.states import DiscreteStates
from deepsynth.program import Program, Function, Lambda, Variable, BasicPrimitive
from deepsynth.type_system import Type, Arrow
from deepsynth.dsl import DSL

class ProgramStates(DiscreteStates):
    """Program states for GFlowNets."""
    
    state_shape = (128,)  # State embedding dimension
    s0 = torch.zeros(128)  # Initial state
    sf = torch.full((128,), float('-inf'))  # Final/terminal state
    n_actions = None  # Will be set by ProgramEnv during initialization
    device = torch.device('cpu')  # Default device, can be changed
    
    def __init__(
        self,
        tensor: torch.Tensor,
        program_graphs: Optional[Batch] = None,
        programs: Optional[List[Program]] = None,
        dsl: Optional[DSL] = None,
        forward_masks: Optional[torch.Tensor] = None,  
        backward_masks: Optional[torch.Tensor] = None
    ):
        super().__init__(tensor, forward_masks, backward_masks)
        self.program_graphs = program_graphs
        self.programs = [] if programs is None else programs
        self.dsl = dsl
        
    @classmethod
    def make_random_states_tensor(cls, batch_shape: tuple) -> torch.Tensor:
        """Create random initial states."""
        return torch.randn((*batch_shape, *cls.state_shape))
        
    def clone(self) -> 'ProgramStates':
        """Create a deep copy of the state."""
        new_tensor = self.tensor.detach().clone()
        new_forward_masks = self.forward_masks.detach().clone() if self.forward_masks is not None else None
        new_backward_masks = self.backward_masks.detach().clone() if self.backward_masks is not None else None
        new_program_graphs = self.program_graphs if self.program_graphs is None else deepcopy(self.program_graphs)
        new_programs = deepcopy(self.programs)
        
        return ProgramStates(
            tensor=new_tensor,
            program_graphs=new_program_graphs,
            programs=new_programs,
            dsl=self.dsl,
            forward_masks=new_forward_masks,
            backward_masks=new_backward_masks
        )

    @staticmethod
    def program_to_graph(program: Optional[Program], dsl: Optional[DSL]) -> Data:
        """Convert a program to a graph representation."""
        if program is None:
            return ProgramStates.create_initial_graph()
            
        edge_index = []
        node_features = []
        node_types = []  # Store type information for each node
        node_to_program = {}  # Map nodes back to program components
        
        def process_node(node: Program, parent_idx: Optional[int] = None) -> int:
            current_idx = len(node_features)
            node_to_program[current_idx] = node
            
            # Create node features based on program type
            if isinstance(node, BasicPrimitive):
                node_features.append([1, 0, 0, 0])
                node_types.append(node.type)
            elif isinstance(node, Variable):
                node_features.append([0, 1, 0, 0]) 
                node_types.append(node.type)
            elif isinstance(node, Lambda):
                node_features.append([0, 0, 1, 0])
                node_types.append(node.type)
            elif isinstance(node, Function):
                node_features.append([0, 0, 0, 1])
                node_types.append(node.type)
            
            # Add edge if there's a parent
            if parent_idx is not None:
                edge_index.append([parent_idx, current_idx])
                edge_index.append([current_idx, parent_idx])  # Bidirectional edges
            
            # Process children recursively
            if isinstance(node, Function):
                for arg in node.arguments:
                    child_idx = process_node(arg, current_idx)
                process_node(node.function, current_idx)
            elif isinstance(node, Lambda):
                process_node(node.body, current_idx)
            
            return current_idx
        
        process_node(program)
        
        # Convert to PyTorch tensors
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            types=node_types,
            node_to_program=node_to_program
        )

    @staticmethod
    def graph_to_program(graph: Data) -> Program:
        """Convert a graph back to a program."""
        node_to_program = graph.node_to_program
        
        # Find root node (node with no incoming edges)
        edge_index = graph.edge_index.t().tolist()
        incoming_edges = {e[1] for e in edge_index}
        root_idx = next(i for i in range(len(graph.x)) if i not in incoming_edges)
        
        return node_to_program[root_idx]

    @staticmethod
    def create_initial_graph() -> Data:
        """Create a graph representation for an initial (empty) program state."""
        return Data(
            x=torch.tensor([[0, 0, 0, 0]], dtype=torch.float),  # Single empty node
            edge_index=torch.zeros((2, 0), dtype=torch.long),   # No edges
            types=[None],  # No type for initial node
            node_to_program={0: None}  # Map node to None program
        )

    @classmethod
    def from_programs(cls, programs: List[Optional[Program]], dsl: DSL) -> 'ProgramStates':
        """Create ProgramStates from a list of programs."""
        # Handle None programs (initial states) by creating special initial graphs
        graph_states = [
            cls.program_to_graph(prog, dsl) if prog is not None 
            else cls.create_initial_graph() 
            for prog in programs
        ]
        batched_graphs = Batch.from_data_list(graph_states)
        
        # Use GNN to get embeddings
        dummy_embeddings = torch.zeros((len(programs), 128))
        
        return cls(
            tensor=dummy_embeddings,
            program_graphs=batched_graphs,
            programs=programs,
            dsl=dsl
        )

    def get_valid_actions(self, cfg) -> torch.Tensor:
        """Get valid actions mask based on current program state and CFG."""
        batch_size = len(self.programs)
        n_actions = len(cfg.get_terminals()) + len(cfg.rules)
        masks = torch.zeros((batch_size, n_actions), dtype=torch.bool)
        
        for i, program in enumerate(self.programs):
            if program is None:  # Initial state
                # All rules starting from the start symbol are valid
                start_rules = cfg.rules.get(cfg.start, {})
                for rule_idx in range(n_actions):
                    masks[i, rule_idx] = rule_idx in start_rules
                continue
                
            # Find current non-terminal based on program state
            current_state = (program.type, None, program.depth if hasattr(program, 'depth') else 0)
            
            # Get valid productions from CFG
            if current_state in cfg.rules:
                for production, next_states in cfg.rules[current_state].items():
                    if hasattr(cfg, 'rule_to_action'):
                        action_idx = cfg.rule_to_action[production]
                    else:
                        # Fallback to primitive index if no mapping exists
                        try:
                            action_idx = list(cfg.rules[current_state].keys()).index(production)
                        except ValueError:
                            continue
                    masks[i, action_idx] = True
                    
            # Check if we can terminate (when program is complete)
            if program.is_complete():
                masks[i, -1] = True  # Last action is exit action
                
        return masks

    def __repr__(self):
        """Pretty print state information."""
        programs_str = [str(p) if p is not None else "None" for p in self.programs]
        return (f"ProgramStates(programs={programs_str}, "
                f"n_valid_actions={self.forward_masks.sum().item() if self.forward_masks is not None else 0})")

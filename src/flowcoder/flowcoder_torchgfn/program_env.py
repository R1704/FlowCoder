from typing import Dict, List, Optional, Tuple, Union

import torch
from gfn.env import DiscreteEnv
from deepsynth.dsl import DSL
from deepsynth.type_system import Type
from deepsynth.DSL.list import semantics as list_semantics, primitive_types as list_primitive_types
from flowcoder.flowcoder_torchgfn.program_states import ProgramStates
from flowcoder.flowcoder_torchgfn.program_actions import ProgramActions
from math import prod
from deepsynth.program import BasicPrimitive

class ProgramEnv(DiscreteEnv):
    """Environment for program synthesis using GFlowNets."""
    
    def __init__(
        self,
        dsl: Optional[DSL] = None,
        type_request: Optional[Type] = None,
        device_str: Optional[str] = None,
        max_program_depth: int = 4,
        test_cases: Optional[List[Tuple]] = None
    ):
        self.dsl = dsl or DSL(list_semantics, list_primitive_types)
        self.type_request = type_request
        self.max_program_depth = max_program_depth
        self.test_cases = test_cases or []  # Initialize with empty list if not provided
        
        # Create CFG from DSL
        self.cfg = self.dsl.DSL_to_CFG(
            type_request=type_request,
            max_program_depth=max_program_depth
        )
        
        # Number of actions = number of CFG rules + terminals
        n_actions = len(self.cfg.get_terminals()) + len(self.cfg.rules)
        
        # Map CFG rules and terminals to action indices
        self.rule_to_action = {}
        self.action_to_rule = {}
        self._build_action_maps()
        
        # Set n_actions in ProgramStates before calling super().__init__
        ProgramStates.n_actions = n_actions
        if device_str:
            ProgramStates.device = torch.device(device_str)
        
        # Initialize the DiscreteEnv with proper States and Actions classes
        super().__init__(
            n_actions=n_actions,
            s0=ProgramStates.s0,
            state_shape=ProgramStates.state_shape,
            device_str=device_str
        )
        
    def _build_action_maps(self):
        """Build mappings between CFG rules and action indices."""
        idx = 0
        
        # Map start rules (initial rules) first
        start_rules = self.cfg.rules.get(self.cfg.start, {})
        for prod in start_rules:
            self.rule_to_action[prod] = idx
            self.action_to_rule[idx] = prod
            idx += 1
        
        # Map remaining terminals and rules
        for terminal in self.cfg.get_terminals():
            if terminal not in self.rule_to_action:  # Avoid duplicates
                self.rule_to_action[terminal] = idx
                self.action_to_rule[idx] = terminal
                idx += 1
            
        for lhs, rules in self.cfg.rules.items():
            if lhs == self.cfg.start:  # Skip start rules as they're already mapped
                continue
            for rhs in rules:
                self.rule_to_action[(lhs, rhs)] = idx
                self.action_to_rule[idx] = (lhs, rhs)
                idx += 1

    def step(self, states: ProgramStates, actions: ProgramActions) -> ProgramStates:
        """Apply production rules to extend the programs."""
        new_states = states.clone()
        
        for i, action in enumerate(actions.tensor):
            if action == actions.exit_action:
                if states.programs[i] and states.programs[i].is_complete():
                    new_states.tensor[i] = self.sf
                continue
            
            try:
                rule = self.action_to_rule[action.item()]
                program = new_states.programs[i]
                
                # Handle initial state (None program)
                if program is None:
                    if isinstance(rule, tuple):  # Non-terminal rule
                        _, rhs = rule
                        new_program = rhs
                    else:  # Terminal rule with type request
                        new_program = BasicPrimitive(rule, self.type_request)
                else:
                    # Apply CFG rule to extend program
                    if isinstance(rule, tuple):  # Non-terminal rule
                        _, rhs = rule
                        new_program = program.apply_rule(rhs)
                    else:  # Terminal rule
                        # Create new terminal node with correct type
                        if hasattr(program, 'type'):
                            terminal = BasicPrimitive(rule, program.type)
                        else:
                            terminal = BasicPrimitive(rule, self.type_request)
                        new_program = terminal
                
                # Verify the program is valid before updating
                if new_program is not None:
                    new_states.programs[i] = new_program
                    new_states.program_graphs[i] = ProgramStates.program_to_graph(new_program, self.dsl)
            except (KeyError, AttributeError) as e:
                logging.warning(f"Invalid action {action.item()}: {e}")
                continue
            
        return new_states

    def update_masks(self, states: ProgramStates) -> None:
        """Update masks of valid actions based on current program state."""
        # Get valid next actions from CFG for each program
        valid_actions = states.get_valid_actions(self.cfg)
        states.forward_masks = valid_actions

    def backward_step(self, states: ProgramStates, actions: ProgramActions) -> ProgramStates:
        """Remove the last action applied to programs."""
        new_states = states.clone()
        
        for i, action in enumerate(actions.tensor):
            if action == actions.dummy_action:
                continue
                
            program = new_states.programs[i]
            # Remove last applied rule
            new_program = program.remove_last_rule()
            new_states.programs[i] = new_program
            # Update graph representation
            new_states.program_graphs[i] = ProgramStates.program_to_graph(new_program, self.dsl)
            
        return new_states

    def reward(self, final_states: Union[ProgramStates, torch.Tensor]) -> torch.Tensor:
        """Calculate rewards based on program correctness and complexity."""
        if isinstance(final_states, torch.Tensor):
            # Handle case where we get a tensor instead of ProgramStates
            return torch.zeros(final_states.shape[0])  # Default reward
            
        batch_size = len(final_states.programs)
        rewards = torch.zeros(batch_size)
        
        if not self.test_cases:  # If no test cases, return zero rewards
            return rewards
            
        for i, program in enumerate(final_states.programs):
            if not program.is_complete():
                continue
                
            # Test program on example inputs/outputs
            correct = 0
            total = len(self.test_cases)
            
            for inputs, expected_output in self.test_cases:
                try:
                    result = program.eval(self.dsl, inputs)
                    if result == expected_output:
                        correct += 1
                except:
                    continue
            
            # Reward = correctness - complexity penalty
            accuracy = correct / total
            complexity_penalty = 0.1 * program.size()  # Penalize long programs
            rewards[i] = accuracy - complexity_penalty
            
        return rewards

    def reset(
        self,
        batch_shape: Optional[Union[int, Tuple[int]]] = None,
        random: bool = False,
        sink: bool = False,
        seed: int = None
    ) -> ProgramStates:
        if batch_shape is None:
            batch_shape = (1,)
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
            
        # Create initial empty programs
        initial_programs = [None] * prod(batch_shape)

        # Create initial program states with graphs
        states = ProgramStates.from_programs(
            programs=initial_programs,
            dsl=self.dsl
        )
        states.tensor = torch.zeros((*batch_shape, *self.state_shape))
        
        # Update masks for valid actions
        self.update_masks(states)
        
        return states
        
    def make_states_class(self) -> type[ProgramStates]:
        """Override to use our custom ProgramStates."""
        return ProgramStates

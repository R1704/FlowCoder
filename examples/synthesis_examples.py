import torch
import logging
from deepsynth.type_system import Arrow, List, INT
from deepsynth.dsl import DSL
from deepsynth.DSL.list import semantics, primitive_types

from flowcoder.flowcoder_torchgfn.program_env import ProgramEnv
from flowcoder.flowcoder_torchgfn.program_states import ProgramStates
from flowcoder.flowcoder_torchgfn.program_actions import ProgramActions
from flowcoder.flowcoder_torchgfn.models import PolicyNetwork

logging.basicConfig(level=logging.INFO, format='%(message)s')

def example1_list_manipulation():
    """Example: Synthesize a program that doubles each number in a list."""
    logging.info("\n=== Example 1: List Manipulation (Double Numbers) ===")
    
    # Define the synthesis task
    type_request = Arrow(List(INT), List(INT))  # [Int] -> [Int]
    logging.info(f"Type Request: {type_request}")
    
    # Create environment
    env = ProgramEnv(
        dsl=DSL(semantics, primitive_types),
        type_request=type_request,
        max_program_depth=5
    )
    
    # Test cases
    env.test_cases = [
        ([1, 2, 3], [2, 4, 6]),
        ([0, 5, 10], [0, 10, 20]),
        ([], [])
    ]
    logging.info(f"Test Cases: {env.test_cases}")
    
    # Initialize policy network
    policy = PolicyNetwork(
        state_dim=128,  # Matches ProgramStates.state_shape
        action_dim=env.n_actions,
        hidden_dim=256
    )
    
    # Run multiple steps
    state = env.reset()
    logging.info("\nInitial State:")
    logging.info(f"- Programs: {state.programs}")
    logging.info(f"- Valid Actions: {state.forward_masks.sum().item()}/{env.n_actions}")
    
    for step in range(5):  # Try 5 steps of synthesis
        action_logits = policy(state)
        action = ProgramActions.from_cfg_rule(action_logits.argmax().item())
        logging.info(f"\nStep {step + 1}:")
        logging.info(f"Chosen Action: {env.action_to_rule[action.tensor.item()]}")
        
        state = env.step(state, action)
        logging.info(f"Current Program: {state.programs[0]}")
        logging.info(f"Valid Actions: {state.forward_masks.sum().item()}/{env.n_actions}")
        
        if state.tensor[0].equal(env.sf):  # Check if we reached a final state
            logging.info("Program complete!")
            break
    
    # Evaluate final program
    rewards = env.reward(state)
    logging.info(f"\nFinal reward: {rewards[0]:.3f}")

def example2_conditional_synthesis():
    """Example: Synthesize a program that keeps only positive numbers."""
    logging.info("\n=== Example 2: Conditional Synthesis (Filter Positive) ===")
    
    # Define the synthesis task
    type_request = Arrow(List(INT), List(INT))
    
    # Create environment with specific primitives
    filtered_primitives = {
        k: v for k, v in primitive_types.items() 
        if k in ['filter', 'gt?', 'map', '0', 'empty']
    }
    dsl = DSL(semantics, filtered_primitives)
    
    env = ProgramEnv(
        dsl=dsl,
        type_request=type_request,
        max_program_depth=4
    )
    
    # Test cases
    env.test_cases = [
        ([-1, 2, -3, 4], [2, 4]),
        ([0, -5, 10], [0, 10]),
        ([-2, -1], [])
    ]
    logging.info(f"Test Cases: {env.test_cases}")
    
    state = env.reset()
    logging.info("\nInitial State:")
    logging.info(f"- Available Primitives: {list(filtered_primitives.keys())}")
    logging.info(f"- Valid Actions: {state.forward_masks.sum().item()}/{env.n_actions}")

def example3_program_visualization():
    """Example: Visualize program AST as a graph."""
    logging.info("\n=== Example 3: Program Visualization ===")
    from torch_geometric.utils import to_networkx
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Create a simple program
    from deepsynth.program import Function, BasicPrimitive
    program = Function(
        BasicPrimitive("map", Arrow(INT, List(INT))),
        [BasicPrimitive("*", Arrow(INT, INT))],
        type_=Arrow(List(INT), List(INT))
    )
    logging.info(f"Created Program: {program}")
    
    # Convert to graph
    graph = ProgramStates.program_to_graph(program, None)
    
    # Visualize
    nx_graph = to_networkx(graph)
    plt.figure(figsize=(8, 8))
    nx.draw(nx_graph, with_labels=True)
    plt.savefig('program_graph.png')
    logging.info(f"Saved graph visualization to: program_graph.png")

def example4_batch_processing():
    """Example: Process multiple programs in parallel."""
    logging.info("\n=== Example 4: Batch Processing ===")
    
    type_request = Arrow(List(INT), INT)
    env = ProgramEnv(
        type_request=type_request,
        max_program_depth=3,
        test_cases=[
            ([1, 2, 3], 6),      # sum
            ([4, 5, 6], 15),     # sum
            ([], 0)              # sum of empty list
        ]
    )
    
    # Create batch of initial states
    batch_size = 8
    states = env.reset(batch_size)
    
    # Create batch of random actions
    actions = torch.randint(0, env.n_actions, (batch_size, 1))
    action_batch = ProgramActions(actions)
    
    # Step environment with batch
    next_states = env.step(states, action_batch)
    
    # Get rewards for batch
    rewards = env.reward(next_states)
    
    logging.info(f"\nBatch Results:")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Programs: {next_states.programs}")
    logging.info(f"Rewards: {rewards.tolist()}")
    logging.info(f"Average Reward: {rewards.mean().item():.3f}")

if __name__ == "__main__":
    example1_list_manipulation()
    example2_conditional_synthesis()
    example3_program_visualization()
    example4_batch_processing()

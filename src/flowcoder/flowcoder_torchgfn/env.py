from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates
from gfn.actions import Actions
import torch
# ...import your DSL/CFG and reward modules as needed...

class FlowCoderStates(DiscreteStates):
    # State now holds an AST rather than a list of rule strings
    def __init__(self, ast_list):
        # ast_list: list of AST objects
        super().__init__(ast_list)
        self.ast_list = ast_list

    def as_tensor(self):
        # If needed, convert AST structure to a tensor (e.g., via our GNN encoder)
        raise NotImplementedError("Pass the ASTs through a GNN encoder for embedding.")


class FlowCoderActions(Actions):
    # Each action corresponds to an allowed deepsynth primitive application.
    # For instance, a tuple (node_index, primitive) or (parent_node, subtree_template).
    action_shape = (1,)  # Adapt this shape as needed
    dummy_action = torch.tensor([0])
    exit_action = torch.tensor([-1])

    @staticmethod
    def make_dummy_actions(batch_shape):
        return FlowCoderActions(torch.zeros(batch_shape, dtype=torch.long))

class FlowCoderEnv(DiscreteEnv):
    def __init__(self, cfg, dsl, reward_fn, max_program_depth):
        super().__init__()
        self.cfg = cfg
        self.dsl = dsl
        self.reward_fn = reward_fn
        self.max_depth = max_program_depth
        # Use deepsynth primitives to create a vocabulary of allowed operations
        self.primitives = dsl.all_primitives()  # e.g., from deepsynth
        # Map primitives to indices for the action space
        self.prim2idx = {prim: i for i, prim in enumerate(self.primitives)}

    def reset(self, batch_size: int = 1):
        # Initialize each state with a minimal AST (perhaps a ROOT node)
        asts = [AST.start_state(self.dsl) for _ in range(batch_size)]
        return FlowCoderStates(asts)

    def valid_actions(self, states: FlowCoderStates):
        # For each AST in states, compute valid attachment operations using deepsynth CFG logic.
        valid_actions = []
        for ast in states.ast_list:
            actions = []
            # For each candidate expansion point in the AST,
            # use deepsynth logic to determine allowed primitives.
            for node in ast.candidate_nodes():
                allowed = self.dsl.allowed_primitives(node)
                for prim in allowed:
                    # Create an action tuple: (node id, primitive index)
                    action_idx = self.prim2idx[prim]
                    actions.append((node.id, action_idx))
            valid_actions.append(actions)
        # Wrap the valid actions in a FlowCoderActions instance (customize as needed)
        return FlowCoderActions(valid_actions)

    def step(self, states: FlowCoderStates, actions: FlowCoderActions):
        next_states = []
        rewards = []
        dones = []
        # Assume actions.tensor contains a list of (node_id, prim_index) pairs per state in the batch.
        for ast, action in zip(states.ast_list, actions.tensor.tolist()):
            if action == FlowCoderActions.exit_action.item():
                # If exit action, the AST is complete.
                new_ast = ast
                done = True
            else:
                # Get node id and primitive from action
                node_id, prim_index = action
                primitive = self.primitives[prim_index]
                # Use deepsynth DSL/CFG to expand the AST at the given node
                new_ast = ast.apply_primitive(node_id, primitive)
                done = new_ast.depth() >= self.max_depth or new_ast.is_complete()
            next_states.append(new_ast)
            rewards.append(self.reward_fn(new_ast))
            dones.append(done)
        return FlowCoderStates(next_states), torch.tensor(rewards), torch.tensor(dones)
import numpy as np

def cellular_automaton_rule(state, rule_number):
    """Compute the next state of a 1D cellular automaton given its current state and a rule number."""
    # Convert the rule number to a binary string with leading zeros
    binary_rule = np.binary_repr(rule_number, 8)

    # Create a lookup table that maps the 3-bit neighborhood states to their next state
    lookup_table = {}
    for i in range(8):
        lookup_table[np.binary_repr(i, 3)] = int(binary_rule[i])

    # Pad the state with zeros on either end to handle boundary conditions
    padded_state = np.pad(state, (1, 1), mode='wrap')

    # Compute the next state using the lookup table
    next_state = np.zeros_like(state)
    for i in range(1, len(state) + 1):
        neighborhood = ''.join(str(x) for x in padded_state[i-1:i+2])
        next_state[i-1] = lookup_table[neighborhood]

    return next_state

# Example usage
state = np.array([0, 1, 0, 0, 1, 1, 1, 0])
rule_number = 18
for i in range(10):
    print(state)
    state = cellular_automaton_rule(state, rule_number)

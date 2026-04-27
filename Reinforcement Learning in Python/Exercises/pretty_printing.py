from typing import Mapping, Sequence

from gridworld_standard_windy import State, WindyGridworld


ValueTable = Mapping[State, float]
ReturnTable = Mapping[State, Sequence[float]]
Policy = Mapping[State, str]


def print_values(V: ValueTable, g: WindyGridworld) -> None:
    """
    Prints the values of each state in a grid representation, using the provided value
    table and grid layout. For states that are not part of the grid's valid states,
    a placeholder 'XXX' is displayed.

    Args:
        V: A value table containing the mapping of states to their respective values.
        g: An instance of WindyGridworld, which provides information about grid
            dimensions, state coordinates, and valid states.

    Returns:
        None
    """
    for i in range(g.rows):
        row_items = []
        for j in range(g.cols):
            state = (i, j)
            if state in g.get_all_states():
                row_items.append(f'{V.get(state, 0.0):>7.2f}')
            else:
                row_items.append('   XXX ')
        print(' '.join(row_items))
    return None

def print_policy(p: Policy, g: WindyGridworld) -> None:
    """
    Prints the policy for a WindyGridworld environment.

    This function visualizes the policy by iterating through each cell of the
    WindyGridworld grid and printing the corresponding action for each state.
    If a state is not part of the grid's valid states, it prints 'XXX'. If
    a state is terminal, it prints 'T'. For valid non-terminal states, it
    prints the action specified in the policy.

    Args:
        p (Policy): The policy representing the actions to be taken for each
            state in the grid.
        g (WindyGridworld): The WindyGridworld environment providing information
            about the grid dimensions, states, and terminal states.

    Returns:
        None
    """
    for i in range(g.rows):
        row_items = []
        for j in range(g.cols):
            state = (i, j)
            if state not in g.get_all_states():
                row_items.append(' XXX ')
            elif g.is_terminal(state):
                row_items.append('  T  ')
            else:
                row_items.append(f'{p.get(state, " "):>5}')
        print(' '.join(row_items))
    return None


def print_returns(returns: ReturnTable) -> None:
    """
    Prints Monte Carlo return histories for each state.

    Args:
        returns: A mapping of states to the sequence of returns observed for that state.

    Returns:
        None
    """
    for state in sorted(returns):
        print(f'{state}: {list(returns[state])}')
    return None

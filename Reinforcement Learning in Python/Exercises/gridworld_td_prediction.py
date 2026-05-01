from random import random, choice
from typing import Dict, List, Tuple, cast

from gridworld_standard_windy import (
    WindyGridworld,
    # standard_gridworld,
    negative_reward_gridworld,
    # assign_random_terminal_rewards,
    State
)

from pretty_printing import print_policy, print_values

# Define types
Action = str
Reward = float
Policy = Dict[State, Action]
ActionValueTable = Dict[State, Dict[Action, Reward]]
SampleCountTable = Dict[State, Dict[Action, int]]
StateSampleCounts = Dict[State, int]
ValueTable = Dict[State, Reward]


# Hyperparameters
GRID_SIZE: Tuple[int, int] = (3, 4)
START_STATE: State = (2, 0)
TERMINAL_STATES: Tuple[State, ...] = ((0, 3), (1, 3))
STEP_COST: float = -0.5
POLICY: Policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }

def epsilon_greedy(
    policy: Policy,
    state: State,
    action_map: Dict[State, Tuple[Action, ...]],
    eps: float = 0.1,
) -> Action:
    """
    Selects an action using an epsilon-greedy strategy, which balances exploration and
    exploitation. With probability `1 - eps`, the action is selected based on the given
    policy. With probability `eps`, a random action from the available actions is
    selected.

    Args:
        policy: The mapping from states to actions, representing the current policy.
        state: The state for which an action needs to be selected.
        action_map: A dictionary mapping states to tuples of possible actions.
        eps: A float value representing the probability of choosing a random action
            (exploration). It must be between 0 and 1, inclusive.

    Returns:
        The selected action is based on the epsilon-greedy algorithm.
    """
    if not 0 <= eps <= 1:
        raise ValueError("Epsilon must be between 0 and 1.")

    p = random()

    if state not in policy:
        raise ValueError(f"Policy has no action for state {state}.")

    available_actions = action_map.get(state, ())
    if not available_actions:
        raise ValueError(f"No available actions for non-terminal state {state}.")

    if p < (1 - eps):
        return policy[state]

    return choice(available_actions)

def initialize_values(grid: WindyGridworld) -> ValueTable:
    V = {s: 0 for s in grid.get_all_states()}

    return V

def td_prediction(
    # grid: WindyGridworld,
    policy: Policy,
    alpha: float = 0.1,
    discount_factor: float = 0.9,
    num_iterations: int = 10_000
) -> Tuple[ValueTable, List[float]]:

    # Initialize environment
    rows, cols = GRID_SIZE
    # g = standard_gridworld(rows, cols, START_STATE, terminal_states=TERMINAL_STATES)
    g = negative_reward_gridworld(rows, cols, START_STATE, terminal_states=TERMINAL_STATES, step_cost=STEP_COST)

    print("Initial policy")
    print_policy(policy, g)
    print()

    # Initialize variables
    V = initialize_values(g)
    print("Initial state values")
    print_values(V, g)
    print()

    changes: List[float] = []

    action_map: Dict[State, Tuple[Action, ...]] = cast(
        Dict[State, Tuple[Action, ...]],
        g.get_action_space(),
    )

    for _ in range(num_iterations):
        # Reset the environment to the starting state
        state = START_STATE
        g.set_state(state)

        #Initialize change variable
        change = 0

        # Iterate through the states as per policy until game is over
        while not g.end_episode():
            a: Action = epsilon_greedy(policy, state, action_map)
            r: Reward = g.move(a)
            next_state: State = g.current_state()

            #Update state returnValue
            v_old = V[state]
            V[state] += alpha * (r + discount_factor * V[next_state] - V[state])
            change = max(change, abs(v_old - V[state]))

            state = next_state

        changes.append(change)

    print("Final policy")
    print_policy(policy, g)
    print()
    print("Final state values")
    print_values(V, g)


    return V, changes

def plot_changes(changes: List[float]) -> None:
    import matplotlib.pyplot as plt

    plt.plot(changes)
    plt.xlabel('Episode')
    plt.ylabel('Max Change in Value')
    plt.title('TD(0) Prediction Convergence', fontweight='bold')
    plt.show()

def main() -> None:
    V, changes = td_prediction(POLICY)
    plot_changes(changes)
    return None

if __name__ == "__main__":
    main()









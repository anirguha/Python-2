from random import choice, shuffle
from typing import Dict, Iterator, List, Tuple, cast

from gridworld_standard_windy import (
    WindyGridworld,
    # standard_gridworld,
    negative_reward_gridworld,
    # assign_random_terminal_rewards,
    State
)
from pretty_printing import print_policy, print_values

GRID_SIZE: Tuple[int, int] = (3, 4)
START_STATE: State = (2, 0)
ALL_POSSIBLE_ACTIONS: Tuple[str, ...] = ('U', 'D', 'L', 'R')
Action = str
Reward = float
Policy = Dict[State, Action]
ActionValueTable = Dict[State, Dict[Action, Reward]]
SampleCountTable = Dict[State, Dict[Action, int]]
TERMINAL_STATES: Tuple[State, ...] = ((0, 3), (1, 3))
GAMMA: float = 0.9


def play_episode(
    grid: WindyGridworld,
    policy: Policy,
    max_steps: int = 20,
) -> Tuple[List[State], List[Action], List[Reward]]:
    """
    Plays a single episode in the WindyGridworld environment following a given policy.

    The function initiates an episode from the START_STATE based on the supplied
    policy. It then follows the policy to select actions, updates the environment, and
    records the sequence of states and rewards until the episode terminates or the maximum
    number of steps is reached.

    Args:
        grid (WindyGridworld): The WindyGridworld environment in which the episode will be played.
        policy (Policy): A policy mapping states to actions, guiding the agent's behavior.
        max_steps (int): Optional; Maximum number of steps to play the episode. Defaults to 20.

    Returns:
        Tuple[List[State], List[Action], List[Reward]]:
            A tuple containing three lists:
            - List of states visited during the episode.
            - List of actions taken during the episode in the same order as the states.
            - List of rewards obtained during the episode in the same order as the states.
    """
    # Returns a list of states and rewards from the current state
    # start at random state and play the episode until max_steps or episode ends
    start_states: List[State] = list(policy.keys())
    start_state: State = choice(start_states)
    grid.set_state(start_state)
    s: State = grid.current_state()

    initial_action_space = cast(Tuple[Action, ...], grid.get_action_space(s))
    initial_actions = list(initial_action_space)
    if not initial_actions:
        raise ValueError(f"Exploring start landed on terminal state {s}, which has no actions.")
    a: Action = choice(initial_actions)

    # Initialize the states and rewards lists
    states: List[State] = [s]
    rewards: List[Reward] = [0]
    actions: List[Action] = []

    steps: int = 0

    while steps < max_steps:
        actions.append(a)
        r: Reward = grid.move(a)
        s = grid.current_state()
        states.append(s)
        rewards.append(r)
        steps += 1

        if grid.end_episode():
            break

        a = policy[s]

    return states, actions, rewards

def get_max_key_value(d: Dict[Action, float]) -> Tuple[Action, float]:
    """
    Finds the key with the maximum value in a dictionary and returns the key along with the value.

    This function identifies the maximum value within the input dictionary and retrieves
    all keys corresponding to that value. When multiple keys share the maximum value, one
    key is chosen at random from the set of matching keys and returned along with the
    maximum value.

    Args:
        d: A dictionary where keys are actions and values are floats.

    Returns:
        A tuple where the first element is an action from the input dictionary and the second
        element is the maximum value (a float) from the dictionary.
    """
    max_val = max(d.values())
    max_keys = [k for k, v in d.items() if v == max_val]

    max_key = choice(max_keys)

    return max_key, max_val

def create_random_policy(g: WindyGridworld) -> Iterator[Tuple[State, Action]]:
    """
    Generates a random policy for the given WindyGridworld environment.

    Args:
        g: An instance of the WindyGridworld environment.

    Yields:
        A tuple containing a state and a randomly selected action for that state.
    """
    action_map: Dict[State, Tuple[Action, ...]] = cast(Dict[State, Tuple[Action, ...]], g.get_action_space())
    states: List[State] = list(action_map.keys())
    shuffle(states) # Shuffle the states to randomize the order of actions

    for s in states:
        available_actions = action_map[s]
        if not available_actions:
            continue
        action = choice(available_actions)
        yield s, action

def initialize_values_returns(g: WindyGridworld) -> tuple[ActionValueTable, SampleCountTable]:
    """
    Initialize Q-values and sample counts for all states and actions in the provided WindyGridworld.

    This function initializes two data structures used in reinforcement learning: a dictionary
    to store Q-values for state-action pairs and another dictionary to keep sample counts for
    state-action pairs. Terminal states in the gridworld are skipped.

    Args:
        g (WindyGridworld): The WindyGridworld environment containing states, actions, and
            terminal conditions.

    Returns:
        Tuple[Dict[State, Reward], Dict[State, List[Reward]]]: A tuple containing:
            - A dictionary mapping each non-terminal state to its actions with initialized Q-values.
            - A dictionary mapping each state to its list of actions with initialized sample counts.
    """
    # Initialize Q(s,a) and returns all states and actions
    Q: ActionValueTable = {}
    sample_counts: SampleCountTable = {}

    action_map = cast(Dict[State, Tuple[Action, ...]], g.get_action_space())
    for s, available_actions in action_map.items():
        Q[s] = {}
        sample_counts[s] = {}
        for a in available_actions:
            Q[s][a] = 0.0
            sample_counts[s][a] = 0

    return Q, sample_counts

def find_optimum_q(g: WindyGridworld,
                   policy: Policy,
                   Q: ActionValueTable,
                   sample_counts: SampleCountTable,
                   num_runs: int) -> list[float]:
    """
    Updates the Q-values for a given policy in a Windy Gridworld environment
    over a specified number of iterations (episodes) and returns the list
    of maximum Q-value changes per iteration for analysis or plotting.

    This function performs iterative updates on the Q-value function based on the
    given environment and policy using a Monte Carlo method. It tracks changes
    in Q-values to evaluate convergence and updates the policy greedily based
    on the updated Q-values.

    Args:
        g (WindyGridworld): The Windy Gridworld environment containing state transitions
            and rewards.
        policy (Policy): Initial policy used for interacting with the environment. This
            policy will be updated based on Q-value improvements during the function
            execution.
        Q (dict[State, Reward]): A mapping of states to dictionaries that contain action-value
            function (Q-values) for each action in that state.
        sample_counts (dict[tuple[tuple[int, int], str], int]): A dictionary mapping state-action
            pairs to the number of times they have been sampled (visited) during
            episodes.
        num_runs (int): The number of episodes to run for updating Q-values and policies.

    Returns:
        list[float]: A list containing the maximum Q-value change for each iteration
            (episode), useful for analyzing the convergence behavior of the learning
            process.
    """
    # Initialize the changes in Q values as a list - to be used for plotting
    changes: List[float] = []
    for it in range(num_runs):
        max_change: float = 0
        states, actions, rewards = play_episode(g, policy)
        states_actions = list(zip(states[:-1], actions))

        return_val: float = 0.0
        for t in range(len(states_actions) - 1, -1, -1):
            s, a = states_actions[t]
            r = rewards[t + 1]
            return_val = r + GAMMA * return_val

            if (s, a) not in states_actions[:t]:
                q_old: float = Q[s][a]
                sample_counts[s][a] += 1
                Q[s][a] = q_old + (1 / sample_counts[s][a]) * (return_val - q_old)

                # Update policy
                policy[s] = get_max_key_value(Q[s])[0]

                # Update change
                max_change = max(max_change, abs(Q[s][a] - q_old))

        changes.append(max_change)
    return changes

def get_state_values_from_q(Q: ActionValueTable) -> Dict[State, Reward]:
    """
    Convert an action-value table into state values using V(s) = max_a Q(s, a).

    Args:
        Q (ActionValueTable): Action-value estimates for each non-terminal state.

    Returns:
        Dict[State, Reward]: State values derived from the greedy action values.
    """
    return {state: max(action_values.values(), default=0.0) for state, action_values in Q.items()}

def plot_changes(changes: List[float], title: str = "Monte Carlo Exploring Start Changes") -> None:
    """
    Plots the changes in Q-values over episodes.

    This function takes a list of changes in Q-values and a title and plots
    these changes against the episodes on a 2D graph. The x-axis represents
    the episodes, and the y-axis represents the maximum change in Q-values
    for the corresponding episode.

    Args:
        changes (List[float]): A list containing the maximum changes in Q-values for
            each episode.
        title (str): The title of the plot. Defaults to "Monte Carlo Exploring Start Changes".
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping change plot.")
        return None

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Maximum Change in Q-value")
    plt.plot(changes)
    plt.show()

    return None

def main() -> None:
    """
    Main function to initialize the gridworld environment, compute optimal Q-values, and
    plot the changes in Q-value updates over time.

    This function sets up a grid world with a negative reward structure, assigns random
    positive rewards to terminal states, and initializes a random action policy. It then
    uses Monte Carlo sampling to find optimal Q-values for the grid world over multiple
    runs and visualizes the progression of Q-value updates.

    Returns:
        None
    """
    rows, cols = GRID_SIZE
    grid = negative_reward_gridworld(rows, cols, START_STATE, terminal_states=TERMINAL_STATES, step_cost=-0.05)
    policy: Policy = dict(create_random_policy(grid))
    Q, sample_counts = initialize_values_returns(grid)
    changes = find_optimum_q(grid, policy, Q, sample_counts, num_runs=10_000)
    plot_changes(changes)
    print("Optimal Policy:")
    print_policy(policy, grid)
    print("Optimal state values from Q:")
    print_values(get_state_values_from_q(Q), grid)




if __name__ == "__main__":
    main()


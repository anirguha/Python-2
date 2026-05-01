from random import choice, shuffle
from typing import Dict, Iterator, List, Tuple

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
# ALL_POSSIBLE_ACTIONS: Tuple[str, ...] = ('U', 'D', 'L', 'R')
Action = str
Reward = float
Policy = Dict[State, Action]
ActionValueTable = Dict[State, Dict[Action, Reward]]
SampleCountTable = Dict[State, Dict[Action, int]]
TERMINAL_STATES: Tuple[State, ...] = ((0, 3), (1, 3))


def play_episode(
    grid: WindyGridworld,
    policy: Policy,
    max_steps: int = 100,
) -> Tuple[List[State], List[Action], List[Reward]]:
    """
    Simulates an episode in the WindyGridworld environment by following a given policy.

    Starting from a random valid state, this function plays an episode by taking actions
    from the provided policy until the maximum allowable steps are reached or the episode ends.
    The function records the sequence of states, actions, and rewards encountered during
    the episode.

    Args:
        grid (WindyGridworld): The WindyGridworld environment to simulate the episode in.
        policy (Policy): A mapping from states to actions, dictating the behavior of the agent.
        max_steps (int, optional): The maximum number of steps to simulate in the episode.
            Defaults to 100.

    Returns:
        Tuple[List[State], List[Action], List[Reward]]: A tuple containing the sequence of
        states, actions, and rewards encountered during the episode.

    Raises:
        ValueError: If the randomly chosen starting state is terminal and has no available
        actions.
    """
    # Returns a list of states and rewards from the current state
    # start at random state and play the episode until max_steps or episode ends
    action_map = grid.get_action_space()
    start_states: List[State] = list(action_map.keys())
    start_state: State = choice(start_states)
    grid.set_state(start_state)
    s: State = grid.current_state()

    initial_action_space = action_map[start_state]
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
        if s not in policy:
            raise ValueError(f"Policy has no action for non-terminal state {s}.")

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
    if not d:
        raise ValueError("Input dictionary is empty.")
    if len(d) == 1:
        return next(iter(d.items()))
    if len(d) == 0:
        raise ValueError("Input dictionary contains no values.")

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
    action_map = g.get_action_space()
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

    action_map = g.get_action_space()
    for s, available_actions in action_map.items():
        Q[s] = {}
        sample_counts[s] = {}
        for a in available_actions:
            Q[s][a] = 0.0
            sample_counts[s][a] = 0

    return Q, sample_counts

def monte_carlo_control_es(g: WindyGridworld,
                   policy: Policy,
                   Q: ActionValueTable,
                   sample_counts: SampleCountTable,
                   gamma: float = 0.9,
                   num_runs: int = 10000,
                   max_steps: int = 100) -> list[float]:
    """
    Uses Monte Carlo control with exploring starts to optimize a policy for a given environment.
    This function applies the Monte Carlo control algorithm to iteratively improve an
    epsilon-greedy policy using first visit and exploring start techniques. The action-value table
    is updated after each episode using the observed returns, and changes are tracked for each
    iteration to analyze convergence.

    Args:
        g (WindyGridworld): The environment where the agent interacts.
        policy (Policy): The epsilon-greedy policy is to be optimized.
        Q (ActionValueTable): The action-value table mapping state-action pairs to values.
        sample_counts (SampleCountTable): A table that tracks how many times each
            state-action pair has been visited.
        gamma (float): The discount factor, controlling the effect of future rewards. The default is 0.9.
        num_runs (int): The number of iterations for running the control algorithm. The default is 10,000.
        max_steps (int): The maximum number of steps allowed per episode. Default is 100.

    Returns:
        list[float]: A list containing the maximum change in Q values at each iteration,
        useful for analyzing the convergence of the algorithm.
    """
    # Initialize the changes in Q values as a list - to be used for plotting
    changes: List[float] = []
    for it in range(num_runs):
        max_change: float = 0
        states, actions, rewards = play_episode(g, policy, max_steps=max_steps)
        states_actions = list(zip(states[:-1], actions))

        first_visit_indices: Dict[Tuple[State, Action], int] = {}
        for t, (s, a) in enumerate(states_actions):
            if (s, a) not in first_visit_indices:
                first_visit_indices[(s, a)] = t

        G: float = 0.0

        for t in range(len(states_actions) - 1, -1, -1):
            s, a = states_actions[t]
            r = rewards[t + 1]
            G = r + gamma * return_val

            if first_visit_indices[(s, a)] == t:
                q_old: float = Q[s][a]
                sample_counts[s][a] += 1
                Q[s][a] = q_old + (1 / sample_counts[s][a]) * (G - q_old)

                # Update policy
                best_action, _ = get_max_key_value(Q[s])
                policy[s] = best_action

                # Update change
                max_change = max(max_change, abs(Q[s][a] - q_old))

        print(f"Iteration {it + 1}: Max Change = {max_change:.4f}")

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
    changes = monte_carlo_control_es(grid, policy, Q, sample_counts, gamma=0.9, num_runs=10_000)
    plot_changes(changes)
    print("Optimal Policy:")
    print_policy(policy, grid)
    print("Optimal state values from Q:")
    print_values(get_state_values_from_q(Q), grid)




if __name__ == "__main__":
    main()


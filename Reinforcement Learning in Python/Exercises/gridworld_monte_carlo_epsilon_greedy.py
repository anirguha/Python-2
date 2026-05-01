from random import random, choice, shuffle
from typing import Dict, List, Tuple, Iterator, cast

import numpy as np
import pandas as pd

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


# Hyperparameters
GRID_SIZE: Tuple[int, int] = (3, 4)
START_STATE: State = (2, 0)
TERMINAL_STATES: Tuple[State, ...] = ((0, 3), (1, 3))
STEP_COST: float = -0.5

def epsilon_greedy(
    policy: Policy,
    state: State,
    action_map: Dict[State, Tuple[Action, ...]],
    eps: float = 0.1,
) -> Action:
    """
    Selects an action using the epsilon-greedy strategy, which chooses the
    optimal action based on the given policy with probability (1 - eps) or
    a random action from available actions with probability eps.

    Args:
        policy (Policy): A mapping from states to actions representing the
            policy for the agent.
        state (State): The current state for which an action needs to be
            selected.
        action_map (Dict[State, Tuple[Action, ...]]): A mapping of states to
            tuples containing the available actions in each state.
        eps (float, optional): The probability of choosing a random action
            instead of the optimal action. Defaults to 0.1.

    Returns:
        Action: The selected action, either the optimal action as defined
        by the policy or a random action from the available actions.

    Raises:
        ValueError: If the given state is not present in the policy.
        ValueError: If there are no available actions for a non-terminal
        state in the provided action map.
    """
    if not 0 <= eps <= 1:
        raise ValueError("Epsilon must be between 0 and 1.")
    if eps == 0:
        return policy[state]
    if eps == 1:
        return choice(action_map[state])

    p = random()

    if state not in policy:
        raise ValueError(f"Policy has no action for state {state}.")

    available_actions = action_map.get(state, ())
    if not available_actions:
        raise ValueError(f"No available actions for non-terminal state {state}.")

    if p < (1 - eps):
        return policy[state]

    return choice(available_actions)

def play_episode(
    grid: WindyGridworld,
    policy: Policy,
    start_state: State = START_STATE,
    max_steps: int = 100,
) -> Tuple[List[State], List[Action], List[Reward]]:
    """
    Runs a single episode of interaction between an agent and the gridworld
    environment using a given policy. The agent starts at a specified state
    and follows the policy until the episode ends or a maximum number of
    steps is reached. The episode captures a sequence of states, actions,
    and rewards.

    Args:
        grid: The WindyGridworld environment that defines the state space,
            actions, and rules of the environment.
        policy: The policy used by the agent to select actions. It defines
            the mapping from state to actions using an epsilon-greedy approach.
        start_state: The initial state where the agent begins the episode. Defaults
            to the constant START_STATE.
        max_steps: The maximum number of steps allowed in the episode. Defaults to 100.

    Returns:
        A tuple containing the following:
            - A list of states visited during the episode.
            - A list of actions taken during the episode.
            - A list of rewards received during the episode.
    """
    action_map: Dict[State, Tuple[Action, ...]] = cast(
        Dict[State, Tuple[Action, ...]],
        grid.get_action_space(),
    )

    # Initialize the grid and start state
    grid.set_state(start_state)
    s = start_state

    # Choose an epsilon greedy action
    a: Action = epsilon_greedy(policy, s, action_map)


    # Initialize the states and rewards lists
    states: List[State] = [s]
    rewards: List[Reward] = []
    actions: List[Action] = [a]

    steps: int = 0

    while steps < max_steps:
        r: Reward = grid.move(a)
        s = grid.current_state()

        states.append(s)
        rewards.append(r)

        steps += 1

        if grid.end_episode():
            break
        else:
            a = epsilon_greedy(policy, s, action_map)
            actions.append(a)

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

def initialize_values_returns(g: WindyGridworld) -> tuple[ActionValueTable, SampleCountTable, StateSampleCounts]:
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
    state_sample_first_visit_counts: StateSampleCounts = {}

    action_map = cast(Dict[State, Tuple[Action, ...]], g.get_action_space())
    for s, available_actions in action_map.items():
        Q[s] = {}
        sample_counts[s] = {}
        state_sample_first_visit_counts[s] = 0
        for a in available_actions:
            Q[s][a] = 0.0
            sample_counts[s][a] = 0

    return Q, sample_counts, state_sample_first_visit_counts

def monte_carlo_control_eg(g: WindyGridworld,
                    policy: Policy,
                    Q: ActionValueTable,
                    sample_counts: SampleCountTable,
                    state_sample_first_visit_counts: StateSampleCounts,
                    gamma: float = 0.9,
                    num_runs: int = 10000,
                    max_steps: int = 100) -> list[float]:
    """
    Performs Monte Carlo control using an epsilon-greedy policy for the WindyGridworld
    environment. This function iteratively improves the action-value function Q and the
    corresponding policy using sample episodes.

    Monte Carlo control is a reinforcement learning method that uses sample episodes
    to estimate the action-value function and improve the policy over time. The algorithm
    discounts future rewards based on the gamma parameter and uses sample counts to
    ensure that learning is based on proper state-action pair visitation frequencies.

    Args:
        g: WindyGridworld instance representing the environment in which the agent interacts.
        policy: Policy object that determines the agent's actions given a state.
        Q: ActionValueTable object mapping state-action pairs to their estimated values.
        sample_counts: SampleCountTable object tracking the frequency of visits
            to each state-action pair.
        state_sample_first_visit_counts: StateSampleCounts object tracking the total
            number of initial visits to each state.
        gamma: Discount factor for future rewards. Must be a float between 0 and 1.
        num_runs: Number of iterations for the learning process. Must be a positive integer.
        max_steps: Maximum number of steps the agent is allowed per episode. Must be a
            positive integer.

    Returns:
        list[float]: A list of maximum Q-value changes for each iteration of the learning
        process. This can be used for analysis or convergence plotting.

    Raises:
        ValueError: If the `gamma` parameter is not between 0 and 1.
        ValueError: If the `num_runs` parameter is not a positive integer.
        ValueError: If the `max_steps` parameter is not a positive integer.
    """
    # Initialize the changes in Q values as a list - to be used for plotting
    changes: List[float] = []

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Discount factor must be between 0 and 1.")
    if num_runs <= 0:
        raise ValueError("Number of runs must be a positive integer.")
    if max_steps <= 0:
        raise ValueError("Maximum steps must be a positive integer.")
    for it in range(num_runs):
        max_change: float = 0
        states, actions, rewards = play_episode(g, policy, start_state=START_STATE, max_steps=max_steps)
        states_actions = list(zip(states[:-1], actions))

        first_visit_indices: Dict[Tuple[State, Action], int] = {}
        for t, (s, a) in enumerate(states_actions):
            if (s, a) not in first_visit_indices:
                first_visit_indices[(s, a)] = t

        G: float = 0.0

        for t in range(len(states_actions) - 1, -1, -1):
            s, a = states_actions[t]
            r = rewards[t]

            # Update G i.e., return
            G = r + gamma * G

            # Check if this is the first visit to this state-action pair

            if first_visit_indices[(s, a)] == t:
                q_old: float = Q[s][a]
                sample_counts[s][a] += 1
                Q[s][a] = q_old + (1 / sample_counts[s][a]) * (G - q_old)

                # Update policy
                best_action, _ = get_max_key_value(Q[s])
                policy[s] = best_action

                # Update state sample count
                state_sample_first_visit_counts[s] += 1

                # Update change
                max_change = max(max_change, abs(Q[s][a] - q_old))

        # print(f"Iteration {it + 1}: Max Change = {max_change:.4f}")

        changes.append(max_change)
    return changes

def plot_changes(changes: List[float]) -> None:
    """
    Plots the changes in values over iterations to visualize the convergence of the
    Monte Carlo Epsilon-Greedy algorithm.

    This function uses matplotlib to plot the changes provided as input. The x-axis
    represents the iteration number, and the y-axis represents the maximum observed
    change during that iteration. The plot includes axis labels and a title for
    clear interpretation.

    Args:
        changes (List[float]): A list of numerical changes, where each element
            represents the maximum change observed in a specific iteration.

    Returns:
        None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping change plot.")
        return None

    plt.plot(changes)
    plt.xlabel('Iteration')
    plt.ylabel('Max Change')
    plt.title('Convergence of Monte Carlo Epsilon-Greedy')

    # Avoid non-interactive backend warnings in headless runs (e.g., Agg backend).
    backend = plt.get_backend().lower()
    if 'agg' in backend:
        plt.close()
    else:
        plt.show()

    return None

def print_action_sample_counts(sample_counts: SampleCountTable, g: WindyGridworld) -> None:
    """
    Prints the total counts of actions sampled for each state in a given gridworld environment.

    This function aggregates action sample counts from the provided sample count table
    into totals for each state and prints them using the provided gridworld environment's
    rendering capability.

    Args:
        sample_counts (SampleCountTable): A dictionary where each key is a state and
            each value is another dictionary mapping actions to their corresponding
            sample counts.
        g (WindyGridworld): The gridworld environment used for rendering the state totals.

    Returns:
        None
    """
    state_totals: Dict[State, float] = {
        state: float(sum(action_counts.values()))
        for state, action_counts in sample_counts.items()
    }
    print_values(state_totals, g)
    return None

def get_state_values_from_q(Q: ActionValueTable) -> Dict[State, Reward]:
    """
    Convert an action-value table into state values using V(s) = max_a Q(s, a).

    Args:
        Q (ActionValueTable): Action-value estimates for each non-terminal state.

    Returns:
        Dict[State, Reward]: State values derived from the greedy action values.
    """
    return {state: max(action_values.values(), default=0.0) for state, action_values in Q.items()}

def main() -> None:
    """
    Executes the main workflow for solving the WindyGridworld problem using Monte Carlo
    control with exploring starts. The script initializes the environment, defines a random
    policy, and iteratively improves the policy based on simulated episodes.

    Workflow includes printing the initial and final policy, calculating the state values,
    and logging sample counts of state visits after the Monte Carlo runs. The performance
    is visualized by plotting the changes in learning over time.

    Raises:
        Any exceptions raised during execution are dependent on the imported modules
        and their implementation.

    """
    # Create the gridworld environment
    rows, cols = GRID_SIZE
    grid: WindyGridworld = negative_reward_gridworld(rows, cols, START_STATE, TERMINAL_STATES, step_cost=STEP_COST)

    # Initialize the policy
    policy: Policy = dict(create_random_policy(grid))
    print("Initial Policy:")
    print_policy(policy, grid)
    print()

    # Play an episode
    Q, sample_counts, state_sample_first_visit_counts = initialize_values_returns(grid)
    changes = monte_carlo_control_eg(
        grid,
        policy,
        Q,
        sample_counts,
        state_sample_first_visit_counts,
        gamma=0.9,
        num_runs=10_000,
    )
    plot_changes(changes)

    print("Final Policy:")
    print_policy(policy, grid)
    print()

    print("Final State Values:")
    print_values(get_state_values_from_q(Q), grid)
    print()

    print("Final state_sample_first_visit_counts:")
    state_sample_counts_arr: np.ndarray = np.zeros((grid.rows, grid.cols))
    for i in range(grid.rows):
        for j in range(grid.cols):
            state_sample_counts_arr[i, j] = state_sample_first_visit_counts.get((i, j), 0)

    df: pd.DataFrame = pd.DataFrame(state_sample_counts_arr)
    print(df)

    # print ("Final action_sample_counts:")
    # print_action_sample_counts(sample_counts, grid)

    return None

if __name__ == "__main__":
    main()

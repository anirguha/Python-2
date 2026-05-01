from random import random, choice, shuffle
from typing import Dict, List, Tuple, Iterator

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
    state_visit_counts: StateSampleCounts, # Denotes the number of times a state has been visited
    truncated_count: int,
    terminated_count: int,
    start_state: State = START_STATE,
    max_steps: int = 100,
) -> Tuple[List[State], List[Action], List[Reward], int, int]:
    """
    Plays a single episode of the WindyGridworld environment using the provided policy and updates the state visit
    counts, terminated, and truncated episode counters.

    The function simulates a sequence of state transitions starting from an initial state. Actions are selected based
    on the provided policy, and the environment responds with rewards and state transitions. The episode terminates
    when either the maximum number of steps is reached or the environment signals the end of the episode.

    Args:
        grid: WindyGridworld instance representing the environment where the episode takes place.
        policy: Policy object used to determine actions based on the current state.
        state_visit_counts: StateSampleCounts, a mapping of states to the number of times they are visited during
            the episode.
        truncated_count: int, counter for episodes terminated due to reaching the maximum step limit.
        terminated_count: int, counter for episodes terminated naturally by reaching an end condition in the environment.
        start_state: State, the initial state for the episode (defaults to START_STATE).
        max_steps: int, the maximum number of steps allowed in the episode (defaults to 100).

    Returns:
        Tuple[List[State], List[Action], List[Reward], int, int]:
            A tuple containing:
            - List of states visited during the episode.
            - List of actions taken during the episode.
            - List of rewards received during the episode.
            - Updated truncated_count.
            - Updated terminated_count.
    """
    action_map = grid.get_action_space()

    # Initialize the grid and start state
    grid.set_state(start_state)
    s = start_state
    state_visit_counts[s] += 1

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
        state_visit_counts[s] += 1

        if grid.end_episode():
            break
        else:
            a = epsilon_greedy(policy, s, action_map)
            actions.append(a)

    if steps == max_steps:
        truncated_count += 1
    else:
        terminated_count += 1

    return states, actions, rewards, truncated_count, terminated_count

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
    action_map = g.get_action_space()
    states: List[State] = list(action_map.keys())
    shuffle(states) # Shuffle the states to randomize the order of actions

    for s in states:
        available_actions = action_map[s]
        if not available_actions:
            continue
        action = choice(available_actions)
        yield s, action

def initialize_values_returns(g: WindyGridworld) -> tuple[
    ActionValueTable, SampleCountTable, StateSampleCounts, StateSampleCounts, int, int]:
    """
    Initializes the action-value function (Q), sample counts for state-action pairs,
    first-visit counts for states, and overall state visit counts for a given
    WindyGridworld environment.

    This function sets up the foundational data structures required for reinforcement
    learning algorithms by preparing the action-value table and other tracking
    dictionaries. Each state-action pair is initialized with default values, and
    visit counts are set to zero.

    Args:
        g (WindyGridworld): The WindyGridworld instance specifying the environment,
            its state space, and the available actions.

    Returns:
        tuple[ActionValueTable, SampleCountTable, StateSampleCounts, StateSampleCounts]:
            A tuple containing:
            - Q: ActionValueTable where each state's actions are initialized to 0.0.
            - sample_counts: A table to track the number of times each state-action
              pair is visited.
            - state_sample_first_visit_counts: A dictionary tracking the first-visit
              count of each state.
            - state_visit_counts: A dictionary tracking the total visit count of each
              state.
    """
    # Initialize Q(s,a) and returns all states and actions
    Q: ActionValueTable = {}
    sample_counts: SampleCountTable = {}
    state_sample_first_visit_counts: StateSampleCounts = {}
    state_visit_counts: StateSampleCounts = {}

    truncated_count = 0
    terminated_count = 0

    action_map = g.get_action_space()
    all_states = g.get_all_states()
    for s, available_actions in action_map.items():
        Q[s] = {}
        sample_counts[s] = {}
        state_sample_first_visit_counts[s] = 0
        for a in available_actions:
            Q[s][a] = 0.0
            sample_counts[s][a] = 0

    for s in all_states:
        state_visit_counts[s] = 0

    return Q, sample_counts, state_sample_first_visit_counts, state_visit_counts, truncated_count, terminated_count

def _validate_monte_carlo_control_inputs(
    gamma: float,
    num_runs: int,
    max_steps: int,
) -> None:
    """Validate Monte Carlo control hyperparameters."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Discount factor must be between 0 and 1.")
    if num_runs <= 0:
        raise ValueError("Number of runs must be a positive integer.")
    if max_steps <= 0:
        raise ValueError("Maximum steps must be a positive integer.")


def _get_first_visit_indices(
    state_actions: List[Tuple[State, Action]],
) -> Dict[Tuple[State, Action], int]:
    """Return the first index at which each state-action pair appears."""
    first_visit_indices: Dict[Tuple[State, Action], int] = {}
    for index, state_action in enumerate(state_actions):
        if state_action not in first_visit_indices:
            first_visit_indices[state_action] = index
    return first_visit_indices


def _update_first_visit_estimate(
    state: State,
    action: Action,
    return_value: float,
    policy: Policy,
    Q: ActionValueTable,
    sample_counts: SampleCountTable,
    state_sample_first_visit_counts: StateSampleCounts,
) -> float:
    """Update the first-visit estimate for a state-action pair and return the change."""
    q_old = Q[state][action]
    sample_counts[state][action] += 1
    Q[state][action] = q_old + (return_value - q_old) / sample_counts[state][action]

    best_action, _ = get_max_key_value(Q[state])
    policy[state] = best_action
    state_sample_first_visit_counts[state] += 1

    return abs(Q[state][action] - q_old)


def monte_carlo_control_eg(
    g: WindyGridworld,
    policy: Policy,
    Q: ActionValueTable,
    sample_counts: SampleCountTable,
    state_sample_first_visit_counts: StateSampleCounts,
    truncated_count: int,
    terminated_count: int,
    state_visit_counts: StateSampleCounts,
    gamma: float = 0.9,
    num_runs: int = 10000,
    max_steps: int = 100,
) -> tuple[list[float], int, int]:
    """
    Performs the Monte Carlo Control with an epsilon-greedy approach to optimize the policy
    for the given environment. The method estimates action-value functions (Q-values) and
    updates the policy iteratively to converge to an optimal policy.

    Args:
        g (WindyGridworld): The environment in which the agent operates.
        policy (Policy): The current policy being evaluated and improved.
        Q (ActionValueTable): The action-value table containing estimates for state-action pairs.
        sample_counts (SampleCountTable): Tracks the number of times each state-action pair
            has been sampled.
        state_sample_first_visit_counts (StateSampleCounts): Stores the count of first-visit
            occurrences for each state.
        truncated_count (int): Counter for episodes that were truncated due to exceeding
            the maximum step limit.
        terminated_count (int): Counter for episodes that successfully terminated in the
            environment's terminal state.
        state_visit_counts (StateSampleCounts): Records the cumulative count of visits
            for each state.
        gamma (float, optional): Discount factor representing the weight of future rewards
            in the return calculation. Must be between 0 and 1. Defaults to 0.9.
        num_runs (int, optional): Number of Monte Carlo iterations to perform. Must be a
            positive integer. Defaults to 10000.
        max_steps (int, optional): Maximum steps allowed per episode. Must be a positive
            integer. Defaults to 100.

    Returns:
        tuple[list[float], int, int]: A tuple containing:
            - A list of floats representing the maximum value changes in Q for each iteration
              (useful for convergence analysis).
            - An integer representing the total count of truncated episodes.
            - An integer representing the total count of successfully terminated episodes.

    Raises:
        ValueError: If `gamma` is not between 0 and 1.
        ValueError: If `num_runs` is not a positive integer.
        ValueError: If `max_steps` is not a positive integer.
    """
    _validate_monte_carlo_control_inputs(gamma, num_runs, max_steps)

    changes: List[float] = []
    for _ in range(num_runs):
        max_change = 0.0
        states, actions, rewards, truncated_count, terminated_count = play_episode(
            g,
            policy,
            state_visit_counts,
            truncated_count,
            terminated_count,
            start_state=START_STATE,
            max_steps=max_steps,
        )
        state_actions = list(zip(states[:-1], actions))
        first_visit_indices = _get_first_visit_indices(state_actions)

        return_value = 0.0
        for index in range(len(state_actions) - 1, -1, -1):
            state, action = state_actions[index]
            reward = rewards[index]
            return_value = reward + gamma * return_value

            if first_visit_indices[(state, action)] != index:
                continue

            change = _update_first_visit_estimate(
                state,
                action,
                return_value,
                policy,
                Q,
                sample_counts,
                state_sample_first_visit_counts,
            )
            max_change = max(max_change, change)

        changes.append(max_change)

    return changes, truncated_count, terminated_count

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

    # Show interactively on GUI backends, save on headless/non-interactive backends.
    backend = plt.get_backend().lower()
    non_interactive_backends = {"agg", "pdf", "pgf", "ps", "svg", "template", "cairo"}
    if backend in non_interactive_backends or backend.startswith("module://matplotlib_inline"):
        output_path = "mc_eg_convergence.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved convergence plot to {output_path} (backend={backend}).")
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

def print_state_sample_counts(sample_counts: StateSampleCounts, g: WindyGridworld) -> None:
    """
    Prints total sample counts for each state in the gridworld.

    Args:
        sample_counts (StateSampleCounts): A dictionary mapping states to visit counts.
        g (WindyGridworld): The gridworld environment used for rendering the counts.

    Returns:
        None
    """
    state_totals: Dict[State, float] = {
        state: float(count)
        for state, count in sample_counts.items()
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
    grid: WindyGridworld = negative_reward_gridworld(
        rows,
        cols,
        START_STATE,
        TERMINAL_STATES,
        step_cost=STEP_COST,
    )

    # Initialize the policy
    policy: Policy = dict(create_random_policy(grid))
    print("Initial Policy:")
    print_policy(policy, grid)
    print()

    # Initialize values and returns
    (
        Q,
        sample_counts,
        state_sample_first_visit_counts,
        state_visit_count,
        truncated_count,
        terminated_count,
    ) = initialize_values_returns(grid)

    # Run Monte Carlo control with epsilon-greedy policy improvement
    changes, truncated_count, terminated_count = monte_carlo_control_eg(
        grid,
        policy,
        Q,
        sample_counts,
        state_sample_first_visit_counts,
        truncated_count,
        terminated_count,
        state_visit_count,
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

    print("Final state_visit_counts:")
    print_state_sample_counts(state_visit_count, grid)
    print()

    print("Final state_sample_first_visit_counts:")
    state_sample_counts_rows: List[List[int]] = []
    for i in range(grid.rows):
        row: List[int] = []
        for j in range(grid.cols):
            row.append(state_sample_first_visit_counts.get((i, j), 0))
        state_sample_counts_rows.append(row)

    for row in state_sample_counts_rows:
        print(row)

    print()
    print(f"Total Truncated Episodes: {truncated_count}")
    print(f"Total Terminated Episodes: {terminated_count}")

    # print ("Final action_sample_counts:")
    # print_action_sample_counts(sample_counts, grid)

    return None

if __name__ == "__main__":
    main()

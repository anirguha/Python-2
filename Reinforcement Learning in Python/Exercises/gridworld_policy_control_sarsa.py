from random import choice, random
from typing import Dict, List, Tuple

from gridworld_standard_windy import (
    WindyGridworld,
    # standard_gridworld,
    negative_reward_gridworld,
    # assign_random_terminal_rewards,
    State
)

from gridworld_monte_carlo_epsilon_greedy import get_max_key_value

from pretty_printing import print_policy, print_values

# Define types
Action = str
Reward = float
Policy = Dict[State, Action]
ActionValueTable = Dict[State, Dict[Action, Reward]]
SampleCountTable = Dict[State, Dict[Action, int]]
StateSampleCounts = Dict[State, float]
ValueTable = Dict[State, Reward]


# Hyperparameters
GRID_SIZE: Tuple[int, int] = (3, 4)
START_STATE: State = (2, 0)
TERMINAL_STATES: Tuple[State, ...] = ((0, 3), (1, 3))
STEP_COST: float = -0.5

def epsilon_greedy(Q: ActionValueTable, s: State, epsilon: float) -> Action:
    """
    Selects an action using the epsilon-greedy policy.

    The epsilon-greedy policy is commonly used in reinforcement learning to balance
    exploration and exploitation in action selection. Given a state, the policy
    chooses a random action with a probability of `epsilon` to explore and selects
    the action with the highest estimated value with a probability of `1 - epsilon`
    to exploit.

    Args:
        Q: Action-value table, where Q[state][action] gives the expected value of
            taking the specified action in the given state.
        s: Current state for which an action is to be selected.
        epsilon: Probability of selecting a random action, must be in the range
            [0, 1].

    Raises:
        ValueError: If `epsilon` is not within the range [0, 1].

    Returns:
        Action: The selected action is based on the epsilon-greedy policy.
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("Epsilon must be between 0 and 1.")

    available_actions = tuple(Q[s].keys())
    if not available_actions:
        raise ValueError(f"No available actions for state {s}.")

    if random() < epsilon:
        return choice(list(available_actions)) # Explore: Select a random action from the available actions in state s
    else:
        arg, _ = get_max_key_value(Q[s]) # Exploit: Select the action with the highest estimated value
        return arg

def initialize_action_values(g: WindyGridworld) -> ActionValueTable:
    """
    Initializes the action-value table for a WindyGridworld environment.

    This function creates and returns an action-value table (Q) as a dictionary
    mapping each state to another dictionary. This nested dictionary maps each
    available action in the state to a corresponding value, which is initialized
    to 0.0. The structure ensures that only valid state-action pairs are included
    in the table, derived from the action space of the provided environment.

    Args:
        g (WindyGridworld): An instance of the WindyGridworld environment, which
            contains information about the states and the corresponding action
            space for each state.

    Returns:
        ActionValueTable: A dictionary mapping states to inner dictionaries, where
            each inner dictionary maps actions to their initialized values (0.0).
    """
    Q: ActionValueTable = {}
    action_map = g.get_action_space()

    for s, available_actions in action_map.items():
        Q[s] = {a: 0.0 for a in available_actions}

    return Q

def print_action_values(Q: ActionValueTable) -> None:
    """Print state-action values in a compact per-state format."""
    for state in sorted(Q):
        action_values = ", ".join(
            f"{action}: {value:6.2f}"
            for action, value in sorted(Q[state].items())
        )
        print(f"{state}: {action_values}")
    return None

def sarsa(g: WindyGridworld,
          epsilon: float,
          alpha: float,
          gamma: float,
          num_episodes: int) -> Tuple[List[float], ActionValueTable, StateSampleCounts]:
    # Initialize state-action value table Q[s,a]
    Q = initialize_action_values(g)
    states = g.get_all_states()

    # Variable to track the number of times Q[s] has been updated
    update_counts: StateSampleCounts = {}
    for s in states:
        update_counts[s] = 0

    # Variable to track the reward per episode
    reward_per_episode: List[float] = []

    for _ in range(num_episodes):
        # Initialize the state and action
        s = g.start
        g.set_state(s)

        # Find the best action for the current state using the epsilon-greedy policy
        a = epsilon_greedy(Q, s, epsilon=epsilon)
        episode_reward: float = 0.0

        # Iterate until the episode terminates
        while not g.end_episode():
            # Perform the action and get the next state and reward
            r: Reward = g.move(a)
            s_next: State = g.current_state()

            # Update episode reward
            episode_reward += r

            # Update Q(s,a) using the Sarsa update rule
            if g.end_episode():
                td_target = r
                Q[s][a] += alpha * (td_target - Q[s][a])
                update_counts[s] += 1
                break

            a_next = epsilon_greedy(Q, s_next, epsilon=epsilon)
            td_target = r + gamma * Q[s_next][a_next]
            Q[s][a] += alpha * (td_target - Q[s][a])

            # Update the update count for state s
            update_counts[s] += 1

            # Update the state and action
            s = s_next
            a = a_next

        # Log the episode reward
        reward_per_episode.append(episode_reward)

    return reward_per_episode, Q, update_counts

def plot_sarsa_results(reward_per_episode: List[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("matplotlib"):
            print("matplotlib is not installed; skipping SARSA reward plot.")
            return None
        raise

    plt.plot(reward_per_episode, 'b-', label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('SARSA Reward per Episode')
    plt.legend()
    plt.show()

    return None

def run_sarsa(g: WindyGridworld,
              epsilon: float,
              alpha: float,
              gamma: float,
              num_episodes: int) -> None:
    reward_per_episode, Q, update_counts = sarsa(g, epsilon, alpha, gamma, num_episodes)
    plot_sarsa_results(reward_per_episode)

    print("Number of times Q[s] has been updated:")
    total = sum(update_counts.values())
    for s, count in update_counts.items():
        update_counts[s] = count / total
    print_values(update_counts, g)
    print()

    # Determine the optimal policy
    optimal_policy: Policy = {}
    V: ValueTable = {}

    for s in g.get_all_states():
        if s in TERMINAL_STATES:
            optimal_policy[s] = 'TERMINAL'
            V[s] = 0.0
        elif Q[s]:
            a, max_q = get_max_key_value(Q[s])
            optimal_policy[s] = a
            V[s] = max_q
        else:
            V[s] = 0.0

    print("Optimal Policy:")
    print_policy(optimal_policy, g)
    print()
    print("Optimal State Values:")
    print_values(V, g)

    return None


def main():
    rows, cols = GRID_SIZE
    g = negative_reward_gridworld(rows, cols, START_STATE, TERMINAL_STATES, STEP_COST)

    print("Initial Policy:")
    initial_policy: Policy = {}
    for s in g.get_all_states():
        if s in TERMINAL_STATES:
            initial_policy[s] = 'TERMINAL'
        else:
            initial_policy[s] = 'N/A'
    print_policy(initial_policy, g)
    print()
    print("Initial State Values:")
    print_values(g.rewards, g)
    print()
    print("Initial State-Action Values:")
    initial_q = initialize_action_values(g)
    print_action_values(initial_q)
    print()

    run_sarsa(g, epsilon=0.1, alpha=0.5, gamma=0.9, num_episodes=10_000)

if __name__ == "__main__":
    main()

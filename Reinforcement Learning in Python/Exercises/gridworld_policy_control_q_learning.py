from random import choice, random
from typing import Dict, List, Tuple

from gridworld_standard_windy import (
    WindyGridworld,
    # standard_gridworld,
    negative_reward_gridworld,
    # assign_random_terminal_rewards,
    State,
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


def epsilon_greedy_action_selection(
    Q: ActionValueTable,
    state: State,
    epsilon: float = 0.1,
) -> Action:
    """
    Selects an action using the epsilon-greedy policy. This function determines the
    action to take in a given state based on the trade-off between exploration and
    exploitation. With a probability of epsilon, a random action is chosen (exploration),
    while with a probability of (1 - epsilon), the action with the maximum value (exploitation)
    in the Q-table is selected.
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("Epsilon must be between 0 and 1.")

    if state not in Q:
        raise KeyError(f"State {state} not found in Q-table.")

    available_actions = tuple(Q[state].keys())

    if not available_actions:
        raise ValueError(f"No available actions for state {state}.")

    if random() < epsilon:
        return choice(available_actions)

    action, _ = get_max_key_value(Q[state])
    return action


def initialize_action_values(g: WindyGridworld) -> ActionValueTable:
    """Initializes the action-value table for a WindyGridworld environment."""
    Q: ActionValueTable = {}
    action_map = g.get_action_space()

    for s, available_actions in action_map.items():
        Q[s] = {a: 0.0 for a in available_actions}

    return Q


def print_action_values(Q: ActionValueTable) -> None:
    """Print state-action values in a compact per-state format."""
    for state in sorted(Q):
        action_values = ", ".join(
            f"{action}: {value:6.2f}" for action, value in sorted(Q[state].items())
        )
        print(f"{state}: {action_values}")
    return None


def q_learning(
    g: WindyGridworld,
    epsilon: float,
    alpha: float,
    gamma: float,
    num_episodes: int,
) -> Tuple[List[float], ActionValueTable, StateSampleCounts]:
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

        episode_reward: float = 0.0

        # Iterate until the episode terminates
        while not g.end_episode():
            # Find the best action for the current state using the epsilon-greedy policy
            a = epsilon_greedy_action_selection(Q, s, epsilon=epsilon)

            # Perform the action and get the next state and reward
            r: Reward = g.move(a)
            s_next: State = g.current_state()

            # Update episode reward
            episode_reward += r

            # Update Q(s,a) using the q-learning update rule
            if g.end_episode():
                td_target = r
                Q[s][a] += alpha * (td_target - Q[s][a])
                update_counts[s] += 1
                break

            _, max_q = get_max_key_value(Q[s_next])
            td_target = r + gamma * max_q
            Q[s][a] += alpha * (td_target - Q[s][a])

            # Update the update count for state s
            update_counts[s] += 1

            # Update the state and action
            s = s_next

        # Log the episode reward
        reward_per_episode.append(episode_reward)

    return reward_per_episode, Q, update_counts


def plot_q_learning_values(reward_per_episode: List[float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("matplotlib"):
            print("matplotlib is not installed; skipping q-learning reward plot.")
            return None
        raise

    plt.plot(reward_per_episode, "b-", label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-learning Reward per Episode")
    plt.legend()
    plt.show()

    return None


def main() -> None:
    rows, cols = GRID_SIZE
    g = negative_reward_gridworld(rows, cols, START_STATE, TERMINAL_STATES, STEP_COST)

    print("Initial Policy:")
    initial_policy: Policy = {}
    for s in g.get_all_states():
        if s in TERMINAL_STATES:
            initial_policy[s] = "TERMINAL"
        else:
            initial_policy[s] = "N/A"
    print_policy(initial_policy, g)
    print()
    print("Initial State Values:")
    print_values(g.rewards, g)
    print()
    print("Initial State-Action Values:")
    initial_q = initialize_action_values(g)
    print_action_values(initial_q)
    print()

    rewards_per_episode, Q, update_counts = q_learning(
        g, epsilon=0.1, alpha=0.5, gamma=0.9, num_episodes=10_000
    )
    plot_q_learning_values(rewards_per_episode)

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
            optimal_policy[s] = "TERMINAL"
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


if __name__ == "__main__":
    main()

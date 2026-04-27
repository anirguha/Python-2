from random import choice
from typing import List, Mapping, Tuple

from gridworld_standard_windy import State, WindyGridworld, standard_gridworld, negative_reward_gridworld
from pretty_printing import print_policy, print_values

GRID_SIZE: Tuple[int, int] = (3, 4)
START_STATE: State = (2, 0)
Action = str
Reward = float
Policy = Mapping[State, Action]

POLICY: dict[State, str] = {
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

GAMMA = 0.9

def play_episode(
    grid: WindyGridworld,
    policy: Policy,
    max_steps: int = 20,
) -> Tuple[List[State], List[Reward]]:
    """
    Plays a single episode in the WindyGridworld environment following a given policy.

    The function initiates an episode from a random starting state based on the supplied
    policy. It then follows the policy to select actions, updates the environment, and
    records the sequence of states and rewards until the episode terminates or the maximum
    number of steps is reached.

    Args:
        grid (WindyGridworld): The WindyGridworld environment in which the episode will be played.
        policy (Policy): A policy mapping states to actions, guiding the agent's behavior.
        max_steps (int): Optional; Maximum number of steps to play the episode. Defaults to 20.

    Returns:
        Tuple[List[State], List[Reward]]:
            A tuple containing two lists:
            - List of states visited during the episode.
            - List of rewards obtained during the episode in the same order as the states.
    """
    # Returns a list of states and rewards from the current state
    # start at random state and play the episode until max_steps or episode ends
    start_states = list(policy.keys())
    start_state = choice(start_states)
    grid.set_state(start_state)
    s = grid.current_state()

    # Initialize the states and rewards lists
    states: List[State] = [s]
    rewards: List[Reward] = [0]
    steps: int = 0
    while not grid.end_episode() and steps < max_steps:
        a: Action = policy[s]  # get action for the state s from the policy
        r: Reward = grid.move(a)  # move to the next state s' and earn reward r
        next_s = grid.current_state()
        states.append(next_s)
        rewards.append(r)
        s = next_s
        steps += 1

    return states, rewards

def initialize_values_returns(g: WindyGridworld) -> Tuple[dict[State, Reward], dict[State, List[Reward]]]:
    """
    Initializes and returns the value function and returns dictionary for all states.

    This function initializes the value function to store the estimated values for
    each state and returns a dictionary to collect rewards for all visited states
    when performing updates. Terminal states are excluded from the returns' dictionary.

    Args:
        g (WindyGridworld): The WindyGridworld instance representing the
            environment, which is used to retrieve the set of all states and
            identify terminal states.

    Returns:
        Tuple[dict[State, Reward], dict[State, List[Reward]]]: A tuple containing:
            - A dictionary mapping each state to its initial value, defaulting to
              0.0.
            - A dictionary mapping non-terminal states to a list of rewards
              collected for updating the value function.
    """
    states = g.get_all_states()
    values: dict[State, Reward] = {state: 0.0 for state in states}
    returns: dict[State, List[Reward]] = {
        state: [] for state in states if not g.is_terminal(state)
    }

    return values, returns

def train_policy(
    g: WindyGridworld,
    policy: Policy,
    values: dict[State, Reward],
    returns: dict[State, List[Reward]],
    num_episodes: int = 1000,
) -> Tuple[dict[State, Reward], dict[State, List[Reward]]]:
    """
    Trains the given policy by producing episodes and updating state-value functions.

    This function conducts Monte Carlo first-visit policy evaluation to estimate
    the value of states under a given policy. It generates episodes by simulating
    the interaction of the agent with the environment and calculates the returns
    based on the observed rewards. These returns are used to compute and update
    the average value of each visited state.

    Args:
        g: WindyGridworld
            The environment in which the agent interacts during training.
        policy: Policy
            The decision-making policy to be evaluated and improved.
        values: dict[State, Reward]
            A dictionary representing the current estimates of the state's values.
        returns: dict[State, List[Reward]]
            A dictionary storing lists of rewards observed for each state, used to
            calculate the average value of each state.
        num_episodes: int
            The number of episodes to generate for the training process. Defaults
            to 1000.

    Returns:
        Tuple[dict[State, Reward], dict[State, List[Reward]]]: A tuple containing
        the updated state-value functions (values) and the returns of each state.
    """
    for _ in range(num_episodes):
        states, rewards = play_episode(g, policy)
        G = 0.0
        for t in range(len(states) - 2, -1, -1):
            s = states[t]
            r = rewards[t + 1]
            G = r + GAMMA * G
            if s not in states[:t]:  # check if state s is first occurrence in the episode
                returns[s].append(G)
                values[s] = sum(returns[s]) / len(returns[s])  # update value of state s

    return values, returns

def main() -> None:
    """
    Executes the main workflow of training a policy in a gridworld environment.

    This function initializes a gridworld based on a standard size and starting state,
    sets up the value and return structures, trains the policy over a specified number
    of episodes, and then prints the resulting values and policy for the environment.

    Returns:
        None
    """
    rows, cols = GRID_SIZE
    grid = negative_reward_gridworld(rows, cols, START_STATE)

    values, returns = initialize_values_returns(grid)
    values, returns = train_policy(grid, POLICY, values, returns, num_episodes=1000)

    print("-" * 10 + "Values" + "-" * 10)
    print_values(values, grid)
    print("-" * 10 + "Policy" + "-" * 10)
    print_policy(POLICY, grid)


if __name__ == '__main__':
    main()

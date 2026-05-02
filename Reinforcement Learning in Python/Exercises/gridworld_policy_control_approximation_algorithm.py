from __future__ import annotations

from typing import Dict, Tuple, List, Literal, cast
from random import choice, random

import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from gridworld_standard_windy import (
    WindyGridworld,
    standard_gridworld,
    State
)

from pretty_printing import print_values, print_policy


# =======================
# Type Aliases
# =======================
Action = Literal['U', 'D', 'L', 'R']
Reward = float
Policy = Dict[State, Action]
ValueTable = Dict[State, Reward]
StateSampleCounts = Dict[State, int]
Samples = List[np.ndarray]
ActionSpace = Dict[State, Tuple[Action, ...]]


# =======================
# Hyperparameters
# =======================
GRID_SIZE = (3, 4)
START_STATE: State = (2, 0)
TERMINAL_STATES = ((0, 3), (1, 3))

STEP_COST = -0.5


def _build_action_encoding(action_map: ActionSpace) -> Tuple[Dict[Action, int], np.ndarray]:
    """
    Builds an action encoding for the given action space.

    This function creates a mapping of actions to unique indices and generates a one-hot encoded
    matrix that represents the actions. It processes all the actions from the provided action map
    to ensure unique indexing and consistent representation in the one-hot encoding.

    Args:
        action_map (ActionSpace): The mapping of entities to their respective lists of actions.

    Returns:
        Tuple[Dict[Action, int], np.ndarray]: A tuple containing:
            - A dictionary mapping each unique action to a unique integer index.
            - A 2D NumPy array representing the one-hot encoding of the actions.
    """
    discovered_actions = sorted({a for actions in action_map.values() for a in actions})
    action_index: Dict[Action, int] = {
        cast(Action, action): idx for idx, action in enumerate(discovered_actions)
    }
    action_onehot = np.eye(len(action_index))
    return action_index, action_onehot


# =======================
# Utilities
# =======================
def one_hot(k: int, action_onehot: np.ndarray) -> np.ndarray:
    """
    Returns a one-hot encoded vector for the specified index.

    This function selects the one-hot encoded vector corresponding to the given
    index `k` from the input array `action_onehot`. It is used to extract
    predefined one-hot encoded actions from an array based on the provided index.

    Args:
        k (int): The index of the one-hot encoded vector to select.
        action_onehot (np.ndarray): A 2D array of one-hot encoded vectors where
            each row represents an encoded action.

    Returns:
        np.ndarray: The one-hot encoded vector corresponding to the provided index `k`.
    """
    return action_onehot[k]


def merge_state_action(
    s: State,
    a: Action,
    action_index: Dict[Action, int],
    action_onehot: np.ndarray,
) -> np.ndarray:
    """
    Merges a state and an action into a single combined representation by concatenating
    the state as an array and the one-hot encoded representation of the action.

    Args:
        s (State): The current state of the system, represented as a State object.
        a (Action): The action to be applied, represented as an Action object.
        action_index (Dict[Action, int]): A mapping of Action objects to their corresponding indices.
        action_onehot (np.ndarray): A pre-allocated NumPy array for one-hot encoding actions.

    Returns:
        np.ndarray: A NumPy array that combines the state and the one-hot encoded action.
    """
    return np.concatenate((np.array(s), one_hot(action_index[a], action_onehot)))


# =======================
# Policy
# =======================
def epsilon_greedy(g: WindyGridworld,
                   model: ValueFunctionApproximator,
                   s: State,
                   eps: float) -> Action:
    """
    Selects an action for the given state using an epsilon-greedy strategy.

    This function chooses an action from the action space of the given
    WindyGridworld instance. It explores a random action with probability
    `eps` and exploits the action with the highest predicted value with
    probability `1 - eps`.

    Args:
        g: An instance of the WindyGridworld class, representing the
            environment that provides the action space for the agent.
        model: A model object that can predict the values of all possible
            actions for a given state in the environment.
        s: The current state for which an action needs to be selected.
        eps: A float between 0 and 1 specifying the probability of exploration
            (choosing a random action).

    Returns:
        Action: The chosen action for the given state following the specified
        epsilon-greedy policy.
    """
    action_map = g.get_action_space()

    if random() < (1 - eps):
        values = model.predict_all_actions(s)
        return action_map[s][np.argmax(values)]
    else:
        return choice(action_map[s])


# =======================
# Sample Collection
# =======================
def collect_samples(
    g: WindyGridworld,
    num_samples: int,
    action_map: ActionSpace,
    action_index: Dict[Action, int],
    action_onehot: np.ndarray,
) -> Samples:
    """
    Collects samples by simulating episodes in a WindyGridworld environment.

    This function generates a specified number of samples by running episodes in the
    WindyGridworld simulation. Each sample is a combination of the current state and
    an applied action, encoded using one-hot representation. The function continues
    to collect data until the required number of samples is reached.

    Args:
        g: The WindyGridworld environment in which the samples are collected.
        num_samples: The number of samples to collect from the environment.
        action_map: A mapping from states to available actions in the environment.
        action_index: A dictionary mapping each action to its corresponding index.
        action_onehot: A numpy array that provides a one-hot encoding for each action.

    Returns:
        A collection of samples represented as state-action pairs, encoded
        for further processing or analysis.
    """
    samples: Samples = []

    for _ in tqdm(range(num_samples), desc="Collecting Samples"):
        g.set_state(START_STATE)
        s = g.current_state()
        while not g.end_episode():
            a = choice(action_map[s])
            samples.append(merge_state_action(s, a, action_index, action_onehot))

            g.move(a)
            s = g.current_state()

    return samples


# =======================
# Value Function Approximator
# =======================
class ValueFunctionApproximator:

    def __init__(self,
                 g: WindyGridworld,
                 num_samples: int,
                 ):
        """
        Initializes the instance with a WindyGridworld environment and other necessary attributes for
        representation and processing of actions and features.

        Args:
            g (WindyGridworld): The WindyGridworld environment instance.
            num_samples (int): The number of samples to collect for RBF-feature mapping.
        """
        self.g = g

        self.action_map = g.get_action_space()
        self.action_index, self.action_onehot = _build_action_encoding(self.action_map)

        # Collect samples for RBF
        self.samples = collect_samples(
            g,
            num_samples,
            self.action_map,
            self.action_index,
            self.action_onehot,
        )

        # RBF feature mapper
        self.sampler = RBFSampler(n_components=100, gamma=1.0)
        self.sampler.fit(self.samples)

        self.w = np.zeros(self.sampler.n_components)

    def predict(self, s: State, a: Action) -> float:
        """
        Predicts the outcome for a given state and action based on the internal model.

        This method computes the dot product of a transformed representation of the
        state-action pair and an internal weight vector, returning the resulting
        prediction as a float.

        Args:
            s (State): The current state.
            a (Action): The action to take in the given state.

        Returns:
            float: The prediction computed by the model.
        """
        sa = merge_state_action(s, a, self.action_index, self.action_onehot)
        x = self.sampler.transform([sa])[0]
        return float(x @ self.w)

    def grad(self, s: State, a: Action) -> np.ndarray:
        """
        Computes the gradient of the given state-action pair through a transformation.

        This method takes a state and an action, merges them into a single representation,
        and applies the transformation defined in the sampler to compute the gradient.

        Args:
            s (State): The state component of the input.
            a (Action): The action component of the input.

        Returns:
            np.ndarray: The computed gradient as a numpy array.
        """
        sa = merge_state_action(s, a, self.action_index, self.action_onehot)
        return self.sampler.transform([sa])[0]

    def predict_all_actions(self, s: State) -> np.ndarray:
        """
        Predicts all actions for a given state based on the current policy.

        This method takes a state as input, retrieves all possible actions for the
        state, and computes the prediction results using the transformed state-action
        batch and the model's weights.

        Args:
            s (State): The current state for which actions need to be predicted.

        Returns:
            np.ndarray: The predicted results for all actions corresponding to the
            given state.
        """
        actions = self.action_map[s]

        sa_batch = np.array([
            merge_state_action(s, a, self.action_index, self.action_onehot)
            for a in actions
        ])
        X = self.sampler.transform(sa_batch)

        return X @ self.w


# =======================
# Q-Learning with Function Approximation
# =======================
def run_q_learning_approximation(
        g: WindyGridworld,
        num_samples: int,
        epsilon: float,
        lr: float,
        gamma: float,
        num_iterations: int):
    """
    Performs Q-learning with function approximation to train an agent on the WindyGridworld environment.

    This function implements a Q-learning algorithm with a value function approximator to solve
    the WindyGridworld problem. The approximator updates its weights based on the temporal
    difference error during training episodes. Training data includes cumulative rewards and state
    visit counts per episode.

    Args:
        g (WindyGridworld): The WindyGridworld environment on which Q-learning is executed.
        num_samples (int): The number of feature samples for the value function approximation.
        epsilon (float): The epsilon parameter governing the exploration-exploitation tradeoff.
        lr (float): The learning rate for updating the function approximator.
        gamma (float): The discount factor for future rewards.
        num_iterations (int): The number of iterations (episodes) for Q-learning training.

    Returns:
        Tuple[List[float], StateSampleCounts, ValueFunctionApproximator]: A tuple containing:
            - A list of cumulative rewards per episode.
            - A dictionary mapping state samples to visit counts.
            - The trained value function approximator.
    """
    approx = ValueFunctionApproximator(g, num_samples)

    reward_per_episode: List[float] = []
    state_visit_count: StateSampleCounts = {}

    for _ in tqdm(range(num_iterations), desc="Training"):

        g.set_state(START_STATE)
        s = g.current_state()

        episode_reward = 0.0

        while not g.end_episode():

            state_visit_count[s] = state_visit_count.get(s, 0) + 1

            a = epsilon_greedy(g, approx, s, epsilon)
            r = g.move(a)
            s_next = g.current_state()

            # Q-learning target
            if g.is_terminal(s_next):
                target = r
            else:
                target = r + gamma * np.max(approx.predict_all_actions(s_next))

            # Update
            grad = approx.grad(s, a)
            prediction = approx.predict(s, a)

            error = target - prediction
            approx.w += lr * error * grad

            episode_reward += r
            s = s_next

        reward_per_episode.append(episode_reward)

    return reward_per_episode, state_visit_count, approx


# =======================
# Plotting
# =======================
def plot_reward_per_episode(rewards: List[float]):
    """
    Plots the rewards per episode and provides visualization of the performance trend over episodes.

    This function creates a plot of rewards across episodes, labels the axes, and either displays the plot
    or saves it as an image based on the backend being used for Matplotlib.

    Args:
        rewards (List[float]): A list of rewards corresponding to episodes.
    """
    plt.plot(rewards)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    if 'agg' in plt.get_backend().lower():
        plt.savefig('reward_per_episode.png')
        plt.close()
    else:
        plt.show()


# =======================
# Value + Policy Extraction
# =======================
def get_predicted_values(g: WindyGridworld,
                         approx: ValueFunctionApproximator) -> ValueTable:
    """
    Computes the value table for all states in the given WindyGridworld instance
    using predictions from the provided Value Function Approximator. States that
    are part of the action space are evaluated by predicting values for all possible
    actions and selecting the maximum. States outside the action space are assigned
    a value of 0.0.

    Args:
        g: An instance of WindyGridworld that defines the environment and
            its states.
        approx: A ValueFunctionApproximator used to estimate values for
            all possible actions in the environment.

    Returns:
        ValueTable: A dictionary where keys are states of the WindyGridworld
        and values are the corresponding predicted state values.
    """
    V: ValueTable = {}

    for s in g.get_all_states():
        if s in g.get_action_space():
            values = approx.predict_all_actions(s)
            V[s] = np.max(values)
        else:
            V[s] = 0.0

    return V


def get_optimal_policy(g: WindyGridworld,
                        approx: ValueFunctionApproximator):
    """
    Determines the optimal policy and value table for the given WindyGridworld
    using a value function approximation approach.

    The function uses the action space from the provided WindyGridworld and
    predicts the value of all possible actions for each state using the supplied
    value function approximator. The optimal action for each state is selected,
    and the corresponding value is stored in the value table. For states without
    available actions, the value is set to 0.0.

    Args:
        g (WindyGridworld): The WindyGridworld environment, containing states and
            the action space.
        approx (ValueFunctionApproximator): The value function approximator used
            to predict reward values for actions in each state.

    Returns:
        Tuple[Policy, ValueTable]: A tuple containing the optimal policy and the
        value table, where:
            - policy (Policy): A dictionary mapping each state to its optimal action.
            - V (ValueTable): A dictionary mapping each state to its optimal value.
    """
    policy: Policy = {}
    V: ValueTable = {}

    action_map = g.get_action_space()

    for s in g.get_all_states():

        if s in action_map:
            values = approx.predict_all_actions(s)

            best_idx = np.argmax(values)

            V[s] = values[best_idx]
            policy[s] = cast(Action, action_map[s][best_idx])
        else:
            V[s] = 0.0

    return policy, V


# =======================
# State Count Table
# =======================
def get_state_sample_counts(state_visit_count: StateSampleCounts):
    """
    Processes a mapping of state visit counts and converts it to a DataFrame.

    This function takes a dictionary-like mapping of state visit counts and uses the
    grid dimensions to populate a NumPy array with these counts. The array is then
    converted into a Pandas DataFrame for further analysis or visualization.

    Args:
        state_visit_count (StateSampleCounts): A mapping where keys are tuples representing
            grid coordinates (i, j) and values are the corresponding visit counts for
            those states.

    Returns:
        pd.DataFrame: A DataFrame representing the grid, where each cell contains the
        visit count for the corresponding state.
    """
    rows, cols = GRID_SIZE
    arr = np.zeros((rows, cols))

    for (i, j), count in state_visit_count.items():
        arr[i, j] = count

    return pd.DataFrame(arr)


# =======================
# Main
# =======================
def main():
    """
    Main module for running a Q-learning approximation algorithm in a gridworld environment.

    This function initializes a gridworld environment with specified dimensions,
    defines learning parameters, and runs the Q-learning approximation. The results
    include accumulated rewards, state visit counts, an optimal policy, and estimated
    state values. The function also visualizes the reward progression across episodes
    using a plot.

    Returns:
        None
    """
    rows, cols = GRID_SIZE

    g = standard_gridworld(rows, cols, START_STATE, TERMINAL_STATES)

    num_samples = 10_000
    epsilon = 0.1
    lr = 0.01
    gamma = 0.9
    num_iterations = 20_000

    rewards, state_counts, approx = run_q_learning_approximation(
        g, num_samples, epsilon, lr, gamma, num_iterations
    )

    plot_reward_per_episode(rewards)

    print("State Values")
    V = get_predicted_values(g, approx)
    print_values(V, g)
    print()

    print("Optimal Policy")
    policy, _ = get_optimal_policy(g, approx)
    print_policy(policy, g)
    print()

    print("State Visit Counts")
    print(get_state_sample_counts(state_counts))


if __name__ == "__main__":
    main()
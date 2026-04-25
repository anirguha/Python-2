"""Deterministic Gridworld iterative policy evaluation example.

This script defines a small grid environment, builds deterministic
transition/reward tables, and evaluates a fixed policy using Bellman
expectation updates until convergence.
"""

from typing import ClassVar, Dict, KeysView, List, Set, Tuple


State = Tuple[int, int]
Action = str
ActionMap = Dict[State, List[Action]]
Policy = Dict[State, Action]
TransitionKey = Tuple[State, Action, State]
TransitionProbs = Dict[TransitionKey, float]
RewardTable = Dict[TransitionKey, float]
ValueTable = Dict[State, float]


class Gridworld:
    """Small deterministic gridworld environment used for policy evaluation."""

    ACTION_DELTAS: ClassVar[Dict[Action, Tuple[int, int]]] = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }

    def __init__(self):
        """
        Encapsulates the configuration and properties for a grid world environment often used in
        reinforcement learning tasks. The grid world consists of a grid defined by rows and columns,
        a defined starting state, terminal states, rewards, and available actions for each cell.

        Attributes:
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            start_state (tuple): The initial state in the grid, represented as a tuple of (row, column).
            terminal_states (list of tuple): List of terminal states in the grid, where the agent's
                task ends, represented as tuples of (row, column).
            rewards (dict): Mapping of specific grid cells to their rewards, where keys are tuples
                of (row, column) and values are the corresponding rewards.
            discount_factor (float): Discount factor used for calculating returns in reinforcement
                learning, influencing the importance of future rewards.
            negative_rewards (dict): Mapping of specific grid cells to their penalties/negative
                rewards, where keys are tuples of (row, column) and values are the corresponding
                negative rewards.
            actions (dict): Defines the set of allowed actions for each grid cell. Keys are
                tuples of (row, column), and values are lists representing the possible actions
                (e.g., 'U', 'D', 'L', 'R' for up, down, left, and right, respectively).
        """
        self.rows: int = 3
        self.cols: int = 4
        self.start_state: State = (2, 0)
        self.terminal_states: List[State] = [(0, 3), (1, 3)]
        self.rewards: Dict[State, float] = {(0, 3): 1, (1, 3): -1}
        self.discount_factor: float = 1.0
        self.negative_rewards: Dict[State, float] = {
            (0, 0): -1,
            (0, 1): -1,
            (0, 2): -1,
            (1, 0): -1,
            (1, 1): -1,
            (1, 2): -1,
            (2, 0): -1,
            (2, 1): -1,
            (2, 2): -1,
            (2, 3): -1,
        }
        self.actions: ActionMap = {
            (0, 0): ['R', 'D'],
            (0, 1): ['L', 'R'],
            (0, 2): ['L', 'R', 'D'],
            (1, 0): ['U', 'D'],
            (1, 1): ['U', 'R'],
            (1, 2): ['L', 'R', 'D'],
            (2, 0): ['U', 'R'],
            (2, 1): ['L', 'R'],
            (2, 2): ['L', 'R', 'U'],
            (2, 3): ['L', 'U'],
        }

    def get_state_action_space(self) -> Tuple[KeysView[State], ActionMap]:
        """
        Retrieves the state-action space defined by the set of action keys and their corresponding actions.

        Returns:
            tuple[dict_keys, dict]: A tuple containing two elements:
                - A dictionary view object of the action keys.
                - The original dictionary of actions mapping keys to their corresponding action details.
        """
        return self.actions.keys(), self.actions

    def get_state_reward(self, state: State) -> float:
        """
        Retrieves the reward associated with a given state. If the state is not
        found in the primary rewards, it attempts to retrieve it from the negative
        rewards. If not found in either, it defaults to 0.

        Args:
            state: The state for which the reward needs to be retrieved.

        Returns:
            int: The reward value associated with the provided state.
        """
        return self.rewards.get(state, self.negative_rewards.get(state, 0))

    def _apply_action(self, state: State, action: Action) -> State:
        """
        Applies the given action to the current state.

        This method calculates the new state based on the given action and the
        provided state. If the action is valid for the specified state, the position
        is updated accordingly.

        Args:
            state (tuple): The current state represented as a tuple (i, j).
            action: The action to be applied to the state.

        Returns:
            tuple: A tuple (i, j) representing the new state after applying the action.
        """
        i, j = state
        if action in self.actions[state]:
            di, dj = self.ACTION_DELTAS[action]
            i += di
            j += dj
        return (i, j)

    def get_next_state(self, state: State, action: Action) -> State:
        """
        Computes the next state based on the current state and the performed action.

        Args:
            state: The current state of the system.
            action: The action to be applied to the current state.

        Returns:
            The next state of the system after applying the action.
        """
        return self._apply_action(state, action)

    def is_terminal_state(self, state: State) -> bool:
        """
        Checks whether the given state is a terminal state.

        Args:
            state: The state to check.

        Returns:
            bool: True if the state is a terminal state, otherwise False.
        """
        return state in self.terminal_states

    def get_all_states(self) -> Set[State]:
        """
        Retrieves all unique states present in the actions and rewards.

        This method combines the keys from the `actions` and `rewards` dictionaries
        to produce a set of all unique states.

        Returns:
            set: A set containing all unique states present in the actions and
                rewards dictionaries.
        """
        return set(self.actions.keys()) | set(self.rewards.keys())

    def get_action_space(self, state: State) -> List[Action]:
        """
        Retrieves the action space corresponding to a given state.

        The method accesses the action space, represented by a dictionary, where the
        keys correspond to states, and the values are the possible actions that can
        be taken from those states.

        Args:
            state: The state for which the action space is requested.

        Returns:
            The list of possible actions associated with the provided state.
        """
        return self.actions[state]

    def get_num_states(self) -> int:
        """
        Returns the total number of states.

        This method calculates and returns the total number of states available
        by fetching all states and determining their count.

        Returns:
            int: The total count of states.
        """
        return len(self.get_all_states())

    def get_num_actions(self, state: State) -> int:
        """
        Retrieves the number of available actions for the given state.

        Args:
            state: The state for which the number of actions is requested.

        Returns:
            int: The number of available actions for the specified state.
        """
        return len(self.actions[state])

    def __repr__(self):
        """
        Returns a string representation of the Gridworld instance, providing details
        about its rows, columns, starting state, and terminal states.

        Returns:
            str: A string representation of the Gridworld instance, formatted as
                 'Gridworld(rows=<rows>, cols=<cols>, start_state=<start_state>,
                 terminal_states=<terminal_states>)'.
        """
        return (
            f"Gridworld(rows={self.rows}, cols={self.cols}, "
            f"start_state={self.start_state}, terminal_states={self.terminal_states})"
        )


def build_transition_and_reward_tables(grid: Gridworld) -> Tuple[TransitionProbs, RewardTable]:
    """Builds the transition probability and reward tables for a given grid.

    This function initializes and constructs the transition probability
    and reward dictionaries for a given grid, iterating through the possible
    states of the grid and taking into account the actions available at
    each state. For each state-action pair, it determines the resulting next
    state, assigns a transition probability of 1, and stores the reward
    assigned to the next state.

    Args:
        grid: The grid for which transition probabilities and rewards need
            to be computed. It must support methods to check for terminal
            states, retrieve the action space for a state, determine the next
            state given a state-action pair, and retrieve the reward for a
            state.

    Returns:
        A tuple containing:
        - transition_probs (dict): A mapping of tuples (state, action,
          next_state) to their corresponding transition probabilities.
        - rewards (dict): A mapping of tuples (state, action, next_state) to
          their corresponding reward values.
    """
    transition_probs: TransitionProbs = {}
    rewards: RewardTable = {}

    for i in range(grid.rows):
        for j in range(grid.cols):
            s = (i, j)
            if not grid.is_terminal_state(s):
                for a in grid.get_action_space(s):
                    s2 = grid.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1
                    rewards[(s, a, s2)] = grid.get_state_reward(s2)

    return transition_probs, rewards


def initialize_values(grid: Gridworld) -> ValueTable:
    """
    Initializes and returns a dictionary of state values for a given grid. Each state is initialized
    with a default value of 0. For terminal states, their values are directly set to their reward
    value from the grid.

    Args:
        grid: An object that represents the grid environment. It must have the methods `get_all_states`,
            `is_terminal_state`, and `get_state_reward`.

    Returns:
        dict: A dictionary where keys are states from the grid and values are the initialized values
            (0 for non-terminal states and the reward for terminal states).
    """
    values: ValueTable = {}
    for s in grid.get_all_states():
        values[s] = 0
        if grid.is_terminal_state(s):
            values[s] = grid.get_state_reward(s)
    return values


def evaluate_policy(
    grid: Gridworld,
    policy: Policy,
    transition_probs: TransitionProbs,
    rewards: RewardTable,
    values: ValueTable,
) -> None:
    """
    Evaluates a given policy for a grid-based environment using the Bellman expectation equation.

    This function iteratively evaluates a policy by updating the state value function for all
    states in the grid until the values stabilize (converge to a small threshold). It uses the
    provided transition probabilities, rewards, and discount factor from the grid to compute
    the updated values.

    Args:
        grid (Grid): The grid environment, which provides states, actions, terminal state
            information, and discount factor for value updates.
        policy (Dict): A dictionary mapping each state to the recommended action for that state
            according to the policy being evaluated.
        transition_probs (Dict): A dictionary mapping (state, action, next_state) tuples to
            their corresponding transition probabilities, which define the environment dynamics.
        rewards (Dict): A dictionary mapping (state, action, next_state) tuples to their
            corresponding rewards.
        values (Dict): A dictionary mapping each state to its current estimated value, which will
            be updated iteratively during policy evaluation.
    """
    it = 0
    while True:
        max_change = 0
        v1 = 0
        v2 = 0

        for s in grid.get_all_states():
            if not grid.is_terminal_state(s):
                v1 = values[s]
                v2 = 0

                for a in grid.get_action_space(s):
                    for s2 in grid.get_all_states():
                        # Deterministic policy: only one action has probability 1.
                        action_prob = 1 if policy.get(s) == a else 0
                        # Deterministic dynamics: at most one reachable s2 has probability 1.
                        transition_prob = transition_probs.get((s, a, s2), 0)
                        reward = rewards.get((s, a, s2), 0)

                        # Bellman expectation contribution for this (a, s2).
                        v2 += action_prob * transition_prob * (
                            reward + grid.discount_factor * values[s2]
                        )

                values[s] = v2
                max_change = max(max_change, abs(v1 - v2))

        print(f'Earlier Value: {v1} New Value: {v2}')
        print(f"Iteration: {it + 1}, Max Change: {max_change}")
        it += 1

        # Stop when values become stable under the current policy.
        if max_change < 0.0001:
            break


# Instantiate environment and print basic metadata.
grid: Gridworld = Gridworld()
states, actions = grid.get_state_action_space()
num_states: int = grid.get_num_states()
num_actions: int = grid.get_num_actions((0, 0))

print(f"Gridworld Info: {grid}")
print(f"States: {states}")
print(f"Actions: {actions}")
print(f"Number of States: {num_states}")
print(f"Number of Actions in state (0, 0): {num_actions}")

# Build model tables used by policy evaluation.
transition_probs, rewards = build_transition_and_reward_tables(grid)
print(f"Transition Probabilities: {transition_probs}")
print(f"Rewards: {rewards}")

# Deterministic policy mapping state -> action.
policy: Policy = {
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 0): 'U',
    (1, 1): 'U',
    (1, 2): 'L',
    (2, 0): 'U',
    (2, 1): 'R',
    (2, 2): 'U',
    (2, 3): 'L',
}
print(f"Policy: {policy}")

# Initialize and evaluate V(s) under the fixed policy.
values: ValueTable = initialize_values(grid)
evaluate_policy(grid, policy, transition_probs, rewards, values)
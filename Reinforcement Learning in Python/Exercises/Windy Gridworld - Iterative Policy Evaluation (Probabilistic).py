"""Probabilistic Gridworld iterative policy evaluation example.

This script defines a small grid environment, builds probabilistic
transition/reward tables, and evaluates a fixed policy using Bellman
expectation updates until convergence.
"""
import math
import random
from typing import Dict, KeysView, List, Optional, Set, Tuple, Union, overload


State = Tuple[int, int]
Action = str
ActionSpace = Tuple[Action, ...]
ActionMap = Dict[State, ActionSpace]
Policy = Dict[State, Dict[Action, float]]
StateTransitionProbs = Dict[State, float]
StateRewardTable = Dict[State, float]
StateAction = Tuple[State, Action]
TransitionKey = Tuple[State, Action, State]
TransitionProbs = Dict[StateAction, StateTransitionProbs]
TransitionProbTable = Dict[TransitionKey, float]
RewardTable = Dict[TransitionKey, float]
ValueTable = Dict[State, float]

class WindyGridworld:
    """
    Represents a gridworld environment with wind dynamics for reinforcement learning
    applications.

    The WindyGridworld simulates a grid-based environment where an agent navigates
    to reach terminal states while encountering stochastic transitions. The grid is
    initialized with specific dimensions, a starting state, and terminal states. Actions,
    rewards, and transitions must be configured before the environment can be used
    for decision-making tasks.

    Attributes:
        rows (int): Number of rows in the grid.
        Cols (int): Number of columns in the grid.
        Start (State): Starting state of the agent, represented as a tuple (row, column).
        Terminal_states (List[State]): List of terminal states where the environment
            episode ends.
        Probs (TransitionProbs): Transition probabilities define the likelihood of moving
            between states given an action.
        Actions (ActionMap): Mapping of states to the list of available actions from each
            state.
        Rewards (StateRewardTable): Reward structure defining the rewards received for
            reaching specific states.
    """
    def __init__(
            self,
            rows: int,
            cols: int,
            start: State,
            terminal_states: Optional[List[State]] = None,
    ) -> None:
        """
        Initializes a grid-based environment with the given dimensions and starting state.

        Args:
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            start (State): The starting state is represented as a tuple (i, j) where 'i' is
                the row index and 'j' is the column index.
        """
        self.rows = rows
        self.cols = cols
        self._validate_state(start)
        self.start = start
        self.i, self.j = start

        self.probs: TransitionProbs = {}
        self.actions: ActionMap = {}
        self.rewards: StateRewardTable = {}
        self._configured = False

        configured_terminal_states = terminal_states or [(0, 3), (1, 3)]
        for state in configured_terminal_states:
            self._validate_state(state)
        self.terminal_states: List[State] = list(configured_terminal_states)

    def _validate_state(self, state: State) -> None:
        row, col = state
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"State {state} is outside the grid bounds "
                f"{self.rows}x{self.cols}."
            )

    def _require_configuration(self) -> None:
        if not self._configured:
            raise RuntimeError(
                "WindyGridworld must be configured with set() before "
                "accessing transitions, rewards, or actions."
            )

    def get_state_action_space(self) -> Tuple[KeysView[State], ActionMap]:
        """
        Retrieves the state-action space defined by the set of action keys and their corresponding actions.

        Returns:
            tuple[dict_keys, dict]: A tuple containing two elements:
                - A dictionary view object of the action keys.
                - The original dictionary of actions mapping keys to their corresponding action details.
        """
        self._require_configuration()
        return self.actions.keys(), self.actions

    def set(
            self,
            rewards: StateRewardTable,
            actions: ActionMap,
            probs: TransitionProbs,
    ) -> None:
        """
        Sets the reward table, action map, and transition probabilities for the state model.
        The method normalizes the input data, validates each state and action, and ensures
        that the transition probabilities are well-formed. If a terminal state is detected in
        the action map or if any validation checks fail, an appropriate exception is raised.

        Args:
            rewards (StateRewardTable): A mapping from states to their corresponding rewards.
            actions (ActionMap): A mapping from states to sets of available actions for each state.
                Terminal states cannot have associated actions.
            probs (TransitionProbs): A mapping from (state, action) pairs to probabilities of transitioning
                to later states. The probabilities must sum up to 1.0 for each (state, action) pair.
        """
        normalized_actions: ActionMap = {}
        for state, available_actions in actions.items():
            self._validate_state(state)
            if state in self.terminal_states:
                raise ValueError(
                    f"Terminal state {state} cannot have available actions."
                )
            normalized_actions[state] = tuple(available_actions)

        for state in rewards:
            self._validate_state(state)

        normalized_probs: TransitionProbs = {}
        for (state, action), next_state_probs in probs.items():
            self._validate_state(state)
            if state not in normalized_actions:
                raise ValueError(
                    f"Transition defined for state {state} with no action space."
                )
            if action not in normalized_actions[state]:
                raise ValueError(
                    f"Transition defined for invalid action {action!r} in state {state}."
                )
            total_probability = sum(next_state_probs.values())
            if not math.isclose(total_probability, 1.0):
                raise ValueError(
                    f"Transition probabilities for {(state, action)} must sum to 1.0, "
                    f"got {total_probability}."
                )
            normalized_next_states: StateTransitionProbs = {}
            for next_state, probability in next_state_probs.items():
                self._validate_state(next_state)
                normalized_next_states[next_state] = probability
            normalized_probs[(state, action)] = normalized_next_states

        self.rewards = dict(rewards)
        self.actions = normalized_actions
        self.probs = normalized_probs
        self._configured = True


    def set_state(self, s: State) -> None:
        """
        Sets the internal state by unpacking the provided State object into two internal
        attributes.

        Args:
            s (State): A State object whose values will be unpacked and set.
        """
        self._validate_state(s)
        self.i, self.j = s

    def current_state(self) -> Tuple[int, int]:
        """
        Returns the current state of the object as a tuple.

        The method provides access to the current state represented by two integer
        values, which might correspond to specific internal attributes of the
        object.

        Returns:
            Tuple[int, int]: A tuple containing two integer values that represent
            the current state.
        """
        return self.i, self.j

    def is_terminal(self, s: State) -> bool:
        """
        Determines if a given state is terminal in a decision-making process.

        A terminal state is a state that does not have any further actions available.

        Args:
            s (State): The state to check for terminality.

        Returns:
            bool: True if the given state is terminal (has no further actions available),
            False otherwise.
        """
        return s in self.terminal_states

    def move(self, action: Action) -> float:
        """
        Executes an action in the current state, transitions to the next state based
        on the defined probabilities, and returns the reward for the new state.

        Args:
            action: The action to perform from the current state.

        Returns:
            float: The reward associated with the state transitioned to after executing
            the action.
        """
        self._require_configuration()
        s = self.current_state()
        if self.is_terminal(s):
            raise ValueError(f"Cannot move from terminal state {s}.")
        available_actions = self.get_action_space(s)
        if action not in available_actions:
            raise ValueError(
                f"Action {action!r} is not valid for state {s}. "
                f"Available actions: {available_actions}."
            )
        next_state_probs = self.probs.get((s, action))
        if next_state_probs is None:
            raise ValueError(f"No transition probabilities configured for {(s, action)}.")
        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())
        s2 = random.choices(next_states, weights=next_probs, k=1)[0]
        self.set_state(s2)
        return self.rewards.get(s2, 0)

    def end_episode(self) -> bool:
        """
        Determines whether the current episode has ended based on the absence of the current
        state in the available actions.

        Returns:
            bool: True if the current state is not present in the list of possible actions,
            indicating the episode has ended; False otherwise.
        """
        return self.is_terminal(self.current_state())

    def get_all_states(self) -> Set[State]:
        """
        Returns all states present in the system by combining keys from the
        actions and rewards dictionaries.

        This method identifies all unique states by retrieving the keys of
        both the actions and rewards dictionaries and performing a union
        operation.

        Returns:
            Set[State]: All unique states from the actions, rewards, and
            terminal state definitions.
        """
        self._require_configuration()
        return set(self.actions.keys()) | set(self.rewards.keys()) | set(self.terminal_states)

    def get_action_space(
            self,
            state: Optional[State] = None,
    ) -> Union[ActionMap, ActionSpace]:
        self._require_configuration()
        if state is None:
            return self.actions
        return self.actions.get(state, ())

    def is_terminal_state(self, s: State) -> bool:
        return self.is_terminal(s)

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
        return len(self.get_action_space(state))

    def __repr__(self) -> str:
        """
        Returns a string representation of the Gridworld instance, providing details
        about its rows, columns, starting state, and terminal states.

        Returns:
            str: A string representation of the Gridworld instance, formatted as
                 'Gridworld(rows=<rows>, cols=<cols>, start_state=<start_state>,
                 terminal_states=<terminal_states>)'.
        """
        return (
            f"WindyGridworld(rows={self.rows}, cols={self.cols}, "
            f"start_state={self.start}, terminal_states={self.terminal_states})"
        )


ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1),
}


def _next_state(s: State, a: Action) -> State:
    """
    Determine the next state based on the current state and the given action.

    The function calculates the new state coordinates by applying the delta values
    from the action to the current state.

    Args:
        s (State): The current state is represented as a tuple of coordinates.
        a (Action): The action to be performed, which determines the direction
            and magnitude of the state change.

    Returns:
        State: The new state resulting from the applied action.
    """
    di, dj = ACTION_DELTAS[a]
    return s[0] + di, s[1] + dj


def standard_gridworld(rows: int, cols: int, start: State) -> WindyGridworld:
    """
    Creates and returns a standard gridworld environment with specified dimensions
    and starting state. The gridworld is predefined with specific rewards, actions,
    and deterministic transitions.

    Args:
        rows (int): The number of rows in the gridworld.
        cols (int): The number of columns in the gridworld.
        start (State): The starting state of the gridworld.

    Returns:
        WindyGridworld: A configured WindyGridworld environment including defined
        rewards, available actions, and state transition probabilities.
    """
    g = WindyGridworld(rows, cols, start)
    rewards = {(0, 3): 1, (1, 3): -1}
    actions: ActionMap = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    # Build probs keyed by (s, a) as expected by move().
    # Each (s, a) deterministically leads to one next state with probability 1.
    probs: TransitionProbs = {}
    for s, a_list in actions.items():
        for a in a_list:
            s2 = _next_state(s, a)
            probs[(s, a)] = {s2: 1.0}

    g.set(rewards, actions, probs)
    return g


def windy_gridworld(rows: int, cols: int, start: State) -> WindyGridworld:
    """
    Creates a WindyGridworld instance with a specified number of rows, columns, and a starting state.

    In this gridworld, specific grid positions and moves are associated with stochastic wind effects that
    alter the standard transition probabilities. The gridworld is initialized using the `standard_gridworld`
    function and then modified with new transition probabilities to integrate the wind effect.

    Args:
        rows (int): Number of rows in the gridworld.
        cols (int): Number of columns in the gridworld.
        start (State): Starting state of the agent in the gridworld.

    Returns:
        WindyGridworld: A gridworld instance with wind effects on transition probabilities
        in certain grid positions.
    """
    g = standard_gridworld(rows, cols, start)
    windy_probs = {((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5}}
    g.probs.update(windy_probs)
    return g

def negative_reward_gridworld(
        rows: int,
        cols: int,
        start: State,
        step_cost: float = -0.5,
) -> WindyGridworld:
    """
    Creates and returns a WindyGridworld environment where all non-terminal
    states receive a constant negative reward (step cost).

    Args:
        rows (int): Number of rows in the gridworld.
        cols (int): Number of columns in the gridworld.
        start (State): Starting state of the agent.
        step_cost (float): Negative reward is applied to every state at each
            step. Defaults to -0.5.

    Returns:
        WindyGridworld: A configured WindyGridworld instance with the
        given step cost applied to all states.
    """
    g = windy_gridworld(rows, cols, start)

    for s in g.get_all_states():
        if not g.is_terminal(s):
            g.rewards[s] = step_cost
    return g

def build_transition_and_reward_tables(
        grid: WindyGridworld,
) -> Tuple[TransitionProbTable, RewardTable]:
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
    transition_probs: TransitionProbTable = {}
    rewards: RewardTable = {}

    for s in grid.get_all_states():
        if grid.is_terminal(s):
            continue

        for a in grid.get_action_space(s):
            for s2, prob in grid.probs[(s, a)].items():
                transition_probs[(s, a, s2)] = prob
                rewards[(s, a, s2)] = grid.rewards.get(s2, 0)

    return transition_probs, rewards

def initialize_values(grid: WindyGridworld) -> ValueTable:
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
    return values

def evaluate_policy(
        g: WindyGridworld,
        policy: Policy,
        rewards: RewardTable,
        values: ValueTable,
        discount_factor: float = 0.9,
) -> ValueTable:
    """
    Evaluates a given policy using the Bellman expectation update until convergence.

    Args:
        g: The gridworld environment to evaluate the policy on.
        policy: A mapping from states to actions that defines the policy to be evaluated.
        rewards: A dictionary mapping (state, action, next_state) tuples to their corresponding rewards.
        values: A dictionary mapping states to their current estimated value.
        discount_factor: A float value representing the discount factor for future rewards.

    Returns:
        ValueTable: The converged value function mapping each state to its estimated value.
    """
    it = 0
    all_states = g.get_all_states()  # precompute once; the environment is not modified during evaluation

    while True:
        max_change = 0

        for s in all_states:
            if g.is_terminal(s):
                continue

            v_old = values[s]
            v_new = 0.0

            for a in g.get_action_space(s):
                action_prob = policy.get(s, {}).get(a, 0)
                if action_prob == 0:
                    continue

                # Iterate directly over the non-zero next states for (s, a)
                # instead of scanning the entire state space.
                for s2, transition_prob in g.probs.get((s, a), {}).items():
                    r = rewards.get((s, a, s2), 0)
                    v_new += action_prob * transition_prob * (
                        r + discount_factor * values[s2]
                    )

            values[s] = v_new
            max_change = max(max_change, abs(v_old - v_new))

        if max_change < 0.0001:
            print(f"Policy evaluation converged after {it + 1} iterations with max_change={max_change:.6f}.")
            break
        it += 1

    return values


def main() -> None:
    # Instantiate environment and print basic metadata.
    grid: WindyGridworld = windy_gridworld(rows=3, cols=4, start=(2, 0))
    _, actions = grid.get_state_action_space()
    # num_states: int = grid.get_num_states()
    # num_actions: int = grid.get_num_actions((0, 0))

    # print(f"Gridworld Info: {grid}")
    # print(f"States: {states}")
    # print(f"Actions: {actions}")
    # print(f"Number of States: {num_states}")
    # print(f"Number of Actions in state (0, 0): {num_actions}")

    # Build transition and reward tables.
    transition_probs, rewards = build_transition_and_reward_tables(grid)
    print(f"Transition Probabilities: {transition_probs}")
    print(f"Reward Table: {rewards}")

    ### probabilistic policy ###
    policy: Policy = {
        (2, 0): {'U': 0.5, 'R': 0.5},
        (1, 0): {'U': 1.0},
        (0, 0): {'R': 1.0},
        (0, 1): {'R': 1.0},
        (0, 2): {'R': 1.0},
        (1, 2): {'U': 1.0},
        (2, 1): {'R': 1.0},
        (2, 2): {'U': 1.0},
        (2, 3): {'L': 1.0},
    }
    # for i in range(grid.rows):
    #     print(f'-' * 10 + f'Row {i + 1} ' + '-' * 10)
    #     for j in range(grid.cols):
    #         if (i, j) not in policy:
    #             continue
    #         print(f' {j + 1} | ', end="")
    #         action_distribution = policy.get((i, j), "")
    #         print(action_distribution, end="")
    #     print()

    # Initialize values and evaluate the policy.
    values: ValueTable = initialize_values(grid)
    values = evaluate_policy(
        g=grid,
        policy=policy,
        rewards=rewards,
        values=values,
        discount_factor=0.9,
    )

    print("\nConverged state values:")
    for state in sorted(values):
        print(f"  V{state} = {values[state]:.4f}")


if __name__ == "__main__":
    main()

import math
import random
from typing import Dict, KeysView, List, Optional, Set, Tuple, Union


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


def standard_gridworld(
        rows: int,
        cols: int,
        start: State,
        terminal_states: Optional[Tuple[State, ...]] = None,
) -> WindyGridworld:
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
    resolved_terminal_states: List[State] = (
        list(terminal_states) if terminal_states is not None else [(0, 3), (1, 3)]
    )
    g = WindyGridworld(rows, cols, start, terminal_states=resolved_terminal_states)
    default_terminal_rewards: Dict[State, float] = {(0, 3): 1, (1, 3): -1}
    rewards: StateRewardTable = {
        s: default_terminal_rewards.get(s, 0.0) for s in resolved_terminal_states
    }
    all_actions: ActionMap = {
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
    actions: ActionMap = {
        s: available_actions
        for s, available_actions in all_actions.items()
        if s not in resolved_terminal_states
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


def windy_gridworld(
        rows: int,
        cols: int,
        start: State,
        terminal_states: Optional[Tuple[State, ...]] = None,
) -> WindyGridworld:
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
    g = standard_gridworld(rows, cols, start, terminal_states=terminal_states)
    windy_probs = {((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5}}
    g.probs.update(windy_probs)
    return g

def negative_reward_gridworld(
        rows: int,
        cols: int,
        start: State,
        terminal_states: Optional[Tuple[State, ...]] = None,
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
    g = windy_gridworld(rows, cols, start, terminal_states=terminal_states)

    for s in g.get_all_states():
        if not g.is_terminal(s):
            g.rewards[s] = step_cost
    return g


def assign_random_terminal_rewards(
         grid: WindyGridworld,
         terminal_states: Optional[Tuple[State, ...]] = None,
         positive_reward: float = 1.0,
         negative_reward: float = -1.0,
 ) -> State:
    """Assign +reward to one terminal state and -reward to the others.

    Returns the terminal state chosen as the positive terminal.
    """
    target_terminal_states: Tuple[State, ...] = (
        terminal_states if terminal_states is not None else tuple(grid.terminal_states)
    )
    if len(target_terminal_states) < 2:
        raise ValueError("At least two terminal states are required for random +/- assignment.")

    positive_terminal = random.choice(list(target_terminal_states))
    for s in target_terminal_states:
        grid.rewards[s] = positive_reward if s == positive_terminal else negative_reward

    return positive_terminal


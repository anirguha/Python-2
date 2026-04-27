import math
import argparse
from random import choice, choices
from typing import Dict, List, Set, Tuple, KeysView

# Define types
State = Tuple[int, int]
Action = str
ActionSpace = Tuple[Action, ...]
ActionMap = Dict[State, ActionSpace]
Policy = Dict[State, Action]
StateAction = Tuple[State, Action]
TransitionKey = Tuple[State, Action, State]
StateTransitionProbs = Dict[State, float]
TransitionModel = Dict[StateAction, StateTransitionProbs]
TransitionProbs = Dict[TransitionKey, float]
StateRewards = Dict[State, float]
RewardTable = Dict[TransitionKey, float]
ValueTable = Dict[State, float]

# Define constants
ACTION_SPACE: List[Action] = ['U', 'D', 'L', 'R']
WINDY_GRID_SIZE: Tuple[int, int] = (3, 4)
WINDY_GRID_START: State = (2, 0)
ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}
GAMMA: float = 0.9
TOLERANCE: float = 1e-4
STEP_COST: float = -2

PROBS: TransitionModel = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        ((0, 1), 'L'): {(0, 0): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((0, 2), 'L'): {(0, 1): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((2, 1), 'L'): {(2, 0): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'U'): {(1, 2): 1.0},
        ((2, 2), 'L'): {(2, 1): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((2, 3), 'L'): {(2, 2): 1.0},
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
        ((1, 2), 'D'): {(2, 2): 1.0},
        ((1, 2), 'R'): {(1, 3): 1.0},
    }

ACTIONS: ActionMap = {
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

class WindyGridworld:
    """
    Represents a grid-based environment where states, actions, and transitions dictate
    agent movement and rewards. Provides tools to define, configure, and interact with
    a gridworld for simulation or analysis purposes.

    This class is commonly used in reinforcement learning contexts to represent a
    Markov Decision Process (MDP). It allows users to specify the dimensions, starting
    state, terminal states, rewards, action space, and transition probabilities of the
    gridworld.

    Attributes:
        rows (int): Number of rows in the gridworld.
        cols (int): Number of columns in the gridworld.
        start (State): The starting state of the agent in the gridworld.
        probs (TransitionModel): Transition probabilities for state-action pairs.
        actions (ActionMap): Mapping of states to available actions.
        rewards (StateRewards): Rewards associated with states.
        policy (Policy): Placeholder for a policy mapping states to actions.
        terminal_states (List[State]): List of terminal states in the environment.
    """
    def __init__(self, rows: int, cols: int, start: State, terminal_states: List[State] = None):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.i, self.j = start

        self.probs: TransitionModel = {}
        self.actions: ActionMap = {}
        self.rewards: StateRewards = {}
        self._configured = False
        self.policy: Policy = {}

        configured_terminal_states: List[Tuple[int, int]] = terminal_states or [(0, 3), (1, 3)]
        for state in configured_terminal_states:
            self._validate_state(state)
        self.terminal_states: List[State] = list(configured_terminal_states)

    def _validate_state(self, s: State) -> None:
        row, col = s
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"State {s} is outside grid bounds {self.rows}x{self.cols}.")

    def _require_configuration(self) -> None:
        if not self._configured:
            raise RuntimeError("Gridworld must be configured with transition probabilities, actions, and rewards before use.")

    def get_state_action_space(self) -> Tuple[KeysView[State], ActionMap]:
        self._require_configuration()
        return self.actions.keys(), self.actions

    def set(
            self,
            rewards: StateRewards,
            actions: ActionMap,
            probs: TransitionModel,
    ) -> None:
        """
        Sets the model's rewards, actions, and transition probabilities while validating their
        integrity and enforcing normalization where applicable.

        Args:
            rewards: A dictionary where each key represents a state and the associated value
                represents the reward for being in that state.
            actions: A dictionary mapping each state to its corresponding tuple of available
                actions. Actions cannot be associated with terminal states.
            probs: A dictionary where each key is a tuple of (state, action) and the associated
                value is a dictionary mapping possible next states to their transition
                probabilities. Transition probabilities for each (state, action) pair must sum
                up to 1.0.

        Raises:
            ValueError: If terminal states are assigned available actions.
            ValueError: If a state in `probs` has no defined action space.
            ValueError: If an invalid action is defined in `probs` for a state.
            ValueError: If transition probabilities for a (state, action) pair do not sum to 1.0.
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

        normalized_probs: TransitionModel = {}
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
        Updates the current state of the object and ensures that the new state
        is valid. Triggers any required configuration checks after the state
        change.

        Args:
            s (State): The new state to be set.
        """
        self._validate_state(s)
        self.i, self.j = s
        self._require_configuration()

    def current_state(self) -> State:
        """
        Returns the current state of the object.

        This method provides the current values of the object's state, represented
        by attributes `i` and `j`. These values encapsulate the essential information
        required to understand the object's present configuration.

        Returns:
            State: The current state, consisting of the attributes `i` and `j`.
        """
        return self.i, self.j

    def get_next_state(self, s: State, a: Action) -> State:
        """
        Determines the next state based on the current state and action provided. Uses
        probability mapping to compute the next state outcome.

        Args:
            s (State): The current state from which the transition occurs.
            a (Action): The action taken in the current state.

        Returns:
            State: The next state following the action from the current state.

        Raises:
            ValueError: If the provided action is not valid for the given state.
        """
        self._validate_state(s)
        if a not in self.actions.get(s, []):
            raise ValueError(f"Invalid action {a!r} in state {s}.")
        next_state_probs = self.probs[(s, a)]
        next_state = choices(
            population=list(next_state_probs.keys()),
            weights=list(next_state_probs.values()),
            k=1,
        )[0]
        return next_state

    def get_reward(self, s: State) -> float:
        """
        Computes and retrieves the reward value associated with a given state.

        This method takes a state as input, validates it, and returns the
        corresponding reward value from the reward mapping if it exists.
        If the state has no associated reward, a default reward value of 0.0
        is returned.

        Args:
            s (State): The state for which the reward needs to be retrieved.

        Returns:
            float: The reward value associated with the input state. If no reward is
            found for the state, returns 0.0.
        """
        self._validate_state(s)
        return self.rewards.get(s, 0.0)

    def get_terminal_states(self) -> List[State]:
        return self.terminal_states

    def is_terminal_state(self, s: State) -> bool:
        return s in self.terminal_states

    def is_terminal(self, s: State) -> bool:
        return self.is_terminal_state(s)

    def end_episode(self) -> bool:
        return self.is_terminal_state(self.current_state())

    def get_all_states(self) -> Set[State]:
        return set(self.actions.keys()) | set(self.rewards.keys())

    def get_action_space(self, state: State) -> ActionSpace:
        self._require_configuration()
        return self.actions.get(state, ())

    def get_num_states(self) -> int:
        return len(self.get_all_states())

    def get_num_actions(self, state: State) -> int:
        return len(self.get_action_space(state))

    def __repr__(self) -> str:
        return f"GridWorld({self.i}, {self.j}, start={self.start}, terminal_states={self.terminal_states})"

def print_values(V: ValueTable, g: WindyGridworld) -> None:
    """
    Prints the values of each state in a grid representation, using the provided value
    table and grid layout. For states that are not part of the grid's valid states,
    a placeholder 'XXX' is displayed.

    Args:
        V: A value table containing the mapping of states to their respective values.
        g: An instance of WindyGridworld which provides information about grid
            dimensions, state coordinates, and valid states.

    Returns:
        None
    """
    for i in range(g.rows):
        row_items = []
        for j in range(g.cols):
            state = (i, j)
            if state in g.get_all_states():
                row_items.append(f'{V.get(state, 0.0):>7.2f}')
            else:
                row_items.append('   XXX ')
        print(' '.join(row_items))
    return None

def print_policy(p: Policy, g: WindyGridworld) -> None:
    """
    Prints the policy for a WindyGridworld environment.

    This function visualizes the policy by iterating through each cell of the
    WindyGridworld grid and printing the corresponding action for each state.
    If a state is not part of the grid's valid states, it prints 'XXX'. If
    a state is terminal, it prints 'T'. For valid non-terminal states, it
    prints the action specified in the policy.

    Args:
        p (Policy): The policy representing the actions to be taken for each
            state in the grid.
        g (WindyGridworld): The WindyGridworld environment providing information
            about the grid dimensions, states, and terminal states.

    Returns:
        None
    """
    for i in range(g.rows):
        row_items = []
        for j in range(g.cols):
            state = (i, j)
            if state not in g.get_all_states():
                row_items.append(' XXX ')
            elif g.is_terminal(state):
                row_items.append('  T  ')
            else:
                row_items.append(f'{p.get(state, " "):>5}')
        print(' '.join(row_items))
    return None

def windy_grid():
    """
    Initializes a Windy Gridworld environment with a predefined size, start
    location, rewards, actions, and transition probabilities.

    The Windy Gridworld is a grid-based environment where each cell may have
    specific rewards or penalties, and the agent must navigate through the
    grid using available actions. The environment uses predefined transition
    probabilities to determine the likelihood of reaching a particular state
    given an action taken.

    Returns:
        WindyGridworld: An instance of the WindyGridworld environment
        configured with its size, start position, rewards, actions, and
        transition probabilities.
    """
    rows, cols = WINDY_GRID_SIZE
    g = WindyGridworld(rows, cols, start=WINDY_GRID_START)
    global PROBS
    global ACTIONS
    rewards = {(0, 3): 1, (1, 3): -1}

    # p(s' | s, a) represented as:
    # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}

    g.set(rewards, ACTIONS, PROBS)
    return g

def windy_grid_penalized(step_cost: float = STEP_COST) -> WindyGridworld:
    """
    Initializes a WindyGridworld instance with specified step cost and predefined
    rewards for each grid cell. The WindyGridworld instance simulates a grid-based
    environment with both positive and negative terminal rewards and probabilistic
    transitions.

    Args:
        step_cost: Default step cost applied to all non-terminal states in the
            grid. This penalizes every move taken by the agent, encouraging
            efficient paths.

    Returns:
        A configured WindyGridworld instance encompassing rows, columns, start state,
        rewards, possible actions, and transition probabilities.
    """
    global PROBS
    global ACTIONS
    rows, cols = WINDY_GRID_SIZE
    g = WindyGridworld(rows=rows, cols=cols, start=WINDY_GRID_START)

    rewards = {
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
        (0, 3): 1,
        (1, 3): -1
    }

    # p(s' | s, a) represented as:
    # KEY: (s, a) --> VALUE: {s': p(s' | s, a)}

    g.set(rewards, ACTIONS, PROBS)
    return g

def get_transition_probs_and_rewards(g: WindyGridworld) -> Tuple[TransitionProbs, RewardTable]:
    """
    Extracts transition probabilities and rewards from a WindyGridworld object.

    This function processes the state-action transition probabilities and the
    corresponding rewards defined in a WindyGridworld instance, and returns them
    as two separate dictionaries. The probabilities and rewards are indexed by
    (state, action, next_state) tuples.

    Args:
        g (WindyGridworld): An instance of the WindyGridworld containing
            the transition probability and reward information.

    Returns:
        Tuple[TransitionProbs, RewardTable]: A tuple containing the transition
            probabilities and rewards. `TransitionProbs` is a dictionary where
            keys are (state, action, next_state) tuples and values are the
            corresponding probabilities. `RewardTable` is a dictionary where
            keys are (state, action, next_state) tuples and values are the
            rewards for those transitions.
    """
    transition_probs: TransitionProbs = {}
    rewards: RewardTable = {}

    for (state, action), next_state_probs in g.probs.items():
        for next_state, probability in next_state_probs.items():
            transition_probs[(state, action, next_state)] = probability
            reward = g.rewards.get(next_state)
            if reward is not None:
                rewards[(state, action, next_state)] = reward

    return transition_probs, rewards


def build_transition_reward_table(g: WindyGridworld) -> Tuple[TransitionProbs, RewardTable]:
    """
    Builds the transition probability table and reward table for a given WindyGridworld instance.

    This function creates and returns the transition probabilities and associated reward table
    necessary for decision-making in the WindyGridworld environment.

    Args:
        g (WindyGridworld): The WindyGridworld instance for which the transition probabilities
            and reward table are to be generated.

    Returns:
        Tuple[TransitionProbs, RewardTable]: A tuple containing the transition probability table
            and the reward table derived from the given WindyGridworld instance.
    """
    return get_transition_probs_and_rewards(g)

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

def evaluate_deterministic_policy(
        g: WindyGridworld,
        policy: Policy,
        transition_probs: TransitionProbs,
        rewards: RewardTable,
        init_values: ValueTable | None = None,
        discount_factor: float = GAMMA,
) -> ValueTable:
    """
    Evaluates the expected state-values for a deterministic policy in a given gridworld environment.

    This function computes the value of each state under a specific policy iteratively using the Bellman
    expectation equation. The convergence of values is determined by a predefined tolerance.

    Args:
        g (WindyGridworld): The gridworld environment, which defines the states and actions.
        policy (Policy): A mapping of states to actions representing the deterministic policy.
        transition_probs (TransitionProbs): A mapping with keys as (state, action, next_state)
            tuples and values as the transition probabilities between states.
        rewards (RewardTable): A mapping with keys as (state, action, next_state) tuples and values
            representing the reward received for transitioning between states under a given action.
        init_values (ValueTable | None): An optional initial value function for all states. If not
            provided, all states are initialized with a value of 0.0.
        discount_factor (float): The discount factor (γ), representing the importance of future rewards.

    Returns:
        ValueTable: A mapping of states to their computed expected values under the given
        deterministic policy.
    """
    values = init_values.copy() if init_values is not None else {
        state: 0.0 for state in g.get_all_states()
    }
    all_states = g.get_all_states()

    while True:
        max_change = 0.0

        for state in g.actions:
            old_value = values[state]
            action = policy[state]
            new_value = 0.0

            for next_state in all_states:
                probability = transition_probs.get((state, action, next_state), 0.0)
                if probability == 0.0:
                    continue
                reward = rewards.get((state, action, next_state), 0.0)
                new_value += probability * (reward + discount_factor * values[next_state])

            values[state] = new_value
            max_change = max(max_change, abs(old_value - new_value))

        if max_change < TOLERANCE:
            break

    return values

def initialize_random_policy(g: WindyGridworld) -> Policy:
    """
    Initializes a random policy for a given WindyGridworld environment.

    This function iterates over all the states in the given WindyGridworld environment and assigns a random
    action from the allowable action space to each non-terminal state. A random policy can serve as a baseline
    or initialization point for reinforcement learning algorithms.

    Args:
        g (WindyGridworld): A WindyGridworld instance representing the environment for which the policy
            is being generated. Provides access to states, actions, and terminal state checks.

    Returns:
        Policy: A dictionary that maps each non-terminal state in the environment to a randomly chosen
            action from its action space.
    """
    return {
        state: choice(g.get_action_space(state)) for state in g.get_all_states() if not g.is_terminal(state)
    }

def iterate_policy(
        g: WindyGridworld,
        p: Policy,
        transition_probs: TransitionProbs,
        rewards: RewardTable,
        init_values: ValueTable | None = None,
) -> Tuple[Policy,ValueTable]:
    """
    Iterates over policy and value table for a given Gridworld environment to find
    an optimal policy using the policy iteration algorithm. The function alternates
    between policy evaluation and policy improvement steps until the policy
    converges.

    Args:
        transition_probs:
        g (Gridworld): The Gridworld environment containing states, actions,
            and transition dynamics.
        p (Policy): The policy to be optimized, indicating the actions to be taken
            at each state.
        rewards (RewardTable): A lookup table that maps state-action-next_state
            tuples to corresponding reward values.
        init_values (ValueTable, optional): An initial value function dictionary
            mapping states to their estimated values. Defaults to all states
            having zero values if not provided.

    Returns:
        Tuple[Policy, ValueTable]: A tuple containing the updated optimal policy
            and the corresponding value table after convergence.
    """
    if init_values is None:
        init_values = {s: 0.0 for s in g.get_all_states()}

    while True:
        v = evaluate_deterministic_policy(
            g=g,
            policy=p,
            transition_probs=transition_probs,
            rewards=rewards,
            init_values=init_values,
        )

        policy_converged = True
        for state in g.actions:
            old_action = p[state]
            best_action = old_action
            best_value = float('-inf')

            for action in g.get_action_space(state):
                candidate_value = 0.0
                for next_state in g.get_all_states():
                    probability = transition_probs.get((state, action, next_state), 0.0)
                    if probability == 0.0:
                        continue
                    reward = rewards.get((state, action, next_state), 0.0)
                    candidate_value += probability * (reward + GAMMA * v[next_state])

                if candidate_value > best_value:
                    best_value = candidate_value
                    best_action = action

            p[state] = best_action
            if best_action != old_action:
                policy_converged = False

        if policy_converged:
            break

    return p, v

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    This function creates an argument parser to parse the command-line arguments for
    running the policy iteration on the windy gridworld. It provides options to
    customize the step cost of non-terminal states.

    Returns:
        argparse.Namespace: A Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run policy iteration on the windy gridworld."
    )
    parser.add_argument(
        "step_cost",
        type=float,
        nargs='?',
        metavar="STEP_COST",
        default=STEP_COST,
        help=f"Step cost applied to non-terminal states (default: {STEP_COST}).",
    )
    return parser.parse_args()


def main(step_cost: float = STEP_COST) -> None:
    """
    Main program to execute the policy iteration algorithm on a grid world.

    The program initializes a standard grid world environment, sets up the initial
    policy and value function, and iteratively performs policy evaluation and
    policy improvement until the policy converges. After convergence, it outputs
    the final optimal policy and value function.

    Returns:
        None
    """
    g = windy_grid_penalized(step_cost=step_cost)
    transition_probs, rewards = get_transition_probs_and_rewards(g)

    # Initialize policy and values
    v = {s: 0.0 for s in g.get_all_states()}
    p_initial = initialize_random_policy(g)
    print('Initial Values:')
    print_values(v, g)
    print('Initial Policy:')
    print_policy(p_initial, g)

    # Iterate policy evaluation and improvement until convergence
    p_final, v_final = iterate_policy(g, p_initial, transition_probs, rewards, v)
    print('Final Values:')
    print_values(v_final, g)
    print('Final Policy:')
    print_policy(p_final, g)

    return None

if __name__ == '__main__':
    args = parse_args()
    main(step_cost=args.step_cost)

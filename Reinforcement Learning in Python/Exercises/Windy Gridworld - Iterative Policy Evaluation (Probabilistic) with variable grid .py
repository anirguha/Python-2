"""
Dynamic Probabilistic Gridworld Policy Evaluation

This script creates a gridworld of arbitrary size, supports:
- terminal states
- blocked/wall states
- deterministic movement
- optional windy/probabilistic transitions
- fixed or random policies
- iterative Bellman policy evaluation
"""

import math
import random
from typing import Dict, List, Optional, Set, Tuple, Union, overload


# -----------------------------
# Type aliases
# -----------------------------

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


# -----------------------------
# Action definitions
# -----------------------------

ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


# -----------------------------
# Gridworld class
# -----------------------------

class DynamicGridworld:
    def __init__(
        self,
        rows: int,
        cols: int,
        start: State,
        terminal_states: Optional[List[State]] = None,
        blocked_states: Optional[Set[State]] = None,
    ) -> None:
        self.rows = rows
        self.cols = cols

        self._validate_state(start)

        self.start = start
        self.i, self.j = start

        self.terminal_states: List[State] = list(terminal_states) if terminal_states is not None else []
        self.blocked_states: Set[State] = set(blocked_states) if blocked_states is not None else set()

        for s in self.terminal_states:
            self._validate_state(s)

        for s in self.blocked_states:
            self._validate_state(s)

        if start in self.blocked_states:
            raise ValueError(f"Start state {start} cannot be a blocked state.")

        if start in self.terminal_states:
            raise ValueError(f"Start state {start} cannot be a terminal state.")

        self.probs: TransitionProbs = {}
        self.actions: ActionMap = {}
        self.rewards: StateRewardTable = {}

        self._configured = False

    def _validate_state(self, state: State) -> None:
        row, col = state

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(
                f"State {state} is outside grid bounds "
                f"{self.rows}x{self.cols}."
            )

    def _require_configuration(self) -> None:
        if not self._configured:
            raise RuntimeError(
                "Gridworld must be configured before use."
            )

    def set(
        self,
        rewards: StateRewardTable,
        actions: ActionMap,
        probs: TransitionProbs,
    ) -> None:
        # Validate rewards
        for s in rewards:
            self._validate_state(s)

            if s in self.blocked_states:
                raise ValueError(f"Blocked state {s} cannot have a reward.")

        # Validate actions
        for s, available_actions in actions.items():
            self._validate_state(s)

            if s in self.terminal_states:
                raise ValueError(f"Terminal state {s} cannot have actions.")

            if s in self.blocked_states:
                raise ValueError(f"Blocked state {s} cannot have actions.")

            for a in available_actions:
                if a not in ACTION_DELTAS:
                    raise ValueError(f"Invalid action {a!r} at state {s}.")

        # Validate transition probabilities
        for (s, a), next_state_probs in probs.items():
            self._validate_state(s)

            if s not in actions:
                raise ValueError(
                    f"Transition defined for state {s}, "
                    f"but state has no action space."
                )

            if a not in actions[s]:
                raise ValueError(
                    f"Transition defined for invalid action {a!r} "
                    f"from state {s}."
                )

            total_probability = sum(next_state_probs.values())

            if not math.isclose(total_probability, 1.0):
                raise ValueError(
                    f"Transition probabilities for {(s, a)} must sum to 1.0. "
                    f"Got {total_probability}."
                )

            for s2, prob in next_state_probs.items():
                self._validate_state(s2)

                if s2 in self.blocked_states:
                    raise ValueError(
                        f"Transition from {(s, a)} goes into blocked state {s2}."
                    )

                if prob < 0:
                    raise ValueError(
                        f"Transition probability cannot be negative. "
                        f"Got {prob} for {(s, a, s2)}."
                    )

        self.rewards = dict(rewards)
        self.actions = dict(actions)
        self.probs = dict(probs)
        self._configured = True

    def set_state(self, s: State) -> None:
        self._validate_state(s)

        if s in self.blocked_states:
            raise ValueError(f"Cannot set current state to blocked state {s}.")

        self.i, self.j = s

    def current_state(self) -> State:
        return self.i, self.j

    def is_terminal(self, s: State) -> bool:
        return s in self.terminal_states

    def is_blocked(self, s: State) -> bool:
        return s in self.blocked_states

    def get_all_states(self) -> Set[State]:
        return {
            (i, j)
            for i in range(self.rows)
            for j in range(self.cols)
            if (i, j) not in self.blocked_states
        }

    @overload
    def get_action_space(self, state: None = None) -> ActionMap:
        ...

    @overload
    def get_action_space(self, state: State) -> ActionSpace:
        ...

    def get_action_space(
        self,
        state: Optional[State] = None,
    ) -> Union[ActionMap, ActionSpace]:
        self._require_configuration()

        if state is None:
            return self.actions

        return self.actions.get(state, ())

    def get_num_states(self) -> int:
        return len(self.get_all_states())

    def get_num_actions(self, state: State) -> int:
        return len(self.get_action_space(state))

    def move(self, action: Action) -> float:
        """
        Simulates one environment step from the current state.
        This is not needed for policy evaluation, but useful for simulation.
        """
        self._require_configuration()

        s = self.current_state()

        if self.is_terminal(s):
            raise ValueError(f"Cannot move from terminal state {s}.")

        available_actions = self.get_action_space(s)

        if action not in available_actions:
            raise ValueError(
                f"Action {action!r} is invalid from state {s}. "
                f"Available actions: {available_actions}"
            )

        next_state_probs = self.probs[(s, action)]

        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())

        s2 = random.choices(next_states, weights=next_probs, k=1)[0]

        self.set_state(s2)

        return self.rewards.get(s2, 0.0)

    def end_episode(self) -> bool:
        return self.is_terminal(self.current_state())

    def __repr__(self) -> str:
        return (
            f"DynamicGridworld(rows={self.rows}, cols={self.cols}, "
            f"start={self.start}, "
            f"terminal_states={self.terminal_states}, "
            f"blocked_states={self.blocked_states})"
        )


# -----------------------------
# Dynamic grid construction
# -----------------------------

def get_next_state(s: State, a: Action) -> State:
    di, dj = ACTION_DELTAS[a]
    return s[0] + di, s[1] + dj


def is_inside_grid(s: State, rows: int, cols: int) -> bool:
    return 0 <= s[0] < rows and 0 <= s[1] < cols


def make_gridworld(
    rows: int,
    cols: int,
    start: State,
    terminal_rewards: Dict[State, float],
    blocked_states: Optional[Set[State]] = None,
    windy_probs: Optional[TransitionProbs] = None,
) -> DynamicGridworld:
    """
    Creates a dynamic gridworld of arbitrary size.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        start: Start state.
        terminal_rewards: Dictionary of terminal states and their rewards.
            Example: {(0, 4): 1.0, (1, 4): -1.0}
        blocked_states: Wall states that cannot be entered.
        windy_probs: Optional stochastic transition overrides.
            Example:
            {
                ((2, 3), "U"): {
                    (1, 3): 0.5,
                    (2, 4): 0.5,
                }
            }

    Returns:
        A configured DynamicGridworld.
    """
    resolved_blocked: Set[State] = set(blocked_states) if blocked_states is not None else set()
    terminal_states = list(terminal_rewards.keys())

    grid = DynamicGridworld(
        rows=rows,
        cols=cols,
        start=start,
        terminal_states=terminal_states,
        blocked_states=resolved_blocked,
    )

    rewards: StateRewardTable = dict(terminal_rewards)
    actions: ActionMap = {}
    probs: TransitionProbs = {}

    all_states = {
        (i, j)
        for i in range(rows)
        for j in range(cols)
    }

    valid_states = all_states - resolved_blocked

    for s in valid_states:
        if s in terminal_states:
            continue

        available_actions: List[Action] = []

        for a, (di, dj) in ACTION_DELTAS.items():
            s2: State = (s[0] + di, s[1] + dj)

            if not is_inside_grid(s2, rows, cols):
                continue

            if s2 in resolved_blocked:
                continue

            available_actions.append(a)

            # Default deterministic transition
            probs[(s, a)] = {s2: 1.0}

        actions[s] = tuple(available_actions)

    # Override deterministic transitions with windy transitions
    if windy_probs is not None:
        for (s, a), next_state_probs in windy_probs.items():
            if s not in valid_states:
                raise ValueError(f"Windy transition starts from invalid state {s}.")

            if s in terminal_states:
                raise ValueError(f"Windy transition cannot start from terminal state {s}.")

            if a not in actions.get(s, ()):
                raise ValueError(f"Action {a!r} is not valid from state {s}.")

            total_probability = sum(next_state_probs.values())

            if not math.isclose(total_probability, 1.0):
                raise ValueError(
                    f"Windy probabilities for {(s, a)} must sum to 1.0. "
                    f"Got {total_probability}."
                )

            for s2 in next_state_probs:
                if s2 not in valid_states:
                    raise ValueError(
                        f"Windy transition from {(s, a)} goes to invalid state {s2}."
                    )

            probs[(s, a)] = dict(next_state_probs)

    grid.set(
        rewards=rewards,
        actions=actions,
        probs=probs,
    )

    return grid


# -----------------------------
# Build transition and reward tables
# -----------------------------

def build_transition_and_reward_tables(
    grid: DynamicGridworld,
) -> Tuple[TransitionProbTable, RewardTable]:
    transition_probs: TransitionProbTable = {}
    rewards: RewardTable = {}

    for s in grid.get_all_states():
        if grid.is_terminal(s):
            continue

        for a in grid.get_action_space(s):
            for s2, prob in grid.probs[(s, a)].items():
                transition_probs[(s, a, s2)] = prob
                rewards[(s, a, s2)] = grid.rewards.get(s2, 0.0)

    return transition_probs, rewards


# -----------------------------
# Value initialization
# -----------------------------

def initialize_values(grid: DynamicGridworld) -> ValueTable:
    """
    Initialize all state values to 0.

    For episodic RL, terminal values are usually 0 because the reward is
    received when entering the terminal state.
    """
    values: ValueTable = {}

    for s in grid.get_all_states():
        values[s] = 0.0

    return values


# -----------------------------
# Policy construction
# -----------------------------

def make_random_policy(grid: DynamicGridworld) -> Policy:
    """
    Creates an equal-probability random policy for every non-terminal state.
    """
    policy: Policy = {}

    for s in grid.get_all_states():
        if grid.is_terminal(s):
            continue

        actions = grid.get_action_space(s)

        if len(actions) == 0:
            continue

        action_prob = 1.0 / len(actions)

        policy[s] = {
            a: action_prob
            for a in actions
        }

    return policy


def make_greedy_right_up_policy(grid: DynamicGridworld) -> Policy:
    """
    A simple hand-coded policy:
    Prefer R if possible, otherwise prefer U, otherwise pick the first available action.

    This is just an example of a dynamic fixed policy.
    """
    policy: Policy = {}

    for s in grid.get_all_states():
        if grid.is_terminal(s):
            continue

        actions = grid.get_action_space(s)

        if len(actions) == 0:
            continue

        if "R" in actions:
            chosen_action = "R"
        elif "U" in actions:
            chosen_action = "U"
        else:
            chosen_action = actions[0]

        policy[s] = {chosen_action: 1.0}

    return policy


# -----------------------------
# Bellman policy evaluation
# -----------------------------

def evaluate_policy(
    grid: DynamicGridworld,
    policy: Policy,
    rewards: RewardTable,
    values: ValueTable,
    discount_factor: float = 0.9,
    tolerance: float = 1e-4,
    max_iterations: int = 10_000,
) -> ValueTable:
    """
    Iterative policy evaluation using the Bellman expectation equation.

    V(s) = sum_a pi(a|s) sum_s 'P(s'|s, a) [r + gamma V(s')]
    """
    all_states = grid.get_all_states()

    for iteration in range(max_iterations):
        max_change = 0.0

        for s in all_states:
            if grid.is_terminal(s):
                continue

            v_old = values[s]
            v_new = 0.0

            for a in grid.get_action_space(s):
                action_prob = policy.get(s, {}).get(a, 0.0)

                if action_prob == 0:
                    continue

                for s2, transition_prob in grid.probs[(s, a)].items():
                    r = rewards.get((s, a, s2), 0.0)

                    v_new += action_prob * transition_prob * (
                        r + discount_factor * values[s2]
                    )

            values[s] = v_new
            max_change = max(max_change, abs(v_old - v_new))

        if max_change < tolerance:
            print(
                f"Policy evaluation converged after {iteration + 1} iterations "
                f"with max_change={max_change:.6f}."
            )
            return values

    print(
        f"Warning: policy evaluation did not converge after "
        f"{max_iterations} iterations."
    )

    return values


# -----------------------------
# Pretty printing
# -----------------------------

def print_policy(grid: DynamicGridworld, policy: Policy) -> None:
    print("\nPolicy:")

    for i in range(grid.rows):
        row_items = []

        for j in range(grid.cols):
            s = (i, j)

            if grid.is_blocked(s):
                row_items.append(" WALL ".center(12))
            elif grid.is_terminal(s):
                row_items.append(" TERM ".center(12))
            else:
                action_dist = policy.get(s, {})
                if not action_dist:
                    row_items.append(" None ".center(12))
                else:
                    policy_str = ",".join(
                        f"{a}:{p:.2f}"
                        for a, p in action_dist.items()
                    )
                    row_items.append(policy_str.center(12))

        print(" ".join(row_items))


def print_values(grid: DynamicGridworld, values: ValueTable) -> None:
    print("\nConverged state values:")

    for i in range(grid.rows):
        row_items = []

        for j in range(grid.cols):
            s = (i, j)

            if grid.is_blocked(s):
                row_items.append(" WALL ".center(10))
            elif grid.is_terminal(s):
                row_items.append(f"{values.get(s, 0.0):8.4f}")
            else:
                row_items.append(f"{values.get(s, 0.0):8.4f}")

        print(" ".join(row_items))


def print_transition_table(
    transition_probs: TransitionProbTable,
    rewards: RewardTable,
) -> None:
    print("\nTransition and reward table:")

    for key in sorted(transition_probs):
        s, a, s2 = key
        prob = transition_probs[key]
        reward = rewards.get(key, 0.0)

        print(
            f"From {s}, action {a!r} -> {s2} "
            f"with prob={prob:.2f}, reward={reward:.2f}"
        )


# -----------------------------
# Example 1: original-style 3x4 grid
# -----------------------------

def run_original_style_3x4_example() -> None:
    print("\n" + "=" * 60)
    print("Example 1: Original-style 3x4 windy gridworld")
    print("=" * 60)

    grid = make_gridworld(
        rows=3,
        cols=4,
        start=(2, 0),
        terminal_rewards={
            (0, 3): 1.0,
            (1, 3): -1.0,
        },
        blocked_states={
            (1, 1),
        },
        windy_probs={
            ((1, 2), "U"): {
                (0, 2): 0.5,
                (1, 3): 0.5,
            }
        },
    )

    transition_probs, rewards = build_transition_and_reward_tables(grid)

    # Same style as your earlier fixed policy
    policy: Policy = {
        (2, 0): {"U": 0.5, "R": 0.5},
        (1, 0): {"U": 1.0},
        (0, 0): {"R": 1.0},
        (0, 1): {"R": 1.0},
        (0, 2): {"R": 1.0},
        (1, 2): {"U": 1.0},
        (2, 1): {"R": 1.0},
        (2, 2): {"U": 1.0},
        (2, 3): {"L": 1.0},
    }

    values = initialize_values(grid)

    values = evaluate_policy(
        grid=grid,
        policy=policy,
        rewards=rewards,
        values=values,
        discount_factor=0.9,
    )

    print_policy(grid, policy)
    print_values(grid, values)


# -----------------------------
# Example 2: dynamic 5x5 grid
# -----------------------------

def run_5x5_example() -> None:
    print("\n" + "=" * 60)
    print("Example 2: Dynamic 5x5 gridworld")
    print("=" * 60)

    grid = make_gridworld(
        rows=5,
        cols=5,
        start=(4, 0),
        terminal_rewards={
            (0, 4): 1.0,
            (1, 4): -1.0,
        },
        blocked_states={
            (1, 1),
            (2, 1),
        },
        windy_probs={
            ((2, 3), "U"): {
                (1, 3): 0.5,
                (2, 4): 0.5,
            }
        },
    )

    transition_probs, rewards = build_transition_and_reward_tables(grid)

    # Option 1: random policy
    policy = make_random_policy(grid)

    # Option 2: use this instead if you want a simple deterministic policy
    #  = make_greedy_right_up_policy(grid)

    values = initialize_values(grid)

    values = evaluate_policy(
        grid=grid,
        policy=policy,
        rewards=rewards,
        values=values,
        discount_factor=0.9,
    )

    print_policy(grid, policy)
    print_values(grid, values)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    run_original_style_3x4_example()
    run_5x5_example()


if __name__ == "__main__":
    main()
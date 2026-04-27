from random import choice
from typing import Dict, List, Set, Tuple


State = Tuple[int, int]
Action = str
ActionMap = Dict[State, List[Action]]
Policy = Dict[State, Action]
TransitionKey = Tuple[State, Action, State]
TransitionProbs = Dict[TransitionKey, float]
StateRewards = Dict[State, float]
RewardTable = Dict[TransitionKey, float]
ValueTable = Dict[State, float]

ACTION_SPACE: List[Action] = ['U', 'D', 'L', 'R']
ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }
TOLERANCE: float = 1e-4
GAMMA: float = 0.9

def get_action_delta(action: Action) -> Tuple[int, int]:
    """Retrieve the (row, col) delta for a given action."""
    return ACTION_DELTAS[action]

class Gridworld:
    """Small deterministic gridworld environment used for policy evaluation."""

    def __init__(self, rows: int, cols: int, start: State):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.i, self.j = start
        self.rewards: StateRewards = {}
        self.actions: ActionMap = {}

    def set(self, rewards: StateRewards, actions: ActionMap) -> None:
        """Configure the gridworld with rewards and possible actions."""
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s: State) -> None:
        """Set the current state of the agent."""
        self.i, self.j = s

    def current_state(self) -> State:
        """Return the current state of the agent."""
        return self.i, self.j

    def is_terminal(self, s: State) -> bool:
        """Check if a state is terminal (i.e., has no possible actions)."""
        return s not in self.actions

    def reset(self) -> State:
        """Reset the agent to the starting position."""
        self.i, self.j = self.start
        return self.i, self.j

    def get_next_state(self, s: State, a: Action) -> State:
        """Determine the next state given a state and an action."""
        i, j = s
        if a in self.actions.get((i, j), []):
            delta_i, delta_j = get_action_delta(a)
            i += delta_i
            j += delta_j
        return i, j

    def get_all_states(self) -> Set[State]:
        """Retrieve all possible states in the gridworld."""
        return set(self.actions.keys()) | set(self.rewards.keys())

    def get_action_space(self, state: State) -> List[Action]:
        """Retrieve all available actions for a non-terminal state."""
        return self.actions[state]

def print_values(V: ValueTable, g: Gridworld) -> None:
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

def print_policy(p: Policy, g: Gridworld) -> None:
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

def standard_grid() -> Gridworld:
    g = Gridworld(rows=3, cols=4, start=(2, 0))
    rewards = {(0, 3): 1.0, (1, 3): -1.0}
    actions = {
        (0, 0): ['D', 'R'],
        (0, 1): ['L', 'R'],
        (0, 2): ['L', 'D', 'R'],
        (1, 0): ['U', 'D'],
        (1, 2): ['U', 'D', 'R'],
        (2, 0): ['U', 'R'],
        (2, 1): ['L', 'R'],
        (2, 2): ['L', 'R', 'U'],
        (2, 3): ['L', 'U'],
    }
    g.set(rewards=rewards, actions=actions)
    return g

def get_transition_probs_and_rewards(g: Gridworld) -> Tuple[TransitionProbs, RewardTable]:
    transition_probs: TransitionProbs = {}
    rewards: RewardTable = {}

    for state in g.actions:
        for action in g.get_action_space(state):
            next_state = g.get_next_state(state, action)
            transition_probs[(state, action, next_state)] = 1.0
            reward = g.rewards.get(next_state)
            if reward is not None:
                rewards[(state, action, next_state)] = reward

    return transition_probs, rewards

def evaluate_deterministic_policy(
    g: Gridworld,
    policy: Policy,
    rewards: RewardTable,
    init_values: ValueTable | None = None,
) -> ValueTable:
    values = init_values.copy() if init_values is not None else {
        state: 0.0 for state in g.get_all_states()
    }

    while True:
        biggest_change = 0.0

        for state in g.actions:
            old_value = values[state]
            new_value = 0.0

            action = policy[state]
            next_state = g.get_next_state(state, action)
            reward = rewards.get((state, action, next_state), 0.0)
            new_value += reward + GAMMA * values[next_state]

            values[state] = new_value
            biggest_change = max(biggest_change, abs(old_value - new_value))

        if biggest_change < TOLERANCE:
            break

    return values

def initialize_random_policy(g: Gridworld) -> Policy:
    return {
        state: choice(g.get_action_space(state))
        for state in g.actions
    }

def iterate_policy(
        g: Gridworld,
        p: Policy,
        rewards: RewardTable,
        init_values: ValueTable | None = None,
) -> Tuple[Policy,ValueTable]:
    if init_values is None:
        init_values = {s: 0.0 for s in g.get_all_states()}

    while True:
        v = evaluate_deterministic_policy(
            g=g,
            policy=p,
            rewards=rewards,
            init_values=init_values,
        )

        policy_converged = True
        for state in g.actions:
            old_action = p[state]
            best_action = old_action
            best_value = float('-inf')

            for action in g.get_action_space(state):
                next_state = g.get_next_state(state, action)
                reward = rewards.get((state, action, next_state), 0.0)
                candidate_value = reward + GAMMA * v[next_state]

                if candidate_value > best_value:
                    best_value = candidate_value
                    best_action = action

            p[state] = best_action
            if best_action != old_action:
                policy_converged = False

        if policy_converged:
            break

    return p, v

def main() -> None:
    g = standard_grid()
    _, rewards = get_transition_probs_and_rewards(g)

    # Initialize policy and values
    v = {s: 0.0 for s in g.get_all_states()}
    p_initial = initialize_random_policy(g)
    print('Initial Values:')
    print_values(v, g)
    print('Initial Policy:')
    print_policy(p_initial, g)

    # Iterate policy evaluation and improvement until convergence
    p_final, v_final = iterate_policy(g, p_initial, rewards, v)
    print('Final Values:')
    print_values(v_final, g)
    print('Final Policy:')
    print_policy(p_final, g)

    return None

if __name__ == '__main__':
    main()


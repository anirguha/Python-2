from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, List, Tuple


_POLICY_ITERATION_PATH = Path(__file__).with_name(
    "Windy Gridworld - Policy Iteration (Probabilistic).py"
)
_POLICY_ITERATION_SPEC = spec_from_file_location(
    "windy_gridworld_policy_iteration_probabilistic",
    _POLICY_ITERATION_PATH,
)

if _POLICY_ITERATION_SPEC is None or _POLICY_ITERATION_SPEC.loader is None:
    raise ImportError(
        f"Could not load policy-iteration module from {_POLICY_ITERATION_PATH}."
    )

_policy_iteration_module = module_from_spec(_POLICY_ITERATION_SPEC)
_POLICY_ITERATION_SPEC.loader.exec_module(_policy_iteration_module)

PROBS = _policy_iteration_module.PROBS
ACTIONS = _policy_iteration_module.ACTIONS
print_policy = _policy_iteration_module.print_policy
print_values = _policy_iteration_module.print_values
windy_grid_penalized = _policy_iteration_module.windy_grid_penalized
build_transition_reward_table = _policy_iteration_module.build_transition_reward_table
initialize_values = _policy_iteration_module.initialize_values

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

def value_iteration(
    transition_model: TransitionModel,
    reward_table: RewardTable,
    states: List[State],
    action_map: ActionMap,
    gamma: float = GAMMA,
    tolerance: float = TOLERANCE
) -> Tuple[ValueTable, Policy]:
    """
    Perform value iteration to compute the optimal value function and policy.

    Args:
        transition_model (TransitionModel): The transition probabilities for each state-action pair.
        reward_table (RewardTable): The rewards for each state-action-next_state transition.
        states (List[State]): The list of all states in the environment.
        action_map (ActionMap): Available actions for each state.
        gamma (float): The discount factor.
        tolerance (float): The convergence threshold.

    Returns:
        Tuple[ValueTable, Policy]: A tuple containing the optimal value function and policy.
    """
    # Initialize value function
    V: ValueTable = {state: 0.0 for state in states}

    while True:
        delta = 0
        for state in states:
            v = V[state]
            action_values = []
            for action in action_map.get(state, ()):
                action_value = 0
                for next_state, prob in transition_model.get((state, action), {}).items():
                    reward = reward_table.get((state, action, next_state), STEP_COST)
                    action_value += prob * (reward + gamma * V[next_state])
                action_values.append(action_value)
            V[state] = max(action_values) if action_values else 0.0
            delta = max(delta, abs(v - V[state]))

        if delta < tolerance:
            break

    # Derive policy from value function
    policy: Policy = {}
    for state in states:
        best_action = None
        best_value = float('-inf')
        for action in action_map.get(state, ()):
            action_value = 0
            for next_state, prob in transition_model.get((state, action), {}).items():
                reward = reward_table.get((state, action, next_state), STEP_COST)
                action_value += prob * (reward + gamma * V[next_state])
            if action_value > best_value:
                best_value = action_value
                best_action = action
        if best_action is not None:
            policy[state] = best_action

    return V, policy

def main():
    """
    Executes the main workflow for solving a grid environment problem with value iteration.

    This function initializes a grid environment, constructs the transition and reward tables,
    and performs value iteration to compute the optimal state-value function and policy.
    Finally, it outputs the computed values and policy to the console.

    Raises:
        Exception: If an error occurs during grid initialization, table construction, or
        value iteration computation.
    """
    g = windy_grid_penalized()
    _, reward_table = build_transition_reward_table(g)
    states = list(g.get_all_states())
    V, policy = value_iteration(g.probs, reward_table, states, g.actions)
    print_values(V, g)
    print_policy(policy, g)

if __name__ == "__main__":
    main()

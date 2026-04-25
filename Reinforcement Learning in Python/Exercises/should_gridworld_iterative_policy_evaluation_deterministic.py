from functools import lru_cache
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "Gridworld - Iterative Policy Evaluation (Deterministic).py"
)


@lru_cache(maxsize=1)
def load_gridworld_module():
    spec = importlib.util.spec_from_file_location("gridworld_iterative_policy_eval", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def bellman_update_for_state(module, state, values):
    grid = module.Gridworld()
    v2 = 0.0
    for action in grid.get_action_space(state):
        for next_state in grid.get_all_states():
            action_prob = 1 if module.policy.get(state) == action else 0
            transition_prob = module.transition_probs.get((state, action, next_state), 0)
            reward = module.rewards.get((state, action, next_state), 0)
            v2 += action_prob * transition_prob * (
                reward + grid.discount_factor * values[next_state]
            )
    return v2


def should_return_configured_terminal_and_step_rewards():
    module = load_gridworld_module()
    grid = module.Gridworld()

    assert grid.get_state_reward((0, 3)) == 1
    assert grid.get_state_reward((1, 3)) == -1
    assert grid.get_state_reward((2, 0)) == -1


def should_return_zero_for_unknown_state_reward():
    module = load_gridworld_module()
    grid = module.Gridworld()

    assert grid.get_state_reward((9, 9)) == 0


def should_move_to_expected_next_state_for_valid_action():
    module = load_gridworld_module()
    grid = module.Gridworld()

    assert grid.get_next_state((2, 0), "U") == (1, 0)
    assert grid.get_next_state((0, 2), "R") == (0, 3)


def should_keep_state_unchanged_for_invalid_action():
    module = load_gridworld_module()
    grid = module.Gridworld()

    assert grid.get_next_state((0, 0), "L") == (0, 0)


def should_identify_terminal_and_non_terminal_states():
    module = load_gridworld_module()
    grid = module.Gridworld()

    assert grid.is_terminal_state((0, 3)) is True
    assert grid.is_terminal_state((1, 3)) is True
    assert grid.is_terminal_state((2, 0)) is False
    assert grid.end_episode((1, 3)) is True
    assert grid.end_episode((0, 0)) is False


def should_create_exactly_one_deterministic_transition_per_state_action_pair():
    module = load_gridworld_module()
    grid = module.Gridworld()

    for state in grid.get_all_states():
        if grid.is_terminal_state(state):
            continue
        for action in grid.get_action_space(state):
            probs = [
                prob
                for (s, a, _), prob in module.transition_probs.items()
                if s == state and a == action
            ]
            assert len(probs) == 1
            assert probs[0] == 1


def should_compute_expected_bellman_update_for_policy_selected_action():
    module = load_gridworld_module()
    grid = module.Gridworld()

    values = {state: 0.0 for state in grid.get_all_states()}
    for state in grid.get_all_states():
        if grid.is_terminal_state(state):
            values[state] = float(grid.get_state_reward(state))

    assert bellman_update_for_state(module, (0, 2), values) == 2.0
    assert bellman_update_for_state(module, (2, 0), values) == -1.0


if __name__ == "__main__":
    should_return_configured_terminal_and_step_rewards()
    should_return_zero_for_unknown_state_reward()
    should_move_to_expected_next_state_for_valid_action()
    should_keep_state_unchanged_for_invalid_action()
    should_identify_terminal_and_non_terminal_states()
    should_create_exactly_one_deterministic_transition_per_state_action_pair()
    should_compute_expected_bellman_update_for_policy_selected_action()


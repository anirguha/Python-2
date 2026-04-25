from functools import lru_cache
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "Windy Gridworld - Iterative Policy Eval;uation (Probabilistic).py"
)


@lru_cache(maxsize=1)
def load_windy_module():
    spec = importlib.util.spec_from_file_location("windy_gridworld_iterative_policy_eval", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assert_raises(expected_exception, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__} to be raised.")


def should_require_configuration_before_accessing_actions_or_states():
    module = load_windy_module()
    grid = module.WindyGridworld(3, 4, (2, 0))

    assert_raises(RuntimeError, grid.get_state_action_space)
    assert_raises(RuntimeError, grid.get_all_states)
    assert_raises(RuntimeError, grid.get_action_space)


def should_use_terminal_states_consistently():
    module = load_windy_module()
    grid = module.windy_gridworld(3, 4, (2, 0))

    assert grid.is_terminal_state((0, 3)) is True
    assert grid.is_terminal_state((1, 3)) is True
    assert grid.is_terminal_state((2, 0)) is False

    grid.set_state((1, 3))
    assert grid.end_episode() is True


def should_reject_invalid_moves_and_terminal_moves():
    module = load_windy_module()
    grid = module.windy_gridworld(3, 4, (2, 0))

    assert_raises(ValueError, grid.move, "L")

    grid.set_state((0, 3))
    assert_raises(ValueError, grid.move, "L")


def should_return_safe_action_counts_and_all_states():
    module = load_windy_module()
    grid = module.windy_gridworld(3, 4, (2, 0))

    assert grid.get_num_actions((0, 3)) == 0
    assert grid.get_num_actions((1, 1)) == 0

    all_states = grid.get_all_states()
    assert isinstance(all_states, set)
    assert (0, 3) in all_states
    assert (1, 3) in all_states
    assert (2, 0) in all_states


def should_return_windy_gridworld_in_repr():
    module = load_windy_module()
    grid = module.windy_gridworld(3, 4, (2, 0))

    assert repr(grid).startswith("WindyGridworld(")


if __name__ == "__main__":
    should_require_configuration_before_accessing_actions_or_states()
    should_use_terminal_states_consistently()
    should_reject_invalid_moves_and_terminal_moves()
    should_return_safe_action_counts_and_all_states()
    should_return_windy_gridworld_in_repr()

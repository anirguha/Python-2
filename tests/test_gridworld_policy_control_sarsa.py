import unittest
from io import StringIO
from unittest.mock import patch

from Reinforcement_Learning_in_Python.Exercises import (
    gridworld_policy_control_sarsa as sarsa_module,
)
from Reinforcement_Learning_in_Python.Exercises.gridworld_standard_windy import (
    WindyGridworld,
    negative_reward_gridworld,
)


class TestGridworldPolicyControlSarsa(unittest.TestCase):
    def setUp(self):
        self.grid: WindyGridworld = negative_reward_gridworld(
            rows=3,
            cols=4,
            start=(2, 0),
            terminal_states=((0, 3), (1, 3)),
            step_cost=-0.5,
        )

    def test_epsilon_greedy_raises_when_state_has_no_actions(self):
        with self.assertRaises(ValueError):
            sarsa_module.epsilon_greedy({(0, 3): {}}, (0, 3), epsilon=0.1)

    def test_sarsa_initializes_from_grid_api_and_handles_terminal_transition(self):
        reward_per_episode, q, update_counts = sarsa_module.sarsa(
            self.grid,
            epsilon=0.0,
            alpha=0.5,
            gamma=0.9,
            num_episodes=1,
        )

        self.assertEqual(len(reward_per_episode), 1)
        self.assertEqual(set(q.keys()), set(self.grid.get_action_space().keys()))
        self.assertGreater(sum(update_counts.values()), 0)

    def test_plot_sarsa_results_skips_when_matplotlib_is_unavailable(self):
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import), patch(
            "sys.stdout",
            new_callable=StringIO,
        ) as stdout:
            sarsa_module.plot_sarsa_results([1.0, 0.5])

        self.assertIn("matplotlib is not installed", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

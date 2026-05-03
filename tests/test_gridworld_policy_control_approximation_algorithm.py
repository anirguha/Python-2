import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from Reinforcement_Learning_in_Python.Exercises import (
    gridworld_policy_control_approximation_algorithm as approx_module,
)
from Reinforcement_Learning_in_Python.Exercises.gridworld_standard_windy import (
    WindyGridworld,
    negative_reward_gridworld,
)


class TestGridworldPolicyControlApproximationAlgorithm(unittest.TestCase):
    def setUp(self):
        self.grid: WindyGridworld = negative_reward_gridworld(
            rows=3,
            cols=4,
            start=(2, 0),
            terminal_states=((0, 3), (1, 3)),
            step_cost=-0.5,
        )

    def _make_mock_approx(self, n_actions: int = 4) -> MagicMock:
        mock_approx = MagicMock()
        mock_approx.predict_all_actions.return_value = np.arange(n_actions, dtype=float)
        mock_approx.grad.return_value = np.zeros(100)
        mock_approx.predict.return_value = 0.5
        mock_approx.w = np.zeros(100)
        return mock_approx

    def test_epsilon_greedy_exploits_model_at_zero_epsilon(self):
        mock_approx = self._make_mock_approx(n_actions=4)
        mock_approx.predict_all_actions.return_value = np.array([0.1, 5.0, 0.3, 0.4])

        with patch.object(approx_module, "random", return_value=0.5):
            action = approx_module.epsilon_greedy(self.grid, mock_approx, (2, 0), eps=0.0)

        action_map = self.grid.get_action_space()
        best_idx = int(np.argmax([0.1, 5.0, 0.3, 0.4]))
        self.assertEqual(action, action_map[(2, 0)][best_idx])

    def test_epsilon_greedy_explores_randomly_at_full_epsilon(self):
        mock_approx = self._make_mock_approx()

        with patch.object(approx_module, "random", return_value=0.99), patch.object(
            approx_module, "choice", return_value="U"
        ) as mock_choice:
            action = approx_module.epsilon_greedy(self.grid, mock_approx, (2, 0), eps=1.0)

        mock_choice.assert_called_once()
        self.assertEqual(action, "U")

    def test_run_q_learning_approximation_returns_correct_structure(self):
        mock_approx = self._make_mock_approx()

        with patch.object(approx_module, "ValueFunctionApproximator", return_value=mock_approx):
            rewards, state_counts, returned_approx = approx_module.run_q_learning_approximation(
                self.grid,
                num_samples=10,
                epsilon=0.0,
                lr=0.1,
                gamma=0.9,
                num_iterations=2,
            )

        self.assertEqual(len(rewards), 2)
        self.assertIsInstance(state_counts, dict)
        self.assertGreater(len(state_counts), 0)
        self.assertIs(returned_approx, mock_approx)

    def test_run_q_learning_approximation_visits_states_in_episode(self):
        mock_approx = self._make_mock_approx()
        visited = []

        original_greedy = approx_module.epsilon_greedy

        def tracking_greedy(g, model, s, eps):
            visited.append(s)
            return original_greedy(g, model, s, eps)

        with patch.object(approx_module, "ValueFunctionApproximator", return_value=mock_approx), patch.object(
            approx_module, "epsilon_greedy", side_effect=tracking_greedy
        ):
            approx_module.run_q_learning_approximation(
                self.grid,
                num_samples=10,
                epsilon=0.0,
                lr=0.1,
                gamma=0.9,
                num_iterations=1,
            )

        self.assertIn((2, 0), visited)

    def test_get_predicted_values_returns_values_for_all_states(self):
        mock_approx = MagicMock()
        mock_approx.predict_all_actions.return_value = np.array([1.0, 0.5])

        values = approx_module.get_predicted_values(self.grid, mock_approx)

        all_states = self.grid.get_all_states()
        self.assertEqual(set(values.keys()), set(all_states))

        action_map = self.grid.get_action_space()
        for s, v in values.items():
            if s in action_map:
                self.assertEqual(v, 1.0)
            else:
                self.assertEqual(v, 0.0)

    def test_get_optimal_policy_returns_valid_policy_and_values(self):
        mock_approx = MagicMock()
        mock_approx.predict_all_actions.return_value = np.array([0.5, 1.0, 0.2, 0.8])

        policy, V = approx_module.get_optimal_policy(self.grid, mock_approx)

        action_map = self.grid.get_action_space()
        for s in self.grid.get_all_states():
            self.assertIn(s, V)
            if s in action_map:
                self.assertIn(s, policy)
                self.assertIn(policy[s], action_map[s])

    def test_get_state_sample_counts_returns_dataframe_with_correct_shape(self):
        state_visit_count = {(0, 0): 5, (1, 2): 3, (2, 3): 1}

        df = approx_module.get_state_sample_counts(state_visit_count)

        self.assertEqual(df.shape, (3, 4))
        self.assertEqual(df.iloc[0, 0], 5.0)
        self.assertEqual(df.iloc[1, 2], 3.0)
        self.assertEqual(df.iloc[2, 3], 1.0)

    def test_get_state_sample_counts_unvisited_states_are_zero(self):
        df = approx_module.get_state_sample_counts({(0, 1): 7})

        self.assertEqual(df.iloc[0, 1], 7.0)
        self.assertEqual(df.iloc[2, 2], 0.0)

    def test_plot_reward_per_episode_saves_file_on_agg_backend(self):
        with patch.object(approx_module.plt, "get_backend", return_value="agg"), patch.object(
            approx_module.plt, "plot"
        ), patch.object(approx_module.plt, "title"), patch.object(
            approx_module.plt, "xlabel"
        ), patch.object(
            approx_module.plt, "ylabel"
        ), patch.object(
            approx_module.plt, "savefig"
        ) as mock_save, patch.object(
            approx_module.plt, "close"
        ):
            approx_module.plot_reward_per_episode([1.0, 0.5, 0.2])

        mock_save.assert_called_once_with("reward_per_episode.png")

    def test_plot_reward_per_episode_calls_show_on_non_agg_backend(self):
        with patch.object(approx_module.plt, "get_backend", return_value="TkAgg"), patch.object(
            approx_module.plt, "plot"
        ), patch.object(approx_module.plt, "title"), patch.object(
            approx_module.plt, "xlabel"
        ), patch.object(
            approx_module.plt, "ylabel"
        ), patch.object(
            approx_module.plt, "show"
        ) as mock_show:
            approx_module.plot_reward_per_episode([1.0])

        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()

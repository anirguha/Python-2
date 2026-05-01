import unittest
from unittest.mock import patch

from Reinforcement_Learning_in_Python.Exercises import gridworld_monte_carlo_epsilon_greedy as mc
from Reinforcement_Learning_in_Python.Exercises.gridworld_standard_windy import (
    WindyGridworld,
    negative_reward_gridworld,
)


class TestGridworldMonteCarloEpsilonGreedy(unittest.TestCase):
    def setUp(self):
        self.grid: WindyGridworld = negative_reward_gridworld(
            rows=3,
            cols=4,
            start=(2, 0),
            terminal_states=((0, 3), (1, 3)),
            step_cost=-0.5,
        )
        self.action_map = self.grid.get_action_space()
        self.deterministic_policy = {
            (2, 0): "U",
            (1, 0): "U",
            (0, 0): "R",
            (0, 1): "R",
            (0, 2): "R",
            (1, 2): "U",
            (2, 1): "L",
            (2, 2): "L",
            (2, 3): "L",
        }

    def test_initialize_values_returns_structure_and_defaults(self):
        q, sample_counts, first_visit_counts, visit_counts, truncated, terminated = (
            mc.initialize_values_returns(self.grid)
        )

        self.assertEqual(truncated, 0)
        self.assertEqual(terminated, 0)

        for state, actions in self.action_map.items():
            self.assertIn(state, q)
            self.assertIn(state, sample_counts)
            self.assertIn(state, first_visit_counts)
            self.assertEqual(first_visit_counts[state], 0)
            for action in actions:
                self.assertIn(action, q[state])
                self.assertEqual(q[state][action], 0.0)
                self.assertEqual(sample_counts[state][action], 0)

        for state in self.grid.get_all_states():
            self.assertIn(state, visit_counts)
            self.assertEqual(visit_counts[state], 0)

    def test_epsilon_greedy_raises_on_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            mc.epsilon_greedy(self.deterministic_policy, (2, 0), self.action_map, eps=-0.1)
        with self.assertRaises(ValueError):
            mc.epsilon_greedy(self.deterministic_policy, (2, 0), self.action_map, eps=1.1)

    def test_epsilon_greedy_returns_policy_action_when_eps_zero(self):
        action = mc.epsilon_greedy(self.deterministic_policy, (2, 0), self.action_map, eps=0.0)
        self.assertEqual(action, "U")

    def test_epsilon_greedy_returns_random_action_when_eps_one(self):
        with patch.object(mc, "choice", return_value="R"):
            action = mc.epsilon_greedy(self.deterministic_policy, (2, 0), self.action_map, eps=1.0)
        self.assertEqual(action, "R")

    def test_epsilon_greedy_raises_for_unknown_state_or_no_actions(self):
        with self.assertRaises(ValueError):
            mc.epsilon_greedy(self.deterministic_policy, (9, 9), self.action_map, eps=0.3)

        bad_action_map = dict(self.action_map)
        bad_action_map[(2, 0)] = ()
        with self.assertRaises(ValueError):
            mc.epsilon_greedy(self.deterministic_policy, (2, 0), bad_action_map, eps=0.3)

    def test_get_max_key_value_returns_max_pair_and_handles_ties(self):
        self.assertEqual(mc.get_max_key_value({"U": 1.5}), ("U", 1.5))
        with patch.object(mc, "choice", return_value="R"):
            self.assertEqual(mc.get_max_key_value({"U": 1.0, "R": 1.0}), ("R", 1.0))
        with self.assertRaises(ValueError):
            mc.get_max_key_value({})

    def test_create_random_policy_yields_valid_state_action_pairs(self):
        with patch.object(mc, "shuffle", side_effect=lambda x: None), patch.object(
            mc, "choice", side_effect=lambda actions: actions[0]
        ):
            pairs = list(mc.create_random_policy(self.grid))

        self.assertEqual(len(pairs), len(self.action_map))
        for state, action in pairs:
            self.assertIn(state, self.action_map)
            self.assertIn(action, self.action_map[state])

    def test_play_episode_counts_terminated_when_terminal_reached(self):
        visit_counts = {s: 0 for s in self.grid.get_all_states()}
        with patch.object(mc, "epsilon_greedy", side_effect=lambda policy, state, action_map: policy[state]):
            states, actions, rewards, truncated, terminated = mc.play_episode(
                self.grid,
                self.deterministic_policy,
                visit_counts,
                truncated_count=0,
                terminated_count=0,
                start_state=(2, 0),
                max_steps=20,
            )

        self.assertGreater(len(states), 1)
        self.assertEqual(len(actions), len(rewards))
        self.assertEqual(truncated, 0)
        self.assertEqual(terminated, 1)
        self.assertTrue(self.grid.is_terminal(states[-1]))

    def test_play_episode_counts_truncated_when_max_steps_reached(self):
        looping_policy = dict(self.deterministic_policy)
        looping_policy[(2, 0)] = "R"
        looping_policy[(2, 1)] = "L"

        visit_counts = {s: 0 for s in self.grid.get_all_states()}
        with patch.object(mc, "epsilon_greedy", side_effect=lambda policy, state, action_map: policy[state]):
            _, _, _, truncated, terminated = mc.play_episode(
                self.grid,
                looping_policy,
                visit_counts,
                truncated_count=0,
                terminated_count=0,
                start_state=(2, 0),
                max_steps=4,
            )

        self.assertEqual(truncated, 1)
        self.assertEqual(terminated, 0)

    def test_validate_monte_carlo_control_inputs(self):
        mc._validate_monte_carlo_control_inputs(gamma=0.9, num_runs=1, max_steps=1)
        with self.assertRaises(ValueError):
            mc._validate_monte_carlo_control_inputs(gamma=1.1, num_runs=1, max_steps=1)
        with self.assertRaises(ValueError):
            mc._validate_monte_carlo_control_inputs(gamma=0.9, num_runs=0, max_steps=1)
        with self.assertRaises(ValueError):
            mc._validate_monte_carlo_control_inputs(gamma=0.9, num_runs=1, max_steps=0)

    def test_get_first_visit_indices(self):
        indices = mc._get_first_visit_indices([
            ((2, 0), "U"),
            ((1, 0), "U"),
            ((2, 0), "U"),
            ((0, 0), "R"),
        ])
        self.assertEqual(indices[((2, 0), "U")], 0)
        self.assertEqual(indices[((1, 0), "U")], 1)
        self.assertEqual(indices[((0, 0), "R")], 3)

    def test_update_first_visit_estimate_updates_q_policy_and_counts(self):
        policy = {(2, 0): "R"}
        q = {(2, 0): {"U": 0.0, "R": 1.0}}
        sample_counts = {(2, 0): {"U": 0, "R": 0}}
        first_visit_counts = {(2, 0): 0}

        change = mc._update_first_visit_estimate(
            state=(2, 0),
            action="U",
            return_value=2.0,
            policy=policy,
            Q=q,
            sample_counts=sample_counts,
            state_sample_first_visit_counts=first_visit_counts,
        )

        self.assertAlmostEqual(change, 2.0)
        self.assertEqual(sample_counts[(2, 0)]["U"], 1)
        self.assertEqual(first_visit_counts[(2, 0)], 1)
        self.assertEqual(q[(2, 0)]["U"], 2.0)
        self.assertEqual(policy[(2, 0)], "U")

    def test_get_state_values_from_q(self):
        q = {(0, 0): {"U": -1.0, "R": 0.5}, (1, 0): {}, (2, 0): {"U": -0.2}}
        values = mc.get_state_values_from_q(q)
        self.assertEqual(values[(0, 0)], 0.5)
        self.assertEqual(values[(1, 0)], 0.0)
        self.assertEqual(values[(2, 0)], -0.2)

    def test_monte_carlo_control_updates_and_propagates_episode_counts(self):
        policy = {(2, 0): "U"}
        q = {(2, 0): {"U": 0.0}}
        sample_counts = {(2, 0): {"U": 0}}
        first_visit_counts = {(2, 0): 0}
        visit_counts = {(2, 0): 0}

        def fake_play_episode(_g, _policy, _visit_counts, truncated_count, terminated_count, **_kwargs):
            return [(2, 0), (1, 0)], ["U"], [1.0], truncated_count, terminated_count + 1

        with patch.object(mc, "play_episode", side_effect=fake_play_episode):
            changes, truncated, terminated = mc.monte_carlo_control_eg(
                self.grid,
                policy,
                q,
                sample_counts,
                first_visit_counts,
                truncated_count=0,
                terminated_count=0,
                state_visit_counts=visit_counts,
                gamma=0.9,
                num_runs=3,
                max_steps=5,
            )

        self.assertEqual(len(changes), 3)
        self.assertEqual(truncated, 0)
        self.assertEqual(terminated, 3)
        self.assertEqual(sample_counts[(2, 0)]["U"], 3)
        self.assertEqual(first_visit_counts[(2, 0)], 3)
        self.assertAlmostEqual(q[(2, 0)]["U"], 1.0)


if __name__ == "__main__":
    unittest.main()

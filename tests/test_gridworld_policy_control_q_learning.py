import unittest
from unittest.mock import patch

from Reinforcement_Learning_in_Python.Exercises import (
    gridworld_policy_control_q_learning_algorithm as q_learning_module,
)
from Reinforcement_Learning_in_Python.Exercises.gridworld_standard_windy import (
    WindyGridworld,
    negative_reward_gridworld,
)


class TestGridworldPolicyControlQLearning(unittest.TestCase):
    def setUp(self):
        self.grid: WindyGridworld = negative_reward_gridworld(
            rows=3,
            cols=4,
            start=(2, 0),
            terminal_states=((0, 3), (1, 3)),
            step_cost=-0.5,
        )
        self.policy = {
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

    def test_epsilon_greedy_action_selection_raises_when_state_has_no_actions(self):
        with self.assertRaises(ValueError):
            q_learning_module.epsilon_greedy_action_selection(
                {(0, 3): {}},
                (0, 3),
                epsilon=0.1,
            )

    def test_q_learning_uses_state_for_action_selection(self):
        observed_states = []

        def fake_select(Q, state, epsilon):
            observed_states.append(state)
            return self.policy[state]

        with patch.object(
            q_learning_module,
            "epsilon_greedy_action_selection",
            side_effect=fake_select,
        ):
            reward_per_episode, q, update_counts = q_learning_module.q_learning(
                self.grid,
                epsilon=0.1,
                alpha=0.5,
                gamma=0.9,
                num_episodes=1,
            )

        self.assertEqual(observed_states, [(2, 0), (1, 0), (0, 0), (0, 1), (0, 2)])
        self.assertEqual(len(reward_per_episode), 1)
        self.assertEqual(set(q.keys()), set(self.grid.get_action_space().keys()))
        self.assertGreater(sum(update_counts.values()), 0)


if __name__ == "__main__":
    unittest.main()

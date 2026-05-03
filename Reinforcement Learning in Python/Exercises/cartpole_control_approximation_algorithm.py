from __future__ import annotations

import sys
from random import random
from typing import Optional, Tuple, List, Dict, Any, Literal, Callable

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from tqdm.auto import tqdm

# =======================
# Type Aliases
# =======================
State = np.ndarray
Action = int # 0 or 1 for Cartpole environment
Reward = float
Samples = List[np.ndarray]
ActionSelector = Literal["sampling", "training"]

# =======================
# Hyperparameters
# =======================
EPSILON: float = 0.1
MODEL: Optional["ValueFunctionApproximator"] = None
GAMMA: float = 0.99
LR: float = 0.01


# =======================
# Utility Functions
# =======================

def epsilon_greedy(state: State,
                   eps: float = EPSILON) -> int | Any:
    """
    Selects an action using the epsilon-greedy strategy.

    This function determines whether to exploit (choose the action with the highest
    predicted value) or explore (select a random action) based on the given epsilon
    value. A smaller epsilon favors exploitation, while a larger epsilon encourages
    exploration. The function relies on a pre-initialized model for predictions and
    ensures that actions are sampled appropriately from the environment.

    Args:
        state (State): The current state of the environment.
        eps (float): The probability of choosing a random action rather than exploiting
            the predicted best action.

    Returns:
        int | Any: The selected action, either the index of the action with the highest
        predicted value or a randomly sampled action.

    Raises:
        ValueError: If MODEL is not initialized before calling this function.
    """
    if MODEL is None:
        raise ValueError("MODEL must be initialized before calling epsilon_greedy().")

    if random() < (1 - eps):
        values = MODEL.predict_all_actions(state)
        return np.argmax(values)
    else:
        return MODEL.env.action_space.sample()

# =======================
# Shared Helper: Run Episode with Optional Sample Collection
# =======================

def run_episode_sampling_training(
    env: gymnasium.Env,
    action_selector: Optional[Callable[[State], Action]] = None,
    sampling_training: Optional[ActionSelector] = None,
) -> Tuple[float, Samples, Optional[float]]:
    """
    Executes a single episode in a reinforcement learning environment, where the agent
    either collects samples for analysis or trains its model depending on the specified
    mode of operation.

    This function allows for flexible operation by enabling two optional modes:
    - Sampling mode: Records state-action pairs for further analysis or offline processing.
    - Training mode: Trains a model in an online manner using gradients to update weights.

    If no mode is specified, the agent interacts with the environment without making
    modifications or collecting additional data beyond tracking the reward. An optional
    action selection function can be provided for more controlled decision-making,
    defaulting to random actions if not specified.

    Args:
        env: A Gymnasium environment in which the agent will interact.
        action_selector: Optional callable that maps a state to an action. If not specified,
            a random action is chosen based on the environment's action space.
        sampling_training: Specifies the mode of operation for the agent. Must be one of
            None, "sampling", or "training". If set to "sampling", state-action pairs
            are collected. If set to "training", model training is performed during the
            episode.

    Returns:
        A tuple containing:
        - The total reward accumulated during the episode (float).
        - A list of state-action pairs collected (Samples). If in "training" mode or no
          mode is specified, an empty list is returned.
        - The final target value computed for the episode (Optional[float]). If not in
          "training" mode, this value is None.
    """
    s, info = env.reset()
    done: bool = False
    truncated: bool = False
    episode_reward: float = 0.0
    samples: Samples = []
    target: Optional[float] = None

    if sampling_training not in (None, "sampling", "training"):
        raise ValueError("sampling_training must be one of None, 'sampling', or 'training'.")

    if action_selector is None:
        action_selector = lambda _: env.action_space.sample()

    while not (done or truncated):
        a = action_selector(s)

        # Collect state-action pair if requested
        if sampling_training == "sampling":
            sa = np.concatenate((s, [a]))
            samples.append(sa)

        s_next, r, done, truncated, info = env.step(a)
        episode_reward += float(r)

        if sampling_training == "training":
            if MODEL is None:
                raise ValueError("MODEL must be initialized before training.")
            if done or truncated:
                target = float(r)
            else:
                values = MODEL.predict_all_actions(s_next)
                target = float(r) + GAMMA * np.max(values)

            g = MODEL.grad(s, a)

            err = target - MODEL.predict(s, a)
            MODEL.w += LR * err * g

        s = s_next

    return episode_reward, samples, target

# =======================
# Sample Collection
# =======================

def collect_samples(env: gymnasium.Env, max_samples: int) -> Samples:
    """
    Collects a specified number of samples by executing episodes in the provided environment.

    This function runs episodes in the given environment using a random action selection
    policy. During each episode, state-action pairs are collected. The samples from all
    episodes are aggregated and returned.

    Args:
        env (gymnasium.Env): The environment in which to execute the episodes. Must be
            compatible with the Gymnasium API.
        max_samples (int): The number of samples to be collected from episodes.

    Returns:
        Samples: A collection of state-action pairs aggregated from all executed episodes.
    """
    all_samples: Samples = []
    random_action_selector = lambda _: env.action_space.sample()

    pbar = tqdm(total=max_samples, desc="Collecting Samples", leave=True, file=sys.stdout)

    while len(all_samples) < max_samples:
        _, samples, _ = run_episode_sampling_training(
            env,
            action_selector=random_action_selector,
            sampling_training="sampling"
        )

        all_samples.extend(samples)
        pbar.update(len(samples))

    pbar.close()
    return all_samples[:max_samples]


# =======================
# Value Function Approximator
# =======================
class ValueFunctionApproximator:

    def __init__(self, env: gymnasium.Env) -> None:
        """
        Initializes the object and sets up the environment and feature transformation mechanism for
        performing later operations.

        Args:
            env (gymnasium.Env): The environment to be used. It provides the interface
                for agent-environment interaction including observation space, action space,
                and environment dynamics.

        Attributes:
            env (gymnasium.Env): The environment object used to configure and explore
                observations and actions.
            sampler (RBFSampler): Random Fourier Features sampler used for approximating the
                radial basis function kernel. Configured with gamma and the number of components.
            w (np.ndarray): The weight vector initialized to zeros, with dimensions matching
                the number of components in the feature transformation.
        """
        self.env = env
        samples = collect_samples(env, max_samples=10_000)  # Collects state-action pairs for feature fitting
        self.sampler = RBFSampler(gamma=0.5, n_components=100)
        self.sampler.fit(samples)
        dims = self.sampler.n_components
        self.w = np.zeros(dims)

    def predict(self, state: State, a: Action) -> float:
        """
        Predicts the outcome of a given state and action.

        This method performs a prediction by combining the given state and action, transforming
        the resulting data through the sampler, and applying the weight vector to compute the
        output.

        Args:
            state (State): The current state input to be used for prediction.
            a (Action): The action to be combined with the state for prediction.

        Returns:
            float: The predicted outcome as a result of the transformation and weight
            application.
        """
        sa = np.concatenate((state, [a]))  # Combines state and action into a single array/ feature vector
        x = self.sampler.transform([sa])[0]
        return float(x @ self.w)

    def predict_all_actions(self, state: State) -> np.ndarray:
        """
        Predicts the outcome of all possible actions for a given state.

        This method iterates through all the actions in the action space,
        predicts the outcome for each action based on the provided state,
        and returns an array containing the predictions for all actions.

        Args:
            state (State): The current state for which predictions will be
                made for all available actions.

        Returns:
            np.ndarray: An array of predictions for all possible actions
            in the given state.
        """
        sa_batch: List[float] = [self.predict(state, a) for a in range(self.env.action_space.n)]

        return np.array(sa_batch)

    def grad(self, state: State, a: Action) -> np.ndarray:
        """
        Computes the gradient for a given state and action using the sampler's transformation.

        The function takes a state and an action, combines them, and applies a transformation
        through the sampler. The result is returned as the computed gradient.

        Args:
            state (State): The state of the system.
            a (Action): The action to be taken.

        Returns:
            np.ndarray: The computed gradient as a result of applying the sampler's transformation.
        """
        sa = np.concatenate((state, [a]))
        x = self.sampler.transform([sa])[0]
        return x

# =======================
# Test an agent
# =======================

def evaluate_trained_agent(env: gymnasium.Env,
                           num_episodes: int) -> float:
    """
    Evaluates the performance of a trained agent over a specified number of episodes.
    The function runs the agent in the given environment for the specified number of episodes,
    accumulates the rewards for each episode, and computes the mean reward.

    Args:
        env (gymnasium.Env): The environment in which the trained agent will be evaluated.
        num_episodes (int): The number of episodes over which the agent will be evaluated.

    Returns:
        float: The average reward obtained by the agent over the specified number of episodes.
    """
    reward_per_episode: np.ndarray = np.zeros(num_episodes)

    for it in tqdm(range(num_episodes), desc="Episodes Running", leave=False, file=sys.stdout):
        episode_reward, _, _ = run_episode_sampling_training(
            env,
            action_selector=lambda state: epsilon_greedy(state, eps=0),
        )
        reward_per_episode[it] = episode_reward

    print("Done Evaluating Agent!")
    return float(np.mean(reward_per_episode))
# =======================
# Watch an agent
# =======================

def watch_agent(env: gymnasium.Env) -> None:
    """
    Runs an agent in the given environment, observes its performance, and prints the
    obtained episode reward after completion.

    Args:
        env (gymnasium.Env): The environment in which the agent will perform actions.

    Returns:
        None
    """
    episode_reward, _, _ = run_episode_sampling_training(
        env,
        action_selector=lambda state: epsilon_greedy(state, eps=0),
    )
    print(f"Episode Reward: {episode_reward}")

    return None
# =======================
# Train an agent
# =======================

def train_agent(env: gymnasium.Env, num_episodes: int) -> List[float]:
    """
    Trains a reinforcement learning agent over a specified number of episodes and
    records the reward obtained in each episode.

    The function uses a training sampling strategy to execute and evaluate the
    training process for each episode. The rewards are collected and returned as a
    list of Python floats.

    Args:
        env (gymnasium.Env): The environment in which the agent will interact. It must
            adhere to the `gymnasium.Env` interface.
        num_episodes (int): The total number of episodes to run the training process.

    Returns:
        List[float]: A list of total rewards obtained by the agent in each episode.
    """
    reward_per_episode: List[float] = []
    for _ in tqdm(range(num_episodes), desc="Training Agent", leave=True, file=sys.stdout):
        episode_reward, _, target = run_episode_sampling_training(
            env,
            action_selector=epsilon_greedy,
            sampling_training="training"
        )

        reward_per_episode.append(float(episode_reward))

    return reward_per_episode

# =======================
# Plot Reward per Episode
# =======================
def plot_reward_per_episode(rewards: List[float]):
    """
    Plots the reward per episode during training.

    This function generates a plot that visualizes the rewards obtained in each episode of
    training. It helps in evaluating the overall performance and trend of the rewards
    across episodes.

    Args:
        rewards (List[float]): A list of float values representing the reward obtained in
            each episode during the training process.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode During Training")
    plt.legend()
    plt.show()
    plt.close()

# =======================
# Main Loop
# =======================
def main() -> None:
    """
    Main function to train, evaluate, and render a reinforcement learning agent on the CartPole-v1
    environment. The function initializes both training and rendering environments, trains the
    agent, evaluates its performance, and displays its behavior. Additionally, it plots the
    training rewards over episodes.

    Global Variables:
        MODEL (ValueFunctionApproximator): Approximation model used by the agent.
        EPSILON (float): Exploration rate for the reinforcement learning agent.
        GAMMA (float): Discount factor for future rewards.

    Raises:
        gymnasium.error.Error: If the specified Gym environment fails to initialize or encounters
            unexpected issues during its execution.

    Returns:
        None
    """
    global MODEL
    global EPSILON
    global GAMMA

    # Create training environment without rendering (faster)
    train_env = gymnasium.make("CartPole-v1")

    # Create agent's value function approximator model
    print("Creating agent and collecting samples...")
    MODEL = ValueFunctionApproximator(train_env)
    print("Agent Created and Samples Collected!")
    print()


    # Create rendering environment for watching the agent
    render_env = gymnasium.make("CartPole-v1", render_mode="human")
    watch_agent(render_env)
    render_env.close()

    # Train the agent
    print("Training Agent...")
    reward_per_episode = train_agent(train_env, num_episodes=1500)
    print("Training Complete!")
    plot_reward_per_episode(reward_per_episode)

    # Evaluate the trained agent
    print("Evaluating Trained Agent...")
    mean_reward = evaluate_trained_agent(train_env, num_episodes=100)
    print(f"Mean Reward: {mean_reward}")
    train_env.close()
    print("Evaluation Complete!")

    # Watch the trained agent
    print("Watching Trained Agent...")
    render_env = gymnasium.make("CartPole-v1", render_mode="human")
    watch_agent(render_env)
    render_env.close()
    print("Watching Complete!")
    print("Execution Complete!")
    print("Goodbye!")
    print("Program Terminated Successfully!")
    print("Thank you for using the CartPole Control Approximation Algorithm!")


if __name__ == "__main__":
    main()

"""
CartPole Control with Function Approximation using RBF Features

This module implements a Q-learning agent with value-function approximation
for solving the CartPole-v1 environment from OpenAI Gymnasium. The agent uses
Random Fourier Features (via scikit-learn's RBFSampler) to approximate the
Q-value function and updates it online during training episodes.

Key Parts:
- epsilon_greedy(): Action selection policy balancing exploration/exploitation
- run_episode_sampling_training(): Core episode execution with flexible modes
- collect_samples(): Gathers state-action pairs for feature fitting
- ValueFunctionApproximator: Linear approximator over RBF features
- train_agent(): Q-learning training loop
- evaluate_trained_agent(): Performance assessment

Dependencies:
- gymnasium: Environment simulation
- numpy: Numerical computations
- scikit-learn: RBF feature transformation
- matplotlib: Optional visualization
- tqdm: Optional progress bars
"""

from __future__ import annotations

import sys
from random import random
from pathlib import Path
from typing import Optional, Tuple, List, Any, Literal, Callable

import numpy as np

try:
    import gymnasium
except ModuleNotFoundError:
    gymnasium = None

_plt: Optional[Any] = None
_RBFSampler: Optional[Any] = None

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Any = None, **_kwargs: Any) -> Any:
        """Fallback tqdm implementation when tqdm is unavailable."""
        if iterable is None:
            return _ProgressFallback()
        return iterable

# =======================
# Type Aliases
# =======================
State = np.ndarray
"""Type alias for environment observations (numpy array of floats)."""

Action = int  # 0 or 1 for Cartpole environment
"""Type alias for discrete actions (0=push left, 1=push right)."""

Reward = float
"""Type alias for scalar reward values."""

Samples = List[np.ndarray]
"""Type alias for collections of state-action feature vectors."""

ActionSelector = Literal["sampling", "training"]
"""Type alias for episode modes: collecting samples or training the model."""

EpisodeMode = Optional[ActionSelector]
"""Type alias for optional episode modes (None, 'sampling', or 'training')."""

# =======================
# Hyperparameters
# =======================
EPSILON: float = 0.1
"""Default exploration probability for epsilon-greedy policy."""

MODEL: Optional["ValueFunctionApproximator"] = None
"""Global model instance for backward compatibility with legacy code."""

GAMMA: float = 0.99
"""Discount factor for future rewards in Q-learning updates."""

LR: float = 0.01
"""Learning rate for gradient-based weight updates."""

DEFAULT_MAX_STEPS_PER_EPISODE: int = 500
"""Default maximum steps per episode to prevent infinite runs."""


class _ProgressFallback:
    """Minimal progress-bar stand-in used when `tqdm` is unavailable.

    This class provides a no-op implementation of tqdm's progress bar interface,
    allowing the code to function gracefully even when tqdm is not installed.
    All methods intentionally do nothing.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """Accept progress-bar constructor arguments and intentionally ignore them.

        Args:
            *_args: Positional arguments (ignored).
            **_kwargs: Keyword arguments (ignored).
        """
        pass

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        """Accept progress updates without displaying anything.

        Args:
            *_args: Positional arguments (ignored).
            **_kwargs: Keyword arguments (ignored).
        """
        pass

    def close(self) -> None:
        """Finish the no-op progress bar."""
        pass


def _require_gymnasium() -> Any:
    """Return the imported Gymnasium module or raise a clear dependency error.

    This function ensures Gymnasium is available and provides a helpful error
    message if it is not installed. It's called at program startup to fail fast
    with clear feedback.

    Returns:
        The gymnasium module object.

    Raises:
        ModuleNotFoundError: If gymnasium is not installed.
    """
    if gymnasium is None:
        raise ModuleNotFoundError(
            "gymnasium is required to create CartPole environments. "
            "Install gymnasium to run this script."
        )
    return gymnasium


def _get_pyplot() -> Any:
    """Import and cache `matplotlib.pyplot` lazily.

    Matplotlib is optional; this function imports it only if plotting is
    actually needed. The module is cached to avoid repeated imports.

    Returns:
        The `matplotlib.pyplot` module when Matplotlib is installed; otherwise
        `None`.
    """
    global _plt
    if _plt is None:
        try:
            import matplotlib.pyplot as pyplot
        except ModuleNotFoundError:
            return None
        _plt = pyplot
    return _plt


def _get_rbf_sampler_class() -> Any:
    """Import and cache scikit-learn's `RBFSampler` lazily.

    This function provides lazy importing of the RBFSampler class, which is
    required for value function approximation. Caching avoids repeated imports.

    Returns:
        The RBFSampler class from scikit-learn.

    Raises:
        ModuleNotFoundError: If scikit-learn is not installed.
    """
    global _RBFSampler
    if _RBFSampler is None:
        try:
            from sklearn.kernel_approximation import RBFSampler
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "scikit-learn is required for ValueFunctionApproximator. "
                "Install scikit-learn to run CartPole function approximation."
            ) from exc
        _RBFSampler = RBFSampler
    return _RBFSampler


def _rng_random(rng: Optional[Any]) -> float:
    """Return a random float from an optional NumPy-style RNG or Python's RNG.

    This helper abstracts the choice between a NumPy random generator (which
    has a `.random()` method) and Python's built-in `random()` function.

    Args:
        rng: Optional NumPy random generator with a `.random()` method.
             If None, uses Python's built-in random.random().

    Returns:
        A random float in [0.0, 1.0).
    """
    if rng is None:
        return random()
    return float(rng.random())


def _reset_env(env: Any, seed: Optional[int] = None) -> State:
    """Reset a Gymnasium-style environment and return the observation as floats.

    This function standardizes environment reset across different versions of
    Gymnasium by handling optional seed parameters and ensuring the observation
    is converted to a float array.

    Args:
        env: Gymnasium-compatible environment object.
        seed: Optional random seed for reproducibility.

    Returns:
        Environment observation as a numpy array of floats.
    """
    if seed is None:
        state, _info = env.reset()
    else:
        state, _info = env.reset(seed=seed)
    return np.asarray(state, dtype=float)


# =======================
# Utility Functions
# =======================

def epsilon_greedy(
        state: State,
        eps: float = EPSILON,
        model: Optional["ValueFunctionApproximator"] = None,
        rng: Optional[Any] = None,
) -> int | Any:
    """Choose a CartPole action with an epsilon-greedy policy.

    This function implements the epsilon-greedy exploration strategy: with
    probability (1 - eps), it selects the action with the highest estimated
    Q-value (exploitation); with probability eps, it samples a random action
    (exploration). This balances learning from good policies with discovering
    better ones.

    Args:
        state: Current environment observation (float array).
        eps: Exploration probability in the inclusive range [0, 1].
             0.0 means pure exploitation, 1.0 means pure exploration.
        model: Value-function model to the query for Q-values. If omitted, the
               legacy global `MODEL` is used for backward compatibility.
        rng: Optional random generator with a `.random()` method. If None,
             uses Python's standard random module.

    Returns:
        The selected action (typically 0 or 1 for CartPole).

    Raises:
        ValueError: If `eps` is outside [0, 1] or no model is available.
    """
    if not 0 <= eps <= 1:
        raise ValueError("eps must be between 0 and 1.")

    active_model = model or MODEL
    if active_model is None:
        raise ValueError("MODEL must be initialized before calling epsilon_greedy().")

    # Exploitation: pick action with highest Q-value
    if _rng_random(rng) < (1 - eps):
        values = active_model.predict_all_actions(state)
        return int(np.argmax(values))

    # Exploration: pick random action
    return active_model.env.action_space.sample()


# =======================
# Shared Helper: Run Episode with Optional Sample Collection
# =======================

def run_episode_sampling_training(
        env: Any,
        action_selector: Optional[Callable[[State], Action]] = None,
        sampling_training: EpisodeMode = None,
        model: Optional["ValueFunctionApproximator"] = None,
        gamma: float = GAMMA,
        lr: float = LR,
        seed: Optional[int] = None,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS_PER_EPISODE,
        bootstrap_on_truncation: bool = True,
) -> Tuple[float, Samples, Optional[float]]:
    """Execute a single episode with optional sample collection or online training.

    This is the core simulation loop. Depending on the `sampling_training` mode,
    the function can:
    - Collect state-action pairs for offline feature fitting
    - Train a model online using Q-learning with function approximation
    - Run an episode without special processing

    In training mode, Q-learning updates are applied:
        Q(s, a) ← Q(s, a) + lr * (target - Q(s, a)) * grads(s, a)
    where target = r + gamma * max_a' Q(s', a') if the episode continues,
    or target = r if terminal.

    Args:
        env: Gymnasium-style environment.
        action_selector: Optional callable that maps state → action. If None,
                        random actions are sampled.
        sampling_training: Mode of operation:
            - None: Run the episode without special processing.
            - "sampling": Collect state-action pairs.
            - "training": Apply Q-learning updates during the episode.
        model: Value-function model for training/evaluation. Falls back to
               global MODEL if not provided.
        gamma: Discount factor in [0, 1] for future reward weighting.
        lr: Positive learning rate for gradient-descent updates.
        seed: Optional seed for environment reset.
        max_steps: Optional cap on episode length. If reached, the episode is
                  marked as truncated.
        bootstrap_on_truncation: If False, treat truncation as terminal
                                (target = r). If True, use bootstrap
                                (target = r + gamma * max_a Q(s', a')).

    Returns:
        A tuple of:
        - episode_reward (float): Total reward accumulated in the episode.
        - samples (Samples): State-action pairs (empty if not in sampling mode).
        - target (Optional[float]): Final Q-learning target value (None if not
                                   in training mode).

    Raises:
        ValueError: If parameters are invalid, or a model is required but missing.
    """
    # Validate hyperparameters
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1.")
    if lr <= 0:
        raise ValueError("lr must be positive.")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided.")

    # Initialize episode state
    s = _reset_env(env, seed=seed)
    done: bool = False
    truncated: bool = False
    episode_reward: float = 0.0
    samples: Samples = []
    target: Optional[float] = None
    steps = 0

    # Validate mode parameter
    if sampling_training not in (None, "sampling", "training"):
        raise ValueError("sampling_training must be one of None, 'sampling', or 'training'.")

    # Default action selector samples randomly from action space
    if action_selector is None:
        action_selector = lambda _: env.action_space.sample()

    # Fallback to global model if not provided
    active_model = model or MODEL

    # Main episode loop
    while not (done or truncated):
        # Select action using provided policy
        a = action_selector(s)

        # Collect state-action pair if sampling
        if sampling_training == "sampling":
            sa = np.concatenate((s, [a]))
            samples.append(sa)

        # Take step in environment
        s_next, r, done, truncated, _info = env.step(a)
        s_next = np.asarray(s_next, dtype=float)
        episode_reward += float(r)
        steps += 1
        reached_step_limit = max_steps is not None and steps >= max_steps

        # Q-learning update if in training mode
        if sampling_training == "training":
            if active_model is None:
                raise ValueError("A model must be initialized before training.")

            # Compute Q-learning target
            if done or (truncated and not bootstrap_on_truncation):
                # Terminal state: target is just the immediate reward
                target = float(r)
            else:
                # Non-terminal: bootstrap using next state's Q-values
                values = active_model.predict_all_actions(s_next)
                target = float(r) + gamma * np.max(values)

            # Compute gradient and update weights
            g = active_model.grad(s, a)
            err = target - active_model.predict(s, a)
            active_model.w += lr * err * g

        # Move to next state
        s = s_next

        # Force truncation if step limit reached
        if reached_step_limit:
            truncated = True

    return episode_reward, samples, target


# =======================
# Sample Collection
# =======================

def collect_samples(
        env: Any,
        max_samples: Optional[int] = None,
        *,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        max_steps_per_episode: Optional[int] = DEFAULT_MAX_STEPS_PER_EPISODE,
) -> Samples:
    """Collect random-policy state-action feature samples from the environment.

    This function runs episodes with random action selection, collecting all
    state-action pairs encountered. These samples are used to fit the feature
    transformer (RBFSampler) before training begins.

    Episodes run until at least the requested number of state-action vectors
    has been collected. The returned list is trimmed to the exact requested
    count to avoid exceeding the target.

    Args:
        env: Gymnasium-style environment used for sampling.
        max_samples: Number of samples to collect. Kept for compatibility with
                     existing call sites. To be replaced by `num_samples`.
        num_samples: Keyword-only alias for `max_samples`. Preferred parameter.
        seed: Optional random seed for environment resets and reproducibility.
        max_steps_per_episode: Optional step limit for each sampling episode
                              to prevent excessively long runs.

    Returns:
        A list of state-action feature vectors (concatenated state and action),
        trimmed to exactly `max_samples` or `num_samples` elements.

    Raises:
        ValueError: If no positive sample count is supplied.
        RuntimeError: If an episode produces no samples (environment issue).
    """
    # Resolve parameter conflicts
    sample_count = max_samples if max_samples is not None else num_samples
    if sample_count is None:
        raise ValueError("max_samples or num_samples must be provided.")
    if sample_count <= 0:
        raise ValueError("sample count must be positive.")

    # Seed environment's action space for reproducibility
    if seed is not None and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    all_samples: Samples = []
    random_action_selector = lambda _: env.action_space.sample()

    # Initialize progress bar
    pbar = tqdm(total=sample_count, desc="Collecting Samples", leave=True, file=sys.stdout)
    episode_index = 0

    # Run episodes until enough samples collected
    while len(all_samples) < sample_count:
        _, samples, _ = run_episode_sampling_training(
            env,
            action_selector=random_action_selector,
            sampling_training="sampling",
            seed=None if seed is None else seed + episode_index,
            max_steps=max_steps_per_episode,
        )

        # Validate that episode produced samples
        if not samples:
            raise RuntimeError("Sample collection produced no samples; check the environment.")

        # Update progress bar (cap at remaining count to avoid overfull)
        remaining = sample_count - len(all_samples)
        all_samples.extend(samples)
        pbar.update(min(len(samples), remaining))
        episode_index += 1

    pbar.close()

    # Return exactly the requested number of samples
    return all_samples[:sample_count]


# =======================
# Value Function Approximator
# =======================
class ValueFunctionApproximator:
    """Linear action-value approximator over random Fourier features (RBF kernel).

    This class implements Q(state, action) ≈ w^T * φ(state, action), where:
    - w is a learned weight vector
    - φ is a feature transformation using Random Fourier Features approximating
      an RBF kernel

    The approximator can be trained with Q-learning updates to estimate optimal
    action values for reinforcement learning.

    Attributes:
        env: The Gymnasium environment (provides action space info).
        sampler: The RBFSampler instance that transforms raw features.
        w: The weight vector; shape (n_components,).
    """

    def __init__(
            self,
            env: Any,
            num_samples: int = 10_000,
            n_components: int = 100,
            rbf_gamma: float = 0.5,
            seed: Optional[int] = None,
            samples: Optional[Samples] = None,
    ) -> None:
        """Create the feature sampler and initialize model weights.

        The feature transformer is fit on random state-action samples to capture
        meaningful structure in the environment. The weight vector is initialized
        to zero.

        Args:
            env: Gymnasium-style environment whose action space defines available
                actions and properties.
            num_samples: Number of random-policy samples to collect when gathering
                        training data. These are used to fit the RBFSampler.
            n_components: Number of random Fourier features (dimensions of φ).
                         Higher values provide more expressiveness but increase
                         computational cost.
            rbf_gamma: RBF kernel bandwidth parameter passed to RBFSampler.
                      Controls the width of the Gaussian kernels.
            seed: Optional random seed for reproducibility in sampling and
                  feature generation.
            samples: Optional precomputed state-action samples. If provided,
                    these are used instead of collecting new ones.

        Raises:
            ModuleNotFoundError: If scikit-learn is unavailable.
            ValueError: If configuration values are invalid or no samples are available.
        """
        self.env = env

        sampler_cls = _get_rbf_sampler_class()

        # Validate configuration
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if n_components <= 0:
            raise ValueError("n_components must be positive.")

        # Collect or use provided samples
        if samples is None:
            samples = collect_samples(env, max_samples=num_samples, seed=seed)
        if not samples:
            raise ValueError("At least one sample is required to fit the feature sampler.")

        # Fit the feature transformer on raw samples
        self.sampler = sampler_cls(
            gamma=rbf_gamma,
            n_components=n_components,
            random_state=seed,
        )
        self.sampler.fit(samples)

        # Initialize weights to zero
        dims = self.sampler.n_components
        self.w = np.zeros(dims)

    def predict(self, state: State, a: Action) -> float:
        """Estimate Q(state, action) using the learned approximator.

        The approximation is computed as:
            Q(s, a) ≈ w^T * φ(s, a)
        where φ is the RBF feature transformation.

        Args:
            state: Environment observation (float array).
            a: Discrete action index.

        Returns:
            Estimated action value as a Python float.
        """
        # Combine state and action into a feature vector
        sa = np.concatenate((state, [a]))

        # Transform into RBF feature space
        x = self.sampler.transform([sa])[0]

        # Compute linear approximation: w^T * φ
        return float(x @ self.w)

    def predict_all_actions(self, state: State) -> np.ndarray:
        """Estimate action values for every discrete action in the environment.

        This is useful for action selection (e.g., picking argmax for greedy
        policies) and computing bootstrap targets during training.

        Args:
            state: Environment observation (float array).

        Returns:
            NumPy array where element [i] is the estimated Q(state, action_i).
        """
        # Predict Q-value for each action
        sa_batch: List[float] = [self.predict(state, a) for a in range(self.env.action_space.n)]

        return np.array(sa_batch)

    def grad(self, state: State, a: Action) -> np.ndarray:
        """Return the RBF feature vector for a state-action pair.

        This feature vector is used as the gradient direction for Q-learning
        weight updates. Since we use a linear approximator, the gradient w.r.t.
        the weights is exactly the feature vector.

        Args:
            state: Environment observation (float array).
            a: Discrete action index.

        Returns:
            Feature vector φ(state, action) as a numpy array.
        """
        # Combine state and action
        sa = np.concatenate((state, [a]))

        # Transform into RBF feature space; this is the gradient direction
        x = self.sampler.transform([sa])[0]
        return x


# =======================
# Test an agent
# =======================

def run_episode_with_collection(
        env: Any,
        action_selector: Optional[Callable[[State], Action]] = None,
        collect_sa_pairs: bool = False,
        seed: Optional[int] = None,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS_PER_EPISODE,
) -> Tuple[float, Samples]:
    """Run one episode and optionally collect state-action samples.

    This wrapper provides a simpler interface over `run_episode_sampling_training`
    for backward compatibility with older code that only needs the reward and
    optionally the samples.

    Args:
        env: Gymnasium-style environment.
        action_selector: Optional policy callable mapping state → action.
        collect_sa_pairs: If True, state-action pairs are collected.
        seed: Optional reset seed.
        max_steps: Optional episode step cap.

    Returns:
        A tuple of (episode_reward, samples).
    """
    reward, samples, _ = run_episode_sampling_training(
        env,
        action_selector=action_selector,
        sampling_training="sampling" if collect_sa_pairs else None,
        seed=seed,
        max_steps=max_steps,
    )
    return reward, samples


def run_episode(
        env: Any,
        eps: float = 0.0,
        model: Optional["ValueFunctionApproximator"] = None,
        seed: Optional[int] = None,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS_PER_EPISODE,
) -> float:
    """Run one evaluation episode without sample collection or training.

    This is a simple wrapper for running a greedy or epsilon-greedy episode.
    It does not modify the model or collect samples.

    Args:
        env: Gymnasium-style environment.
        eps: Exploration probability for epsilon-greedy action selection.
             0.0 = pure exploitation (greedy).
        model: Optional model for value estimation. Falls back to global MODEL.
        seed: Optional reset seed.
        max_steps: Optional episode step cap.

    Returns:
        Total reward for the episode as a float.
    """
    active_model = model or MODEL

    # Create an epsilon-greedy action selector
    action_selector = lambda state: epsilon_greedy(
        state,
        eps=eps,
        model=active_model,
    )

    # Run episode without training or sample collection
    reward, _samples = run_episode_with_collection(
        env,
        action_selector=action_selector,
        collect_sa_pairs=False,
        seed=seed,
        max_steps=max_steps,
    )
    return float(reward)


def evaluate_trained_agent(
        env: Any,
        num_episodes: int,
        model: Optional["ValueFunctionApproximator"] = None,
        seed: Optional[int] = None,
) -> float:
    """Evaluate a greedy policy over multiple episodes and return mean reward.

    This function runs the model in pure exploitation mode (eps=0) to assess
    how well it has learned.

    Args:
        env: Gymnasium-style environment.
        num_episodes: Number of evaluation episodes to run.
        model: Optional model. Falls back to global MODEL if not provided.
        seed: Optional base seed; episode index is added for reproducibility
              across runs.

    Returns:
        Mean episode reward across all evaluation episodes.

    Raises:
        ValueError: If num_episodes is not positive.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    # Accumulate rewards for each episode
    reward_per_episode: np.ndarray = np.zeros(num_episodes)

    for it in tqdm(range(num_episodes), desc="Episodes Running", leave=False, file=sys.stdout):
        episode_reward = run_episode(
            env,
            eps=0,  # Pure exploitation
            model=model,
            seed=None if seed is None else seed + it,
        )
        reward_per_episode[it] = episode_reward

    print("Done Evaluating Agent!")
    return float(np.mean(reward_per_episode))


# =======================
# Watch an agent
# =======================

def watch_agent(env: Any, model: Optional["ValueFunctionApproximator"] = None) -> None:
    """Run one greedy episode and print its reward.

    This function is typically used with a rendering environment to visualize
    the agent's behavior.

    Args:
        env: Gymnasium-style environment, typically with render_mode="human".
        model: Optional model. Falls back to global MODEL if not provided.
    """
    episode_reward = run_episode(
        env,
        eps=0,  # Pure exploitation
        model=model,
    )
    print(f"Episode Reward: {episode_reward}")

    return None


# =======================
# Train an agent
# =======================

def train_agent(
        env: Any,
        num_episodes: int,
        model: Optional["ValueFunctionApproximator"] = None,
        epsilon: float = EPSILON,
        gamma: float = GAMMA,
        lr: float = LR,
        seed: Optional[int] = None,
        max_steps_per_episode: Optional[int] = DEFAULT_MAX_STEPS_PER_EPISODE,
) -> List[float]:
    """Train a Q-learning agent with function approximation.

    The agent is trained by running episodes under an epsilon-greedy policy
    and updating the value approximator's weights using Q-learning updates at
    each step.

    Args:
        env: Gymnasium-style environment for interaction.
        num_episodes: Number of training episodes.
        model: Model to update. Falls back to global MODEL if not provided.
        epsilon: Exploration probability for epsilon-greedy during training.
                 Higher values encourage more exploration.
        gamma: Discount factor in [0, 1] for future reward weighting.
        lr: Positive learning rate for gradient-descent updates.
        seed: Optional base seed; episode index is added for reproducibility.
        max_steps_per_episode: Optional cap on episode length.

    Returns:
        List of episode rewards, one per training episode.

    Raises:
        ValueError: If the configuration is invalid or no model is available.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    active_model = model or MODEL
    if active_model is None:
        raise ValueError("A model must be provided or MODEL must be initialized.")

    reward_per_episode: List[float] = []

    # Create a seeded RNG for reproducible exploration
    rng = np.random.default_rng(seed)

    # Training loop: run episodes and update model
    for episode_index in tqdm(range(num_episodes), desc="Training Agent", leave=True, file=sys.stdout):
        episode_reward, _, target = run_episode_sampling_training(
            env,
            # Create action selector with epsilon-greedy policy
            action_selector=lambda state: epsilon_greedy(
                state,
                eps=epsilon,
                model=active_model,
                rng=rng,
            ),
            sampling_training="training",  # Triggers Q-learning updates
            model=active_model,
            gamma=gamma,
            lr=lr,
            seed=None if seed is None else seed + episode_index,
            max_steps=max_steps_per_episode,
        )

        reward_per_episode.append(float(episode_reward))

    return reward_per_episode


# =======================
# Plot Reward per Episode
# =======================
def plot_reward_per_episode(
        rewards: List[float],
        output_path: Optional[str | Path] = None,
) -> None:
    """Plot or save the training reward history.

    This function visualizes learning progress. It adapts to the display
    backend: on interactive backends (Jupyter, interactive Python), it shows
    the plot; on non-interactive backends, it saves it to a file.

    Args:
        rewards: Episode reward history from training.
        output_path: Optional path to save the figure. If omitted and on a
                     non-interactive backend, saves to
                     'cartpole_reward_per_episode.png'.
    """
    pyplot = _get_pyplot()
    if pyplot is None:
        print("matplotlib is not installed; skipping reward plot.")
        return None

    # Create figure and plot rewards
    pyplot.figure(figsize=(12, 6))
    pyplot.plot(rewards, label="Reward per Episode")
    pyplot.xlabel("Episode")
    pyplot.ylabel("Reward")
    pyplot.title("Reward per Episode During Training")
    pyplot.legend()

    # Determine backend and output method
    backend = pyplot.get_backend().lower()
    non_interactive_backends = {"agg", "pdf", "pgf", "ps", "svg", "template", "Cairo"}

    if output_path is not None:
        pyplot.savefig(output_path)
    elif backend in non_interactive_backends or backend.startswith("module://matplotlib_inline"):
        # Non-interactive backend: save it to file
        pyplot.savefig("cartpole_reward_per_episode.png")
    else:
        # Interactive backend: display in notebook/repl
        pyplot.show()

    pyplot.close()

    return None


# =======================
# Main Loop
# =======================
def main() -> None:
    """Train, evaluate, plot, and render a CartPole-v1 agent.

    This is the main entry point. The script:
    1. Creates a CartPole-v1 environment
    2. Builds a value-function approximator (samples an RBF feature space)
    3. Trains the agent via Q-learning for several episodes
    4. Plots the training reward history
    5. Evaluates the greedy policy over multiple episodes
    6. Renders one final greedy episode

    Raises:
        ModuleNotFoundError: If Gymnasium is not installed.
        gymnasium.error.Error: If environment creation/interaction fails.
    """
    gym = _require_gymnasium()
    global MODEL

    # Setup: Create training environment and set seed
    seed = 123
    train_env = gym.make("CartPole-v1")
    train_env.action_space.seed(seed)

    # Create agent's value function approximator model
    print("Creating agent and collecting samples...")
    MODEL = ValueFunctionApproximator(train_env, seed=seed)
    print("Agent Created and Samples Collected!")
    print()

    # Train the agent
    print("Training Agent...")
    reward_per_episode = train_agent(train_env, num_episodes=1500, model=MODEL, seed=seed)
    print("Training Complete!")
    plot_reward_per_episode(reward_per_episode)

    # Evaluate the trained agent
    print("Evaluating Trained Agent...")
    mean_reward = evaluate_trained_agent(train_env, num_episodes=100, model=MODEL, seed=seed + 10_000)
    print(f"Mean Reward: {mean_reward}")
    train_env.close()
    print("Evaluation Complete!")

    # Watch the trained agent
    print("Watching Trained Agent...")
    render_env = gym.make("CartPole-v1", render_mode="human")
    watch_agent(render_env, model=MODEL)
    render_env.close()
    print("Watching Complete!")
    print("Execution Complete!")
    print("Goodbye!")
    print("Program Terminated Successfully!")
    print("Thank you for using the CartPole Control Approximation Algorithm!")


if __name__ == "__main__":
    main()
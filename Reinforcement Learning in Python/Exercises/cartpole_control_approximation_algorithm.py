from __future__ import annotations

import gymnasium
from random import choice, random
import numpy as np
import matplotlib.pyplot as plt
from numpy import signedinteger, floating, unsignedinteger
from numpy._typing import _32Bit, _64Bit
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Any

# =======================
# Type Aliases
# =======================
State = np.ndarray
Action = str | int
Reward = float
Policy = Dict[State, Action]
ValueTable = Dict[State, Reward]
StateSampleCounts = Dict[State, int]
Samples = List[np.ndarray]
ActionSpace = Dict[State, Tuple[Action, ...]]

# =======================
# Policy
# =======================

def epsilon_greedy(model: ValueFunctionApproximator, state: State, eps: float) -> unsignedinteger[_32Bit | _64Bit] | Any:
    if random() < (1 - eps):
        values = model.predict_all_actions(state)
        return np.argmax(values)
    else:
        return model.env.action_space.sample()

# =======================
# Sample Collection
# =======================

def collect_samples(env: gymnasium.Env, num_samples: int) -> Samples:
    samples: Samples = []
    for _ in tqdm(range(num_samples)):
        s, info = env.reset()
        done: bool = False
        truncated: bool = False

        while not (done or truncated):
            a = env.action_space.sample()
            sa = np.concatenate((s, [a]))
            samples.append(sa)

            s, r, done, truncated, info = env.step(a)

    return samples

# =======================
# Run Agent Episode
# =======================
def run_episode(model: ValueFunctionApproximator, env: gymnasium.Env, eps: float) -> float:
    s, info = env.reset()
    done: bool = False
    truncated: bool = False
    episode_reward: float = 0.0

    while not (done or truncated):
        a = epsilon_greedy(model, s, eps)
        s, r, done, truncated, info = env.step(a)
        episode_reward += r

    return episode_reward

# =======================
# Value Function Approximator
# =======================
class ValueFunctionApproximator:

    def __init__(self, env: gymnasium.Env) -> None:
        self.env = env
        samples = collect_samples(env, num_samples=10)  # Reduced from 10,000 for faster training
        self.sampler = RBFSampler(gamma=0.5, n_components=100)
        self.sampler.fit(samples)
        dims = self.sampler.n_components
        self.w = np.zeros(dims)

    def predict(self, state: State, a: Action) -> np.ndarray:
        sa = np.concatenate((state, [a]))  # Action is not used for prediction
        x = self.sampler.transform([sa])[0]
        return x @ self.w

    def predict_all_actions(self, state: State) -> np.ndarray:
        sa_batch: List[np.ndarray] = [self.predict(state, a) for a in range(self.env.action_space.n)]

        return np.array(sa_batch)

    def grad(self, state: State, a: Action) -> np.ndarray:
        sa = np.concatenate((state, [a]))
        x = self.sampler.transform([sa])[0]
        return x

def evaluate_trained_agent(model: ValueFunctionApproximator, env: gymnasium.Env, num_episodes: int, eps: float) -> floating[Any]:
    reward_per_episode: np.ndarray = np.zeros(num_episodes)

    for it in tqdm(range(num_episodes), desc="Episodes Running", leave=False):
        episode_reward = run_episode(model, env, eps)
        reward_per_episode[it] = episode_reward

    return np.mean(reward_per_episode)

def watch_agent(model: ValueFunctionApproximator, env: gymnasium.Env, eps: float) -> None:
    episode_reward = run_episode(model, env, eps)
    print(f"Episode Reward: {episode_reward}")

    return None

def main() -> None:
    # Create training environment without rendering (faster)
    train_env = gymnasium.make("CartPole-v1")

    model = ValueFunctionApproximator(train_env)
    reward_per_episode = []

    # Create rendering environment for watching the agent
    render_env = gymnasium.make("CartPole-v1", render_mode="human")
    watch_agent(model, render_env, eps=0.1)
    render_env.close()
    train_env.close()

if __name__ == "__main__":
    main()







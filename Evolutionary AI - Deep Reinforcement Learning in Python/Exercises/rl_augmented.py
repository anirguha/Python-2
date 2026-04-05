# %%
# Import necessary libraries
import argparse
import json
import numpy as np
from datetime import datetime
from functools import partial
from pathlib import Path
from uuid import uuid4
import gymnasium as gym
import matplotlib.pyplot as plt
from multiprocess import Pool

ENV_NAME = "MountainCar-v0"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
VIDEO_DIR = SCRIPT_DIR / " videos"


def parse_args():
    parser = argparse.ArgumentParser(description="Train an ARS policy on MountainCar-v0.")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of ARS training iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Gradient ascent step size.")
    parser.add_argument("--population-size", type=int, default=64, help="Number of perturbation pairs per iteration.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise scale for parameter perturbations.")
    parser.add_argument("--processes", type=int, default=8, help="Number of worker processes for rollout evaluation.")
    parser.add_argument("--hidden-units", type=int, default=M, help="Hidden units in the policy network.")
    parser.add_argument("--stats-episodes", type=int, default=5, help="Episodes used to estimate observation normalization stats.")
    parser.add_argument("--stats-max-steps", type=int, default=200, help="Max steps per stats collection episode.")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience in iterations.")
    parser.add_argument("--min-improvement", type=float, default=1e-3, help="Minimum average reward gain to reset patience.")
    parser.add_argument("--position-reward-scale", type=float, default=100.0, help="Scale factor for max-position progress in the shaped training objective.")
    parser.add_argument("--velocity-reward-scale", type=float, default=10.0, help="Scale factor for max-velocity progress in the shaped training objective.")
    parser.add_argument("--success-reward", type=float, default=1000.0, help="Bonus added to the shaped training objective when the goal is reached.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for NumPy and environment resets.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes used to evaluate the final policy.")
    parser.add_argument(
        "--record-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record a video of the final policy. Use --no-record-final to disable.",
    )
    return parser.parse_args()


def get_env_dimensions(env_name):
    env = gym.make(env_name)
    try:
        return env.observation_space.shape[0], env.action_space.n
    finally:
        env.close()


def get_env_bounds(env_name):
    env = gym.make(env_name)
    try:
        return float(env.unwrapped.min_position), float(env.unwrapped.goal_position)
    finally:
        env.close()


# %%
# environment setup
D, K = get_env_dimensions(ENV_NAME)
MIN_POSITION, GOAL_POSITION = get_env_bounds(ENV_NAME)
M = 128  # Number of hidden units of the neural network to be used as the policy
print("State space dimension: ", D)
print("Action space dimension: ", K) 

# %%
# Define non-linear function for the neural network
def relu(x):
    return np.maximum(0, x)

# %%
# Neural Network Policy
class ANN:
    def __init__(self, D, M, K, f=relu):
        self.D = D
        self.M = M
        self.K = K
        self.f = f
        
    # Initialize weights
    def init(self):
        self.W1 = np.random.randn(self.D, self.M) / np.sqrt(self.D)
        self.W2 = np.random.randn(self.M, self.K) / np.sqrt(self.M)
        self.b1 = np.zeros((self.M, 1))
        self.b2 = np.zeros((self.K, 1))
        
    # Forward pass
    def forward(self, X):
        Z = self.f(X @ self.W1 + self.b1.T)
        return Z @ self.W2 + self.b2.T
    
    # Convert the 1-D action vector to 2-D array and return the action
    def sample_actions(self, x):
        X = np.atleast_2d(x)
        Y = self.forward(X)
        return np.argmax(Y[0])
    
    # Return all parameters as a single vector
    def get_params(self):
        return np.concatenate([self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten()])
    
    # Store the parameters in a dictionary to be used later for saving and loading the model
    def get_params_dict(self):
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2
        }
        
    def set_params(self, params):
        # params is a flat list
        # unflatten into individual weights
        D, M, K = self.D, self.M, self.K
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D * M: D * M + M].reshape(M, 1)
        self.W2 = params[D * M + M: D * M + M + M * K].reshape(M, K)
        self.b2 = params[-K:].reshape(K, 1)

# %%
# Class for Online standardization of the states
class OnlineStandardizer:
    def __init__(self, D):
        self.n = 0
        self.mean = np.zeros(D)
        self.var = np.zeros(D)

        
    def partial_fit(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.var += delta * delta2 

    def get_stats(self):
        if self.n < 2:
            return self.mean.copy(), np.ones_like(self.mean)

        variance = np.maximum(self.var / (self.n - 1), 1e-8)
        return self.mean.copy(), np.sqrt(variance)
    
    def transform(self, x):
        mean, std = self.get_stats()
        return (x - mean) / (std + 1e-8)


def collect_observation_stats(env_name, num_episodes=5, max_steps=200, seed=123):
    env = gym.make(env_name)
    scaler = OnlineStandardizer(D)
    rng = np.random.default_rng(seed)

    try:
        for episode_idx in range(num_episodes):
            obs, _ = env.reset(seed=seed + episode_idx)
            scaler.partial_fit(obs)

            done = False
            steps = 0
            while not done and steps < max_steps:
                action = int(rng.integers(K))
                obs, _, terminated, truncated, _ = env.step(action)
                scaler.partial_fit(obs)
                done = terminated or truncated
                steps += 1
    finally:
        env.close()

    return scaler.get_stats()


def save_training_artifacts(reward_history, final_params, evaluation_rewards, evaluation_summary, config):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "reward_history.npy", reward_history)
    np.save(OUTPUT_DIR / "final_params.npy", final_params)
    np.save(OUTPUT_DIR / "evaluation_rewards.npy", evaluation_rewards)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "evaluation": evaluation_summary,
        "num_iterations_completed": int(len(reward_history)),
        "config": config,
    }
    (OUTPUT_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, len(reward_history) + 1), reward_history, linewidth=2)
    ax.set_title("MountainCar ARS Training Reward")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reward_history.png", dpi=150)
    plt.close(fig)

# %%
# Function for optimizer
class GradientAscentOptimizer:
    def __init__(self, params, learning_rate=0.01):
        self.params = params
        self.alpha = learning_rate

    def update(self, grad):
        self.params += self.alpha * grad
        return self.params

# %%
# Function for evolution strategies
def evolution_strategy(initial_params, num_iterations, lr, population_size, sigma, f, pool,
                       patience=25, min_improvement=1e-3):
    
    # Intialize
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iterations)
    
    # Initialize optimizer
    params = initial_params.copy()
    optimizer = GradientAscentOptimizer(params, learning_rate=lr)
    best_reward = -np.inf
    stale_iterations = 0
    
    for iteration in range(num_iterations):
        t0 = datetime.now()
        eps = np.random.randn(population_size, num_params)
        
        p = [params + sigma * eps[i] for i in range(population_size)] + \
            [params - sigma * eps[i] for i in range(population_size)]
        
        R = pool.map(f, p)
        R = np.array(R) # Covert list to numpy array
        
        # Split into positive and negative rewards
        R_pos = R[:population_size]
        R_neg = R[population_size:]
        
        m = R.mean()
        s = R.std()
        s = s if s > 1e-8 else 1e-8  # Avoid division by zero
        
        reward_per_iteration[iteration] = m
        g = eps.T @ (R_pos - R_neg) / (population_size * s)
        params = optimizer.update(g)
        
        if m > best_reward + min_improvement:
            best_reward = m
            stale_iterations = 0
        else:
            stale_iterations += 1

        if stale_iterations >= patience:
            print("Convergence detected. Stopping training.")
            break
        
        t1 = datetime.now()
        print(f"Iteration: {iteration+1}/{num_iterations}, Average Reward: {m:.2f}, Time: {(t1-t0).total_seconds():.2f} seconds")
        
    completed_iterations = iteration + 1
    return reward_per_iteration[:completed_iterations], params


def normalize_observation(obs, obs_mean=None, obs_std=None):
    if obs_mean is None or obs_std is None:
        return obs

    return (obs - obs_mean) / (obs_std + 1e-8)


def make_env(env_name, record=False, video_root=VIDEO_DIR):
    if record:
        env = gym.make(env_name, render_mode="rgb_array")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        video_root.mkdir(parents=True, exist_ok=True)
        video_folder = video_root / f"mountaincar-{timestamp}-{uuid4().hex[:8]}"
        print(f"Recording final policy video to: {video_folder}")
        env = gym.wrappers.RecordVideo(env, video_folder=str(video_folder), episode_trigger=lambda _: True)
    else:
        env = gym.make(env_name)

    return gym.wrappers.RecordEpisodeStatistics(env)


def run_episode(model, env_name=ENV_NAME, record=False, obs_mean=None, obs_std=None, seed=None):
    env = make_env(env_name, record=record)

    episode_reward = 0.0
    episode_length = 0
    done = False
    obs, _ = env.reset(seed=seed)
    max_position = float(obs[0])
    max_speed = float(abs(obs[1]))
    success = False

    while not done:
        normalized_obs = normalize_observation(obs, obs_mean=obs_mean, obs_std=obs_std)
        action = model.sample_actions(normalized_obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        max_position = max(max_position, float(obs[0]))
        max_speed = max(max_speed, float(abs(obs[1])))
        success = success or bool(terminated)
        done = terminated or truncated

    env.close()

    episode_info = info.get("episode")
    assert episode_info is not None, "Missing episode statistics."
    assert np.isclose(float(episode_info["r"]), episode_reward), "Reward mismatch!"
    assert int(episode_info["l"]) == episode_length, "Episode length mismatch!"

    return {
        "reward": float(episode_reward),
        "length": int(episode_length),
        "success": bool(success),
        "max_position": float(max_position),
        "max_speed": float(max_speed),
    }

# %%
# Define the reward function to be used in the evolution strategy
def reward_function(params, record=False, env_name=ENV_NAME, obs_mean=None, obs_std=None, seed=None):
    model: ANN = ANN(D, M, K)
    model.set_params(params)
    episode = run_episode(model, env_name=env_name, record=record, obs_mean=obs_mean, obs_std=obs_std, seed=seed)
    return episode["reward"]


def training_reward_function(params, env_name=ENV_NAME, obs_mean=None, obs_std=None, seed=None,
                             position_reward_scale=100.0, velocity_reward_scale=10.0,
                             success_reward=1000.0):
    model: ANN = ANN(D, M, K)
    model.set_params(params)
    episode = run_episode(model, env_name=env_name, record=False, obs_mean=obs_mean, obs_std=obs_std, seed=seed)

    normalized_progress = (episode["max_position"] - MIN_POSITION) / (GOAL_POSITION - MIN_POSITION)
    shaped_reward = (
        episode["reward"]
        + position_reward_scale * normalized_progress
        + velocity_reward_scale * episode["max_speed"]
        + (success_reward if episode["success"] else 0.0)
    )
    return float(shaped_reward)


def evaluate_policy(params, num_episodes, env_name=ENV_NAME, obs_mean=None, obs_std=None, seed=123):
    model = ANN(D, M, K)
    model.set_params(params)
    episodes = [
        run_episode(
            model,
            env_name=env_name,
            record=False,
            obs_mean=obs_mean,
            obs_std=obs_std,
            seed=seed + episode_idx,
        )
        for episode_idx in range(num_episodes)
    ]

    rewards = np.array([episode["reward"] for episode in episodes], dtype=float)
    max_positions = np.array([episode["max_position"] for episode in episodes], dtype=float)
    successes = np.array([episode["success"] for episode in episodes], dtype=float)

    summary = {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(ddof=0)),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "mean_max_position": float(max_positions.mean()),
        "best_max_position": float(max_positions.max()),
        "success_rate": float(successes.mean()),
        "num_episodes": int(num_episodes),
    }
    return rewards, summary
# %%
# Main function to run the evolution strategy
if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    # Initialize the model and get the initial parameters
    model = ANN(D, args.hidden_units, K)
    model.init()
    initial_params = model.get_params()
    
    # Keep the module-level hidden size in sync with the configured policy.
    M = args.hidden_units

    obs_mean, obs_std = collect_observation_stats(
        ENV_NAME,
        num_episodes=args.stats_episodes,
        max_steps=args.stats_max_steps,
        seed=args.seed,
    )
    reward_fn = partial(
        training_reward_function,
        env_name=ENV_NAME,
        obs_mean=obs_mean,
        obs_std=obs_std,
        position_reward_scale=args.position_reward_scale,
        velocity_reward_scale=args.velocity_reward_scale,
        success_reward=args.success_reward,
    )

    # Run the evolution strategy with a pool that is cleaned up deterministically
    with Pool(processes=args.processes) as pool:
        reward_per_iteration, final_params = evolution_strategy(initial_params, args.iterations, \
            args.learning_rate, args.population_size, args.sigma, reward_fn, pool,
            patience=args.patience, min_improvement=args.min_improvement)

    evaluation_rewards, evaluation_summary = evaluate_policy(
        final_params,
        num_episodes=args.eval_episodes,
        env_name=ENV_NAME,
        obs_mean=obs_mean,
        obs_std=obs_std,
        seed=args.seed,
    )

    if args.record_final:
        reward_function(
            final_params,
            record=True,
            env_name=ENV_NAME,
            obs_mean=obs_mean,
            obs_std=obs_std,
            seed=args.seed,
        )

    save_training_artifacts(
        reward_per_iteration,
        final_params,
        evaluation_rewards,
        evaluation_summary,
        {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "population_size": args.population_size,
            "sigma": args.sigma,
            "processes": args.processes,
            "hidden_units": args.hidden_units,
            "stats_episodes": args.stats_episodes,
            "stats_max_steps": args.stats_max_steps,
            "patience": args.patience,
            "min_improvement": args.min_improvement,
            "position_reward_scale": args.position_reward_scale,
            "velocity_reward_scale": args.velocity_reward_scale,
            "success_reward": args.success_reward,
            "seed": args.seed,
            "eval_episodes": args.eval_episodes,
            "record_final": args.record_final,
        },
    )
    print(
        "Final policy evaluation: "
        f"mean={evaluation_summary['mean_reward']:.2f}, "
        f"std={evaluation_summary['std_reward']:.2f}, "
        f"best_position={evaluation_summary['best_max_position']:.3f}, "
        f"success_rate={evaluation_summary['success_rate']:.2%}, "
        f"episodes={evaluation_summary['num_episodes']}"
    )

# %%

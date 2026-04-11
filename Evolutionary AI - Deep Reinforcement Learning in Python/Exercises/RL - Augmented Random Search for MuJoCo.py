# %%
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
from multiprocess import Pool
from datetime import datetime

ENV_NAME = "HalfCheetah-v5"
# %%
# Hyperparameters
env = gym.make(ENV_NAME)
D = env.observation_space.shape[0]  # Number of states
M = 128  # Number of hidden units
K = env.action_space.shape[0]  # Number of actions
action_max = env.action_space.high[0]
print("State space dimension: ", D)
print("Action space dimension: ", K)
print("Action space max: ", action_max)

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
        return (Z @ self.W2 + self.b2.T) * action_max
    
    # Convert the 1-D action vector to 2-D array and return the action
    def sample_actions(self, x):
        X = np.atleast_2d(x)
        Y = self.forward(X)
        return Y[0]
    
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
        self.b1 = params[D * M: D * M + M]
        self.W2 = params[D * M + M: D * M + M + M * K].reshape(M, K)
        self.b2 = params[-K:]
    
    
        
# %%
# Class for online standardization of states
class OnlineStandardScaler:
    def __init__(self, num_inputs):
        self.n = 0
        self.mean = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)
        
    def partial_fit(self, x):
        self.n += 1
        delta1 = x - self.mean
        self.mean += delta1 / self.n
        delta2 = x - self.mean
        self.var += delta1 * delta2
        
    def transform(self, x):
        return (x - self.mean) / (np.sqrt(self.var / self.n) + 1e-8)
    
scaler = OnlineStandardScaler(D)
# %%
# Class for optimizer
class Optimizer:
    def __init__(self, params, learning_rate=0.01):
        self.params = params
        self.learning_rate = learning_rate
        
    def update(self,grad):
        self.params += self.learning_rate * grad
        
        return self.params

# %%
# Function for evolution strategies
def evolution_strategy(initial_params, num_iterations, lr, population_size, sigma, f, pool):
    
    # Intialize
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iterations)
    
    # Initialize optimizer
    params = initial_params.copy()
    adam = Optimizer(params, learning_rate=lr)
    
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
        if s == 0:
            print("Standard deviation is zero. Skipping iteration.")
            continue
        
        reward_per_iteration[iteration] = m
        g = eps.T @ (R_pos - R_neg) / (population_size * s)
        params = adam.update(g)
        
        t1 = datetime.now()
        print(f"Iteration: {iteration+1}/{num_iterations}, Average Reward: {m:.2f}, Max Reward: {R.max():.2f}, \
            Time: {(t1-t0).total_seconds():.2f} seconds")
        
    return reward_per_iteration, params

# %%
# Define the rward function to be used in the evolution strategy
def reward_function(params, record=False, env_name=ENV_NAME):
    model = ANN(D, M, K)
    model.set_params(params)
    
    # Record videos of the agent's performance in record mode else just create the environment
    if record:
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda eps: True)
    else:
        env = gym.make(env_name)
        
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Play one episode and return the total reward
    episode_reward = 0
    episode_length = 0
    done = False
    obs, _ = env.reset()
    while not done:
        scaler.partial_fit(obs)
        obs = scaler.transform(obs)
        action = model.sample_actions(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        done = terminated or truncated
        
    env.close()
    
    assert info["episode"]["r"] == episode_reward, "Reward mismatch!"
    assert info["episode"]["l"] == episode_length, "Episode length mismatch!"
    
    return episode_reward
    
# %%

# Main function to run the evolution strategy
if __name__ == "__main__":
    # Initialize the model and get the initial parameters
    model = ANN(D, M, K)
    model.init()
    initial_params = model.get_params()
    
    # Hyperparameters for evolution strategy
    num_iterations = 1000
    learning_rate = 0.01
    population_size = 64
    sigma = 0.1
    
    # Create a pool of workers for parallel evaluation
    pool = Pool(processes=8)
    
    # Run the evolution strategy
    reward_per_iteration, final_params = evolution_strategy(initial_params, num_iterations, \
        learning_rate, population_size, sigma, reward_function, pool)
    
    # Plot the rewards per iteration
    plt.plot(reward_per_iteration)
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title("Evolution Strategy - Average Reward per Iteration")
    plt.show()
    
    # Save the final model parameters
    np.save("final_params.npy", final_params)
    
    # Test the final model and record a video
    final_reward = reward_function(final_params, record=True)
    print(f"Final Reward: {final_reward:.2f}")

# %%

# %%
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
from multiprocess import Pool

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

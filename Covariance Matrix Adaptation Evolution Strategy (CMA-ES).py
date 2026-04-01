# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import gymnasium as gym

# import multiprocessing
from multiprocess import Pool

import sys

import cma

env_name = "HalfCheetah-v5"

# Hyperparameters
_env = gym.make(env_name)
D = np.prod(_env.observation_space.shape)
M = 128 # No. of hidden layers
K = _env.action_space.shape[0]
action_max = _env.action_space.high[0]
_env.close()
del _env

def relu(x):
    return np.maximum(0, x)


# Define ANN
class ANN:
    def __init__(self, D, M, K, f=relu):
        self.D = D
        self.M = M
        self.K = K
        self.f = f

    def init(self):
        self.W1 = np.random.randn(self.D, self.M) * np.sqrt( self.D)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, self.K) * np.sqrt(self.M)
        self.b2 = np.zeros(self.K)

    def forward(self, x):
        Z = self.f(np.dot(x, self.W1) + self.b1)
        return np.tanh(Z @ self.W2 + self.b2) * action_max

    def sample_action(self, x):
        # assume that the input is a vector of size (D, K)
        # Convert it to (N, D) matrix for ML
        X = np.atleast_2d(x)
        Y = self.forward(X)
        return Y[0]

    def get_params(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def get_params_dict(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_params(self, params):
        D, M, K = self.D, self.M, self.K
        self.W1 = params[:D*M].reshape(D, M)
        self.b1 = params[D*M:D*M+M]
        self.W2 = params[D*M+M:D*M+M+M*K].reshape(M, K)
        self.b2 = params[-K:]

    def __repr__(self):
        return f"ANN(D={self.D}, M={self.M}, K={self.K}, f={self.f.__name__})"

class OnlineStandardScaler:
    def __init__(self, num_inputs):
        self.n = 0
        self.mean = np.zeros(num_inputs)
        self.ssd = np.zeros(num_inputs)  # sum of squared deviations

    def partial_fit(self, X):
        # Welford's online algorithm for mean and variance
        self.n += 1
        delta = X - self.mean
        self.mean += delta / self.n
        delta2 = X - self.mean
        self.ssd += delta * delta2

    def transform(self, X):
        m = self.mean
        v = (self.ssd / self.n).clip(min=1e-2)
        s = np.sqrt(v)
        return (X - m) / s

scaler = OnlineStandardScaler(D)

def evolution_strategy(
  f,
  population_size,
  sigma,
  lr,
  initial_params,
  num_iters,
  pool,
  ):

  # assume initial params is a 1-D array
  num_params = len(initial_params)
  reward_per_iteration = np.array([])

  # create optimizer
  params = initial_params
  options = {
      # 'popsize': population_size,
      'AdaptSigma': True,
      'maxiter': num_iters,
      'verbose': -9,
      'verb_disp': 1,
      'CMA_diagonal': False, # returns full covarinace matrix,
      'CMA_mirrors': True
  }
  es = cma.CMAEvolutionStrategy(initial_params, sigma, options)

  while not es.stop():
    t0 = datetime.now()
    solutions = es.ask()

    ### fast way
    R = pool.map(f, solutions)
    R = np.array(R)

    es.tell(solutions, -R) # CMS-ES minimizes, so we negate the reward
    reward_per_iteration = np.append(reward_per_iteration, R.mean())

    print("Iter:", es.countiter, "Avg Reward:", R.mean(), "Max Reward:", R.max(), "Duration:", datetime.now() - t0)

  return es.best.get()[0], reward_per_iteration

def reward_function(params, record=False, env_name=env_name):
  # run one episode of env w/ params
  model = ANN(D, M, K)
  model.set_params(params)

  if record:
      env = gym.make(env_name, render_mode="rgb_array")
      video_folder = f"videos/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
      env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda eps: True)
  else:
      env = gym.make(env_name)
  env = gym.wrappers.RecordEpisodeStatistics(env)

  # play one episode and return total reward
  episode_reward = 0
  episode_length = 0
  done = False
  state, _ = env.reset()
  while not done:
      scaler.partial_fit(state)
      state = scaler.transform(state)

      # get action
      action = model.sample_action(state)

      # perform action
      state, reward, done, truncated, info = env.step(action)
      done = done or truncated

      # update total reward and length
      episode_reward += reward
      episode_length += 1

  # close env
  env.close()

  assert (info['episode']['r'] == episode_reward)

  return episode_reward

if __name__ == '__main__':

  # create model
  model = ANN(D, M, K)

  if len(sys.argv) > 1 and sys.argv[1] == 'play':
    # play with a saved model
    j = np.load('es_mujoco_results.cma.npz')
    best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])
  else:
    rewards = None
    # pool for parallel evaluation
    with Pool(4) as pool:
      # train and save model
      model.init()
      params = model.get_params()
      best_params, rewards = evolution_strategy(
        f=reward_function,
        population_size=30,
        sigma=0.05,
        lr=0.02,
        initial_params=params,
        num_iters=150,
        pool=pool,
      )

    if rewards is not None:
      # plot the rewards per iteration
      plt.plot(rewards)
      plt.xlabel("Iteration")
      plt.ylabel("Reward")
      plt.title("Reward per Iteration")
      plt.show()

      # save params
      model.set_params(best_params)
      np.savez(
        'es_mujoco_results.cma',
        train=rewards,
        **model.get_params_dict(),
      )

  # play with saved model / test episode
  print("Test:", reward_function(best_params, record=True))
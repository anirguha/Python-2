# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import talib
from datetime import datetime
from functools import partial
from typing import Tuple, Callable
import multiprocessing
from multiprocessing import Pool
import sys
from multiprocessing.pool import Pool as PoolType
# %%
# Set the period for data retrieval
start_date = '2010-01-01'
end_date = '2024-12-31'
# Define the list of stock tickers
tickers = ['VNQ', 'SPY', 'GLD', 'BTC-USD']
# Last element is the cash allocation weight (0 = fully invested, no cash held).
benchmark_weights = np.array([0.1, 0.8, 0.05, 0.05, 0.0])
# Function to download and preprocess data
try:
    downloaded_data = yf.download(tickers, start=start_date, end=end_date, interval='1mo', progress=False)
    data = downloaded_data['Close'] if downloaded_data is not None else None
except Exception as e:
    print(f"An error occurred while downloading data: {e}")
    data = None

if isinstance(data, pd.Series):
  # Normalize to DataFrame shape when only a single close series is returned.
  data = data.to_frame(name=tickers[0])
    
if data is not None:
    data = data.dropna()

return_tickers = []
for ticker in tickers:
    rtick = f'{ticker}_return'
    if data is not None:
        data[rtick] = data[ticker].pct_change()
    return_tickers.append(rtick)
    

# %%
# Compute RSI and MACD indicators
if data is not None:
  for ticker in tickers:
    price_arr = data[ticker].to_numpy(dtype=float)
    data[f'ticker_{ticker}_rsi'] = talib.RSI(price_arr, timeperiod=14) / 50 - 1 # To normalize between -1 and 1
    macd, macdsignal, macdhist = talib.MACD(price_arr, fastperiod=12, slowperiod=26, signalperiod=9)
    data[f'ticker_{ticker}_macd'] = macd / data[ticker] * 10 # Normalize by price to get a percentage
    data[f'ticker_{ticker}_macdsignal'] = macdsignal / data[ticker] * 10 # Normalize by price to get a percentage
    data[f'ticker_{ticker}_macdhist'] = macdhist
  # Drop warm-up rows introduced by rolling indicators.
  data = data.dropna()
# %%
# Split into train and test data
train_data = None
test_data = None
if data is not None:
    split = int(len(data) * 0.8)
    train_data = data.iloc[:split]
    test_data = data.iloc[split:]

# %%
# Create an environment class for portfolio optimization
class Env:
    def __init__(self, data:pd.DataFrame, tickers:list, lag:int=5):
        self.columns = []
        for ticker in tickers:
            self.columns.append(f'{ticker}_return')
            self.columns.append(f'ticker_{ticker}_rsi')
            self.columns.append(f'ticker_{ticker}_macd')
            self.columns.append(f'ticker_{ticker}_macdsignal')
            self.columns.append(f'ticker_{ticker}_macdhist') 
        self.state = data[self.columns].to_numpy()
        self.prices = data[tickers].to_numpy() # for computing returns
        self.current_step = lag
        self.lag = lag
        
    # resets the environment to the initial state
    def reset(self):
        self.pos = self.lag
        return self.state[:self.pos].flatten()
    
    def step(self, action):
        action = np.asarray(action).reshape(-1)

        # Validate the dimensions of the action space
        expected_action_dim = self.prices.shape[-1] + 1  # include cash weight
        if action.shape[-1] != expected_action_dim:
            raise ValueError(
                f"Action shape {action.shape} does not match expected dimension {expected_action_dim}"
            )

        # Compute the portfolio return
        next_pos = self.pos + self.current_step
        asset_returns = (self.prices[next_pos] - self.prices[self.pos]) / self.prices[self.pos]
        asset_returns = np.concatenate((asset_returns, [0])) # Add a zero return for the cash position
        portfolio_return = np.dot(action, asset_returns) # reward is the portfolio return

        # Update the current position and return the next state, reward, and done flag
        self.pos = next_pos

        # Check if we have reached the end of the data
        done = self.pos >= len(self.state) - self.current_step

        return self.state[self.pos - self.current_step: self.pos].flatten(), portfolio_return, done, asset_returns
            
# %%
# Define RELU function for the ANN
def relu(x):
    return np.maximum(0, x)

#%% Define the softmax function for action selection
def softmax(x):
    e_x = np.exp(x - np.max(x)) # for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)
# %%
class ANN:
  def __init__(self, D, M, K, f=relu):
    self.D = D
    self.M = M
    self.K = K
    self.f = f

  def init(self):
    D, M, K = self.D, self.M, self.K
    self.W1 = np.random.randn(D, M) / np.sqrt(D)
    # self.W1 = np.zeros((D, M))
    self.b1 = np.zeros(M)
    self.W2 = np.random.randn(M, K) / np.sqrt(M)
    # self.W2 = np.zeros((M, K))
    self.b2 = np.zeros(K)

  def forward(self, X):
    Z = self.f(X.dot(self.W1) + self.b1)
    return Z.dot(self.W2) + self.b2

  # def sample_action(self, x):
  #   # assume input is a single state of size (D,)
  #   # first make it (N, D) to fit ML conventions
  #   X = np.atleast_2d(x)
  #   Y = self.forward(X)
  #   return Y[0] # the first row

  def get_params(self):
    # return a flat array of parameters
    return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

  def get_params_dict(self):
    return {
      'W1': self.W1,
      'b1': self.b1,
      'W2': self.W2,
      'b2': self.b2,
    }

  def set_params(self, params):
    # params is a flat list
    # unflatten into individual weights
    D, M, K = self.D, self.M, self.K
    self.W1 = params[:D * M].reshape(D, M)
    self.b1 = params[D * M:D * M + M]
    self.W2 = params[D * M + M:D * M + M + M * K].reshape(M, K)
    self.b2 = params[-K:]

# %%
class OnlineStandardScaler:
  def __init__(self, num_inputs):
    self.n = 0
    self.mean = np.zeros(num_inputs)
    self.ssd = np.zeros(num_inputs)

  def partial_fit(self, X):
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

# %%
class Adam:
  def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    self.lr = lr
    self.m = 0 # first moment
    self.v = 0 # second moment
    self.b1 = beta1
    self.b2 = beta2
    self.eps = eps
    self.t = 1 # time step
    self.params = params

  def update(self, g):
    # new m
    self.m = self.b1 * self.m + (1 - self.b1) * g
    # new v
    self.v = self.b2 * self.v + (1 - self.b2) * g**2
    # bias correction
    m_hat = self.m / (1 - self.b1**self.t)
    v_hat = self.v / (1 - self.b2**self.t)
    # update time step
    self.t += 1
    # update params
    self.params += self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    return self.params

# %%
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
  reward_per_iteration = np.zeros(num_iters)

  # create optimizer
  params = initial_params
  adam = Adam(params, lr)

  for t in range(num_iters):
    t0 = datetime.now()
    eps = np.random.randn(population_size, num_params)

    ### slow way
    # R = np.zeros(population_size)
    # for i in range(population_size):
    #   R[i] = f(params + sigma * eps[i])

    ### fast way
    R = pool.map(f, [params + sigma * eps[i] for i in range(population_size)])
    R = np.array(R)

    m = R.mean()
    s = R.std()
    if s == 0:
      # we can't apply the following equation
      print("Skipping")
      continue

    A = (R - m) / s
    reward_per_iteration[t] = m
    g = eps.T @ A / (population_size * sigma)
    params = adam.update(g)

    print("Iter:", t, "Avg Reward:", m, "Max Reward:", R.max(), "Duration:", datetime.now() - t0)

  return params, reward_per_iteration

# %%
# Define sortino ratio function for evaluation
def sortino_ratio(returns, risk_free_rate=0):
  returns = np.asarray(returns, dtype=float)
  returns = returns[np.isfinite(returns)]
  if returns.size == 0:
    return 0.0

  # Calculate the mean return and the downside deviation
  mean_return = np.mean(returns)
  downside_returns = returns[returns < risk_free_rate]
  downside_deviation = np.sqrt(np.mean(downside_returns**2))

  # Avoid division by zero
  if downside_deviation == 0:
    return np.inf  # Infinite Sortino ratio if there is no downside risk

  # Calculate and return the Sortino ratio
  return (mean_return - risk_free_rate) / downside_deviation

# %%
# Define Sharpe ratio function for evaluation
def sharpe_ratio(returns, risk_free_rate=0):
  returns = np.asarray(returns, dtype=float)
  returns = returns[np.isfinite(returns)]
  if returns.size == 0:
    return 0.0

  # Calculate the mean return and the standard deviation of returns
  mean_return = np.mean(returns)
  std_return = np.std(returns)

  # Avoid division by zero
  if std_return == 0:
    return np.inf  # Infinite Sharpe ratio if there is no volatility

  # Calculate and return the Sharpe ratio
  return (mean_return - risk_free_rate) / std_return

# %%
# Define hyperparameters for the ES algorithm
if train_data is not None:
    D = len(Env(train_data, tickers).reset()) # type: ignore
else:
    raise ValueError("train_data is None. Data download or preprocessing failed.")
M = 128
K = len(tickers) + 1
scaler = OnlineStandardScaler(D)

# %%
# Define the reward function for the ES algorithm
def reward_function(params, df=None, plot=False):
  if df is None:
    df = train_data
  if df is None:
    raise ValueError("Training data is unavailable for reward evaluation.")

  model = ANN(D, M, K)
  model.set_params(params)

  # create environment
  env = Env(df, tickers)

  portfolio_returns = []
  asset_returns = [] # keep track of each individual asset's returns for benchmark
  obs = env.reset()
  done = False
  G = 0 # not needed if we use sortino ratio

  while not done:
    scaler.partial_fit(obs)
    x = scaler.transform(obs)
    logits = model.forward(x)

    # get action
    weights = softmax(logits)

    # perform action
    obs, reward, done, individual_returns = env.step(weights)
    portfolio_returns.append(reward)
    asset_returns.append(individual_returns)

    # update total reward
    G += reward

  # convert to numpy
  portfolio_returns = np.asarray(portfolio_returns, dtype=float)
  asset_returns = np.asarray(asset_returns, dtype=float)

  if plot:
    cumulative_gross_return = np.cumprod(1 + portfolio_returns)

    # compute benchmark returns
    benchmark_returns = asset_returns @ benchmark_weights
    cumulative_benchmark_return = np.cumprod(1 + benchmark_returns)

    # plot
    plt.plot(cumulative_gross_return, label='es')
    plt.plot(cumulative_benchmark_return, label='benchmark')
    plt.legend()
    plt.title("Cumulative Gross Return")
    plt.show()

    print("Sortino/Sharpe Ratio for Benchmark:", sharpe_ratio(benchmark_returns))

  return sharpe_ratio(portfolio_returns)

# %%
# Main function to run the ES algorithm
if __name__ == '__main__':

  # create model
  model = ANN(D, M, K)

  if len(sys.argv) > 1 and sys.argv[1] == 'play':
    # play with a saved model
    j = np.load('es_dynamic_portfolio_results.npz')
    best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])

  else:
    # pool for parallel evaluation — fork context avoids pickling reward_function,
    # which would fail under spawn (macOS/Windows default) because worker processes
    # cannot reimport a function defined in __main__.
    pool = multiprocessing.get_context('fork').Pool(4)

    # train and save model
    model.init()
    params = model.get_params()
    best_params, rewards = evolution_strategy(
      f=reward_function,
      population_size=30,
      sigma=0.05,
      lr=0.02,
      initial_params=params,
      num_iters=100,
      pool=pool,
    )

    # plot the rewards per iteration
    plt.plot(rewards)
    plt.show()

    # save params
    model.set_params(best_params)
    np.savez(
      'es_dynamic_portfolio_results.npz',
      train=rewards,
      **model.get_params_dict(),
    )

  # play with saved model / test episode
  print("Test Sortino/Sharpe Ratio:", reward_function(best_params, plot=True))

# %%

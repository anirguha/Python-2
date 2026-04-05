# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime
from functools import partial
from typing import Tuple, Callable
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType

# %%
start = datetime(2010, 1, 1)
end = datetime(2024, 12, 31)

tickers = ['VNQ', 'SPY', 'GLD', 'BTC-USD']
benchmark_weights = np.array([0.1, 0.8, 0.05, 0.05])
try:
    downloaded_data = yf.download(tickers, start=start, end=end, interval='1mo', progress=False)
    data = downloaded_data['Close'] if downloaded_data is not None else None
except Exception as e:
    print(f"An error occurred while downloading data: {e}")
    data = None
    
if data is not None:
    data = data.dropna()
    data.head()
# %%
return_tickers = []
for ticker in tickers:
    rtick = f'{ticker}_return'
    if data is not None:
        data[rtick] = data[ticker].pct_change()
    return_tickers.append(rtick)

# %%
# Split into train and test
train_data = None
test_data = None
if data is not None:
    train_split = int(len(data) * 0.8)
    train_data = data.iloc[:train_split]
    test_data = data.iloc[train_split:]

# %%
# Pre-compute returns for train and test
train_returns = None
test_returns = None
if data is not None and train_data is not None and test_data is not None:
    train_returns = train_data[return_tickers].dropna().to_numpy()
    test_returns = test_data[return_tickers].dropna().to_numpy()
# %%
# Add a column of zeros for the cash position
if train_data is not None and test_data is not None:
    train_data['Cash'] = 0.0
    test_data['Cash'] = 0.0
    
# %%
# Evolutionary Strategy
def evolution_strategy(
    f: Callable[[np.ndarray], float],
    pop_size: int,
    sigma: float,
    learning_rate: float,
    initial_weights: np.ndarray,
    num_iterations: int,
    pool: PoolType
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evolutionary strategy for optimization.
    
    Args:
        f: Objective function to optimize
        pop_size: Population size
        sigma: Standard deviation for mutations
        learning_rate: Learning rate for updates
        initial_weights: Initial solution
        num_iterations: Number of iterations
        pool: Multiprocessing pool for parallel evaluation
        
    Returns:
        Best weights found by the algorithm
    """
    weights = initial_weights.copy()
    reward_per_iteration = np.zeros(num_iterations)
    num_parameters = len(initial_weights)
    
    for i in range(num_iterations):
        t0 = datetime.now()
        # Generate random noise
        N = np.random.randn(pop_size, num_parameters)
        
        R = pool.map(f, [weights + sigma * N[j] for j in range(pop_size)])
        R = np.array(R)
        
        m = R.mean()
        s = R.std()
        if s == 0:
            print("Standard deviation is zero, skipping update.")
            continue
        A = (R - m) / s
        reward_per_iteration[i] = m
        weights += learning_rate / (pop_size * sigma) * np.dot(N.T, A)

        print(f"Interation {i+1}/{num_iterations}, Reward: {m:.4f}, \
            Time: {(datetime.now() - t0).total_seconds():.2f} seconds")
        
    return weights, reward_per_iteration
# %%
# Create softmax function for portfolio weights
def softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes the softmax of the input array x, returning an array of the
    same shape where each element represents the normalized exponential
    value, useful for probability distributions.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# %%
# Function for Sortini ratio calculation
def sortino_ratio(returns: np.ndarray, target: float = 0.0) -> float:
    """
    Calculates the Sortino ratio for a series of financial returns in returns,
    measuring risk-adjusted performance relative to a target return. The
    ratio penalizes only downside volatility. Returns a float value.
    """
    downside_returns = returns[returns < target]
    downside_returns_deviation = np.sqrt(np.mean((downside_returns - target) ** 2)) if len(downside_returns) > 0 else 0.0
    return (returns.mean() - target) / downside_returns_deviation if downside_returns_deviation > 0 else 1e-8

# %%
# Calculate reward function for portfolio optimization
def reward_function(weights: np.ndarray, returns: np.ndarray | None = train_returns, plot: bool = False, mode: str = "train") -> float:
    """
    Calculates the Sortino ratio for a portfolio given asset weights and
    historical returns, optionally plotting cumulative gross returns against
    a benchmark.
    """
    if returns is None:
        raise ValueError("returns cannot be None")

    weights = softmax(weights)
    portfolio_returns = np.dot(returns, weights)
    
    if plot:
        cumulative_gross_returns = np.cumprod(portfolio_returns + 1)
        
        # Benchamark cumulative returns
        benchmark_portfolio_returns = np.dot(returns, benchmark_weights)
        benchmark_cumulative_gross_returns = np.cumprod(benchmark_portfolio_returns + 1)
        
        plt.figure(figsize=(12, 6),layout='constrained')
        plt.plot(cumulative_gross_returns, label='Optimized Portfolio')
        plt.plot(benchmark_cumulative_gross_returns, label='Benchmark Portfolio', linestyle='--')
        plt.title(f'Cumulative Gross Returns for {mode.capitalize()} Data', fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Gross Return')
        plt.legend()
        plt.show()
    
    return sortino_ratio(portfolio_returns)
        
        

# %%
# Main function
if __name__ == "__main__":
    
    if train_returns is not None:
        # Create multiprocessing pool
        with Pool(8) as pool:
            # Run evolutionary strategy
            initial_weights = np.random.rand(len(tickers))
            objective = partial(reward_function, returns=train_returns)
            best_weights, rewards = evolution_strategy(
                f=objective,
                pop_size=50,
                sigma=0.1,
                learning_rate=0.01,
                initial_weights=initial_weights,
                num_iterations=100,
                pool=pool
            )
    else:
        print("Error: train_returns is None. Data download or processing failed.")
        
    # Plot rewards over iterations
    if 'rewards' in locals():
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title('Reward per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Reward (Sortino Ratio)')
        plt.grid()
        plt.show()
    
    if 'best_weights' in locals():
        print("Best weights:")
        best_weights = softmax(best_weights)
        for ticker, weight in zip(tickers, best_weights):
            print(f"{ticker}: {weight:.4f}")
        
    # Plot Sortino ratio for the best portfolio for train and test data and compare with benchmark
    if train_returns is not None:
        best_portfolio_sortino_train = reward_function(best_weights, returns=train_returns, plot=True, mode="train") # type: ignore
        print(f"Best Portfolio Sortino Ratio in Training: {best_portfolio_sortino_train:.4f}")
        
        # Plot cumulative returns for the best portfolio
        best_portfolio_sortino_test = reward_function(best_weights, returns=test_returns, plot=True, mode="test") # type: ignore
        print(f"Best Portfolio Sortino Ratio in Testing: {best_portfolio_sortino_test:.4f}") # type: ignore
        
        # Compare with benchmark
        benchmark_sortino_train = reward_function(benchmark_weights, returns=train_returns, plot=False, mode="train")
        benchmark_sortino_test = reward_function(benchmark_weights, returns=test_returns, plot=False, mode="test")
        print(f"Benchmark Sortino Ratio in Training: {benchmark_sortino_train:.4f}")
        print(f"Benchmark Sortino Ratio in Testing: {benchmark_sortino_test:.4f}")
        
    # Plot benchmark cumulative returns
    if test_returns is not None:
        portfolio_benchmark_returns = np.dot(test_returns, benchmark_weights)
        sortino_ratio_benchmark_test = sortino_ratio(portfolio_benchmark_returns)
        print(f"Benchmark Sortino Ratio in Testing: {sortino_ratio_benchmark_test:.4f}")



# %%

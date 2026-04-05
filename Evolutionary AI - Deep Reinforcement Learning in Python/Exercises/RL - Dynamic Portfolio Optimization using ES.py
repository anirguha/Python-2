# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import talib
from datetime import datetime
from functools import partial
from typing import Tuple, Callable
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
# %%
# Set the period for data retrieval
start_date = '2010-01-01'
end_date = '2024-12-31'
# Define the list of stock tickers
tickers = ['VNQ', 'SPY', 'GLD', 'BTC-USD']
benchmark_weights = np.array([0.1, 0.8, 0.05, 0.05])
# Function to download and preprocess data
try:
    downloaded_data = yf.download(tickers, start=start_date, end=end_date, interval='1mo', progress=False)
    data = downloaded_data['Close'] if downloaded_data is not None else None
except Exception as e:
    print(f"An error occurred while downloading data: {e}")
    data = None
    
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
for ticker in tickers:
    data[f'ticker_{ticker}_rsi'] = talib.RSI(data[ticker], timeperiod=14)/50 - 1 # To normalize between -1 and 1
    macd, macdsignal, macdhist = talib.MACD(data[ticker], fastperiod=12, slowperiod=26, signalperiod=9)
    data[f'ticker_{ticker}_macd'] = macd / data[ticker] * 10 # Normalize by price to get a percentage
    data[f'ticker_{ticker}_macdsignal'] = macdsignal / data[ticker] * 10 # Normalize by price to get a percentage
    data[f'ticker_{ticker}_macdhist'] = macdhist
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
        
        # resets the environment to the initial state
        def reset(self):
            self.pos = lag
            return self.state[:self.pos].flatten()
        
        def step(self, action):
            
            # Validate the dimensions of the action space
            if action.shape[-1] != self.prices.shape[-1]:
                raise ValueError(f"Action shape {action.shape} does not match the number of assets {self.prices.shape[-1]}")
            
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

# %%

# %%

# %%

# %%

# %%

# %%

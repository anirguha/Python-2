"""
Trading Linear Q-Learning

This module implements a Linear Q-Learning-based trading agent for multi-stock
portfolio management. The agent uses a linear model to approximate Q-values
for trading actions across multiple stocks.

Action encoding per stock:
    0 = sell
    1 = hold
    2 = buy

For n stocks, there are 3 ** n possible combined actions.
"""
from __future__ import annotations

import argparse
import itertools
import pickle
from pathlib import Path
from typing import List, Protocol, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# ---------------------------
# Experiment Constants
# ---------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
DEFAULT_DATA_FILE: Path = SCRIPT_DIR / "data" / "aapl_msi_sbux.csv"
MODEL_DIR: Path = SCRIPT_DIR / "model"
MODEL_FILE: Path = MODEL_DIR / "trading_agent.npz"
SCALER_FILE: Path = MODEL_DIR / "trading_scaler.pkl"

START_INVESTMENT: float = 100_000.0
LEARNING_RATE: float = 0.001
DISCOUNT_FACTOR: float = 0.95
EPSILON: float = 0.1
EPSILON_DECAY: float = 0.999999
EPSILON_MIN: float = 0.01
GRADIENT_CLIP_NORM: float = 10.0
MOMENTUM: float = 0.9
RANDOM_SEED: int = 123
TRADE_FRACTION: float = 0.25 # Percentages of the stock that can be sold/ cash that can be invested in one step
# ---------------------------
# Type Aliases and Protocols
# ---------------------------
Data = np.ndarray
Observation = np.ndarray
Action = int


class Environment(Protocol):
    """
    Protocol defining the interface for reinforcement learning environments.

    This protocol establishes a standard contract for environments used in
    reinforcement learning scenarios. It defines the essential attributes and
    methods that any conforming environment implementation must provide,
    ensuring compatibility with RL agents and training loops. The protocol
    specifies how environments should handle state initialization, action
    execution, and step-by-step interaction dynamics.

    Attributes:
        n_step: The current step number or total number of steps in the
            environment's timeline.
        action_space: A numpy array representing the available actions in the
            environment.
    """

    n_step: int
    action_space: np.ndarray

    def reset_env(self) -> np.ndarray:
        """
        Resets the environment to its initial state and returns the initial
        observation.

        These methods reinitialize the environment to prepare it for a new episode
        or simulation run. It clears any previous state information and generates
        a fresh starting observation that represents the beginning state of the
        environment.

        Returns:
            np.ndarray: The initial observation of the environment after reset,
                representing the starting state for a new episode.
        """
        ...

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Executes one time step within the environment using the given action.

        Takes an action as input and advances the environment by one time step,
        returning the resulting observation, reward, done flag, and additional
        information. This is the primary method for interacting with the
        environment and progressing through episodes.

        Args:
            action: The action to be executed in the environment.

        Returns:
            A tuple containing four elements:
                - observation: The agent's observation of the current environment
                  state after taking the action
                - reward: The amount of reward returned as a result of taking the
                  action
                - done: Boolean flag indicating whether the episode has ended
                - info: Dictionary containing auxiliary diagnostic information for
                  debugging or logging purposes
        """
        ...


# ---------------------------
# Get and Load Data
# ---------------------------
def get_data(data_file: str | Path) -> Data:
    """
    Loads and processes stock price data from a CSV file into a numeric array.

    Reads a CSV file containing stock price data, validates that it contains
    numeric columns, checks for missing values, and converts the numeric data
    to a NumPy array. The function performs several validation steps, including
    file existence verification, numeric column detection, and missing value
    checks to ensure data quality before conversion.

    Args:
        data_file: Path to the CSV file containing stock price data. Can be
            provided as a string or Path object. Supports tilde expansion for
            home directory paths.

    Returns:
        numpy.ndarray: A 2D NumPy array of float64 values containing only the
            numeric columns from the CSV file.

    Raises:
        FileNotFoundError: If the specified data file does not exist at the
            resolved path.
        ValueError: If no numeric columns are found in the CSV file.
        ValueError: If any missing (NaN) values are detected in the numeric
            columns of the CSV file.
    """
    data_path: Path = Path(data_file).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = cast(pd.DataFrame, pd.read_csv(data_path)) # type: ignore
    print(df.head())

    numeric_df: pd.DataFrame = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("No numeric stock price columns found in the CSV file.")

    if numeric_df.isna().any().any():
        raise ValueError("CSV contains missing numeric values. Please clean or fill them first.")

    return numeric_df.to_numpy(dtype=np.float64)


# ---------------------------
# Scale Data
# ---------------------------
def scale_data(env: Environment, seed: int = RANDOM_SEED) -> StandardScaler:
    """
    Scales environment state data by collecting random samples and fitting a
    standard scaler.

    This function generates a sequence of environment states by taking random
    actions, then fits a StandardScaler to normalize these states. The collected
    states are used to compute the mean and standard deviation for future state
    normalization. The environment is reset after data collection is complete.

    Args:
        env: The environment from which to collect state samples for scaling.
        seed: Random seed for reproducible action selection during data
            collection.

    Returns:
        StandardScaler: A fitted scaler object that can be used to normalize
            environment states based on the collected samples.

    Raises:
        ValueError: If env.n_step is less than or equal to zero.
    """
    if env.n_step <= 0:
        raise ValueError(f"n_step must be positive, got {env.n_step}")

    rng = np.random.default_rng(seed)
    states: List[np.ndarray] = [env.reset_env()]

    for _ in range(env.n_step - 1):
        action: int = int(rng.choice(env.action_space))
        next_state, _, done, _ = env.step(action)
        states.append(next_state)

        if done:
            break

    scaler = StandardScaler()
    scaler.fit(np.asarray(states, dtype=np.float64))

    env.reset_env()
    return scaler


# ---------------------------
# Save / Load Scaler
# ---------------------------
def save_scaler(scaler: StandardScaler, filepath: str | Path = SCALER_FILE) -> None:
    """
    Saves a StandardScaler object to a file using pickle serialization.

    This function persists a fitted StandardScaler object to disk by serializing
    it with pickle. It ensures the target directory exists before saving and
    handles path expansion and resolution automatically.

    Args:
        scaler: The fitted StandardScaler object to be saved.
        filepath: The destination path where the scaler will be saved. Can be
            either a string or Path object. Defaults to SCALER_FILE constant.

    Raises:
        OSError: If the file cannot be created or written to.
        PickleError: If the scaler object cannot be serialized.
    """
    scaler_path: Path = Path(filepath).expanduser().resolve()
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)


def load_scaler(filepath: str | Path = SCALER_FILE) -> StandardScaler:
    """
    Loads a previously saved StandardScaler object from a pickle file.

    This function deserializes a StandardScaler instance saved to disk,
    allowing it to be reused for transforming data with the same scaling
    parameters that were fitted during training. The function handles path
    expansion and resolution and validates that the file exists before
    attempting to load it.

    Args:
        filepath: Path to the pickle file containing the serialized
            StandardScaler object. Can be either a string or Path object.
            Defaults to SCALER_FILE constant.

    Returns:
        StandardScaler: The deserialized StandardScaler object that can be used
            for data transformation.

    Raises:
        FileNotFoundError: If the specified scaler file does not exist at the
            given path. The error message includes the resolved path and
            suggests running with --mode train first.
    """
    scaler_path: Path = Path(filepath).expanduser().resolve()

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {scaler_path}. Run with --mode train first."
        )

    with scaler_path.open("rb") as f:
        scaler: StandardScaler = pickle.load(f)

    return scaler


# ---------------------------
# Linear Model for Q-Learning
# ---------------------------
class LinearModel:
    """
    A simple linear model for Q-value approximation in reinforcement learning.

    This class implements a linear function approximator that maps state features
    to action values (Q-values). It uses stochastic gradient descent with momentum
    for optimization and provides methods for prediction, training, and model
    persistence. The model is designed for discrete action spaces and uses
    mean squared error as the loss function.

    Attributes:
        W (np.ndarray): Weight matrix of shape (n_dim, n_action) mapping state
            features to Q-values for each action.
        b (np.ndarray): Bias vector of shape (n_action,) for each action.
        velocity_W (np.ndarray): Momentum buffer for weight updates, same shape
            as W.
        velocity_b (np.ndarray): Momentum buffer for bias updates, same shape
            as b.
        losses (List[float]): History of mean squared error losses from training
            iterations.
    """

    def __init__(self, n_dim: int, n_action: int):
        """
        Initializes a linear model for action selection with random weights and
        zero biases.

        Creates a linear transformation model that maps state dimensions to action
        values. The weight matrix is initialized using random values scaled by
        the square root of the input dimension (He initialization variant), while
        biases are initialized to zero. Velocity terms for momentum-based
        optimization are also initialized to zero, and a list to track training
        losses is created.

        Args:
            n_dim: The dimensionality of the input state space. Must be a
                positive integer representing the number of features in the
                state representation.
            n_action: The number of possible actions. Must be a positive integer
                representing the size of the action space.

        Raises:
            ValueError: If n_dim is less than or equal to 0.
            ValueError: If n_action is less than or equal to 0.
        """
        if n_dim <= 0:
            raise ValueError(f"n_dim must be positive, got {n_dim}")
        if n_action <= 0:
            raise ValueError(f"n_action must be positive, got {n_action}")

        self.W: np.ndarray = np.random.randn(n_dim, n_action) / np.sqrt(n_dim)
        self.b: np.ndarray = np.zeros(n_action)

        self.velocity_W: np.ndarray = np.zeros_like(self.W)
        self.velocity_b: np.ndarray = np.zeros_like(self.b)

        self.losses: List[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions using the trained linear model.

        This method applies the learned linear transformation to input data by
        computing the matrix multiplication of the input features with the weight
        matrix and adding the bias term. The input is automatically converted to
        a numpy array with float64 dtype for numerical stability.

        Args:
            X: Input feature matrix where each row represents a sample and each
                column represents a feature.

        Returns:
            np.ndarray: Predicted values as a 2-D array where each row contains
                predictions for the corresponding input sample.

        Raises:
            ValueError: If the input array X is not 2-dimensional.
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim}D")

        return X @ self.W + self.b

    def sgd_optimize_params(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        learning_rate: float = LEARNING_RATE,
        momentum: float = MOMENTUM,
        gradient_clip_norm: float = GRADIENT_CLIP_NORM,
    ) -> None:
        """
        Performs a single stochastic gradient descent optimization step to update
        model parameters using momentum and gradient clipping. The method computes
        predictions, calculates the mean squared error loss, derives gradients with
        respect to weights and biases, applies gradient clipping if necessary,
        updates velocities using momentum, and finally updates the model parameters.
        The computed loss is appended to the internal loss history for tracking.

        Args:
            X: Input feature matrix where each row represents a sample and each
                column represents a feature.
            Y: Target output matrix where each row corresponds to the target values
                for the corresponding input sample.
            learning_rate: Step size multiplier for gradient descent updates,
                controlling how much to adjust parameters in the direction opposite
                to the gradient. Defaults to LEARNING_RATE.
            momentum: Coefficient for momentum term that helps speeds up gradient
                descent in relevant directions and dampens oscillations. Defaults
                to MOMENTUM.
            gradient_clip_norm: Maximum allowed L2 norm for gradients before
                scaling is applied to prevent exploding gradients. Defaults to
                GRADIENT_CLIP_NORM.

        Raises:
            ValueError: If X is not 2-dimensional.
            ValueError: If Y is not 2-dimensional.
            ValueError: If X and Y have different batch sizes (number of rows).
            ValueError: If the prediction shape does not match the target shape.
            FloatingPointError: If non-finite values are detected in predictions
                or targets before the model update.
            FloatingPointError: If model parameters become non-finite during
                optimization.
            FloatingPointError: If model loss becomes non-finite during
                optimization.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim}D")
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D, got {Y.ndim}D")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have same batch size, got {X.shape[0]} and {Y.shape[0]}")

        Y_pred: np.ndarray = self.predict(X)

        if Y_pred.shape != Y.shape:
            raise ValueError(f"Prediction shape {Y_pred.shape} does not match target shape {Y.shape}")

        if not np.all(np.isfinite(Y_pred)) or not np.all(np.isfinite(Y)):
            raise FloatingPointError("Non-finite Q-learning values detected before model update.")

        num_values: int = int(np.prod(Y.shape))
        error: np.ndarray = Y_pred - Y

        grad_W: np.ndarray = 2 * X.T @ error / num_values
        grad_b: np.ndarray = 2 * np.sum(error, axis=0) / num_values

        grad_norm: float = float(np.sqrt(np.sum(grad_W**2) + np.sum(grad_b**2)))

        if grad_norm > gradient_clip_norm:
            scale: float = gradient_clip_norm / (grad_norm + 1e-8)
            grad_W *= scale
            grad_b *= scale

        self.velocity_W = momentum * self.velocity_W - learning_rate * grad_W
        self.velocity_b = momentum * self.velocity_b - learning_rate * grad_b

        self.W += self.velocity_W
        self.b += self.velocity_b

        if not np.all(np.isfinite(self.W)) or not np.all(np.isfinite(self.b)):
            raise FloatingPointError("Model parameters became non-finite during optimization.")

        mse: float = float(np.mean(error**2))

        if not np.isfinite(mse):
            raise FloatingPointError("Model loss became non-finite during optimization.")

        self.losses.append(mse)

    def save_model(self, filepath: str | Path) -> None:
        """
        Saves the model parameters to a file in NumPy compressed format.

        This method persists in the current state of the model by saving the weight
        matrix and bias vector to disk. The parent directories of the specified
        file path are created automatically if they do not exist. The model
        parameters are stored in NumPy's compressed .npz format, which allows
        efficient storage and retrieval of multiple arrays.

        Args:
            filepath: The destination path where the model should be saved. Can be
                provided as either a string or a Path object. The path is expanded
                to resolve user home directory symbols and converted to an absolute
                path.

        Returns:
            None
        """
        model_path: Path = Path(filepath).expanduser().resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(model_path, W=self.W, b=self.b)
        print(f"Model saved to {model_path}")

    def load_model(self, filepath: str | Path) -> None:
        """
        Loads a previously saved model from a file and initializes velocity
        parameters for momentum-based optimization.

        This method loads model parameters (weights and biases) from a NumPy
        archive file and restores them to the current model instance. After
        loading, it initializes velocity parameters to zero arrays matching
        the shapes of the loaded parameters, which are used for momentum-based
        optimization algorithms. The method resolves relative paths and expands
        user home directory references in the file path.

        Args:
            filepath: Path to the model file to load. Can be either a string or
                Path object. Supports user home directory expansion (e.g., "~/")
                and relative paths.

        Raises:
            FileNotFoundError: If the specified model file does not exist at the
                given path.
        """
        model_path: Path = Path(filepath).expanduser().resolve()

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded_data = np.load(model_path)

        self.W = loaded_data["W"]
        self.b = loaded_data["b"]

        self.velocity_W = np.zeros_like(self.W)
        self.velocity_b = np.zeros_like(self.b)

        print(f"Model loaded from {model_path}")


# ---------------------------
# Environment Class for Trading
# ---------------------------
class TradingEnvironment:
    """
    Trading environment for simulating stock portfolio management with discrete actions.

    This class implements a reinforcement learning environment for trading multiple stocks
    simultaneously. It manages a portfolio consisting of cash and stock holdings, processes
    trading actions, and calculates rewards based on portfolio value changes. The environment
    uses historical price data and supports discrete actions for each stock (sell, hold, or buy).
    Trading episodes proceed through timesteps until all historical data is exhausted.

    Attributes:
        data (Data): Two-dimensional array of historical stock prices with shape
            (n_timesteps, n_stocks).
        n_step (int): Total number of timesteps available in the historical data.
        num_stocks (int): Number of different stocks in the portfolio.
        start_investment (float): An initial cash balance is available at the start of
            each episode.
        cur_step (int): Current timestep index in the trading episode.
        stock_owned (np.ndarray): Array of share quantities currently owned for
            each stock.
        stock_price (np.ndarray): Array of current prices for each stock at the
            current timestep.
        cash_in_hand (float): Current available cash balance.
        action_space (np.ndarray): Array of valid action indices.
        action_list (List[List[int]]): List of all possible action vectors, where
            each vector contains actions for all stocks (0=sell, 1=hold, 2=buy).
        state_dim (int): Dimension of the observation space.
    """

    def __init__(self, data: Data, start_investment: float = START_INVESTMENT):
        """
        Initializes the trading environment with market data and starting
        capital.

        Sets up the trading environment by validating and storing market data,
        initializing portfolio state (cash, stock positions), and configuring
        the action and state spaces. The environment supports multiple stocks
        and discrete actions (sell, hold, buy) for each stock.

        Args:
            data: A 2D array of stock price data where rows represent timesteps
                and columns represent different stocks. All values must be
                positive and finite.
            start_investment: The initial capital is available for trading. Must
                be a positive value.

        Raises:
            ValueError: If `data` is not 2-dimensional.
            ValueError: If `data` contains fewer than two timesteps.
            ValueError: If `data` contains fewer than one stock column.
            ValueError: If `data` contains non-finite values (NaN or infinity).
            ValueError: If any stock price is non-positive.
            ValueError: If `start_investment` is non-positive.
        """
        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 2:
            raise ValueError(f"data must be 2-D, got {data.ndim}D")
        if data.shape[0] < 2:
            raise ValueError("data must contain at least two timesteps")
        if data.shape[1] < 1:
            raise ValueError("data must contain at least one stock column")
        if not np.all(np.isfinite(data)):
            raise ValueError("data contains non-finite values")
        if np.any(data <= 0):
            raise ValueError("stock prices must be positive")
        if start_investment <= 0:
            raise ValueError("start_investment must be positive")

        self.data: Data = data
        self.n_step, self.num_stocks = data.shape
        self.start_investment: float = float(start_investment)

        self.cur_step: int = 0
        self.stock_owned: np.ndarray = np.zeros(self.num_stocks, dtype=np.float64)
        self.stock_price: np.ndarray = self.data[self.cur_step].copy()
        self.cash_in_hand: float = self.start_investment

        self.action_space: np.ndarray = np.arange(3**self.num_stocks)
        self.action_list: List[List[int]] = [
            list(x) for x in itertools.product([0, 1, 2], repeat=self.num_stocks)
        ]

        self.state_dim: int = self.num_stocks * 2 + 1
        self.reset_env()

    def reset_env(self) -> Observation:
        """
        Resets the environment to its initial state and returns the first observation.

        These methods reinitialize all environment variables to their starting values,
        including the current step counter, stock ownership portfolio, current stock
        prices, and available cash. It sets up the environment for a new episode of
        trading simulation.

        Returns:
            Observation: The initial observation of the environment after reset,
                containing the current state information needed for the agent to make
                its first decision.
        """
        self.cur_step = 0
        self.stock_owned = np.zeros(self.num_stocks, dtype=np.float64)
        self.stock_price = self.data[self.cur_step].copy()
        self.cash_in_hand = self.start_investment

        return self._get_observation()

    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        """
        Executes a single trading step in the environment with the given action.

        Advances the environment by one time step, processes the trading action,
        and returns the resulting state information. The method updates the current
        stock prices, executes the trade based on the action, calculates the reward
        based on portfolio value change, and determines if the episode has ended.
        The reward is computed as the relative change in portfolio value compared
        to the previous step.

        Args:
            action: The trading action to execute must be a valid action from
                the action space.

        Returns:
            A tuple containing four elements:
                - observation (Observation): The current state observation after
                  executing the action.
                - reward (float): The normalized reward calculated as the relative
                  change in portfolio value.
                - done (bool): Flag indicating whether the episode has terminated.
                - info (dict): Dictionary containing additional information with
                  the current portfolio value.

        Raises:
            ValueError: If the provided action is not in the valid action space.
            RuntimeError: If step() is called after the trading episode has already
                ended without calling reset_env() first.
            FloatingPointError: If the previous portfolio value is zero or negative,
                indicating an invalid portfolio state.
        """
        if action not in self.action_space:
            raise ValueError(f"{action} is not a valid action.")
        if self.cur_step >= self.n_step - 1:
            raise RuntimeError(
                "Cannot call step() after the trading episode is done. Call reset_env() first."
            )

        prev_portfolio_value: float = self.get_portfolio_value()

        if prev_portfolio_value <= 0:
            raise FloatingPointError(
                f"Invalid previous portfolio value: {prev_portfolio_value}"
            )

        self.cur_step += 1
        self.stock_price = self.data[self.cur_step].copy()

        self._trade(action)

        new_portfolio_value: float = self.get_portfolio_value()
        reward: float = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        done: bool = self.cur_step >= self.n_step - 1
        info: dict = {"portfolio_value": new_portfolio_value,
                      "stock_owned": self.stock_owned.copy()}
        observation: Observation = self._get_observation()

        return observation, reward, done, info

    def _get_observation(self) -> Observation:
        """
        Constructs and returns the current observation state of the trading environment.

        This method assembles a complete observation vector representing the current state
        of the portfolio, including stock holdings, current market prices, and available
        cash. The observation follows a fixed structure where stock ownership quantities
        occupy the first portion, followed by current stock prices, and ending with the
        available cash balance.

        Returns:
            Observation: A numpy array containing the complete state observation with
                stock ownership counts in the first segment, current stock prices in
                the middle segment, and cash in hand as the final element

        Note:
            The observation array has a fixed size determined by `state_dim`, which
            accommodates `num_stocks` ownership values, `num_stocks` price values,
            and one cash balance value.
        """
        obs: Observation = np.zeros(self.state_dim, dtype=np.float64)
        obs[: self.num_stocks] = self.stock_owned
        obs[self.num_stocks : 2 * self.num_stocks] = self.stock_price
        obs[-1] = self.cash_in_hand

        return obs

    def _trade(self, action: int) -> None:
        """
        Executes a trading action by buying, selling, or holding stocks based on
        the provided action index.

        This method processes a trading action that determines whether to sell 25%
        of holdings, hold the current portfolio, or buy stocks with 25% of available
        cash. The action is translated into a vector of individual stock actions,
        where selling operations are executed first, followed by buying operations to
        ensure sufficient cash is available for purchases.

        Args:
            action: The index representing the trading action to execute, must be
                within the valid action space.

        Raises:
            ValueError: If the provided action is not in the valid action space.
            FloatingPointError: If cash in hand becomes negative after executing
                trades, indicating a calculation error.

        The action encoding for each stock is as follows:
        0 - sell 25% of the current holdings
        1 - hold the portfolio
        2 - buy with 25% of the current cash
        """

        if action not in self.action_space:
            raise ValueError(f"{action} is not a valid action.")

        action_vec: List[int] = self.action_list[action]

        # stocks_to_sell: List[int] = [i for i, a in enumerate(action_vec) if a == 0]


        for i, a in enumerate(action_vec):
        # Sell first.
            if a == 0:
                stocks_to_sell: float = np.floor(self.stock_owned[i] * TRADE_FRACTION)
                self.cash_in_hand += self.stock_price[i] * stocks_to_sell
                self.stock_owned[i] = max(0.0, self.stock_owned[i] - stocks_to_sell)

        stocks_to_buy: List[int] = [i for i, a in enumerate(action_vec) if a == 2]

        if stocks_to_buy:
            cash_per_stock: float = self.cash_in_hand * TRADE_FRACTION / len(stocks_to_buy)

            for i in stocks_to_buy:
                shares_to_buy: int = int(cash_per_stock // self.stock_price[i])

                if shares_to_buy > 0:
                    cost: float = shares_to_buy * self.stock_price[i]
                    self.cash_in_hand -= cost
                    self.stock_owned[i] += shares_to_buy


            if self.cash_in_hand < -1e-8:
                raise FloatingPointError(f"Cash became negative: {self.cash_in_hand}")

    def get_portfolio_value(self) -> float:
        """
        Calculates the total value of the current portfolio including stocks and cash.

        This method computes the total portfolio value by summing the value of all owned
        stocks (quantity multiplied by current price) and the available cash in hand.
        The stock value is calculated as a dot product of stock quantities and their
        respective prices.

        Returns:
            float: The total portfolio value, which is the sum of all stock holdings
                valued at current prices plus the cash balance.
        """
        stock_value: float = float(np.sum(self.stock_owned * self.stock_price))
        total_value: float = stock_value + self.cash_in_hand

        return float(total_value)


# ---------------------------
# Agent Class
# ---------------------------
class TradingAgent:
    """
    A reinforcement learning agent that uses Q-learning with a linear function
    approximator for making trading decisions.

    This agent implements the Q-learning algorithm with epsilon-greedy exploration
    to learn optimal trading policies. It maintains a linear model to approximate
    Q-values for state-action pairs and updates these estimates through temporal
    difference learning. The agent gradually reduces exploration over time through
    epsilon decay, transitioning from random exploration to exploitation of learned
    policies.

    Attributes:
        state_size (int): Number of features in each environment observation.
        action_size (int): Number of discrete actions available to the agent.
        learning_rate (float): Step size for gradient descent updates to the model
            parameters.
        gamma (float): Discount factor for future rewards in Q-learning updates.
        epsilon (float): Current probability of taking random exploratory actions.
        epsilon_decay (float): A multiplicative factor is applied to epsilon after each
            training step.
        epsilon_min (float): Lower bound for epsilon to ensure minimum exploration.
        steps_done (int): Total number of training steps performed by the agent.
        model (LinearModel): Linear function approximator that maps states to
            Q-values for all actions.
    """

    def __init__(self, state_size: int, action_size: int):
        """
        Initializes the agent with specified state and action space dimensions
        and default hyperparameters.

        This constructor sets up a reinforcement learning agent by validating
        and storing the state and action space sizes, initializing learning
        hyperparameters from global constants, and creating a linear model for
        Q-value approximation. The agent maintains an epsilon-greedy exploration
        strategy with decay and tracks the number of steps taken during training.

        Args:
            state_size: The dimensionality of the state space representation.
                Must be a positive integer.
            action_size: The number of possible discrete actions the agent can
                take. Must be a positive integer.

        Raises:
            ValueError: If state_size is less than or equal to 0.
            ValueError: If action_size is less than or equal to 0.
        """
        if state_size <= 0:
            raise ValueError(f"state_size must be positive, got {state_size}")
        if action_size <= 0:
            raise ValueError(f"action_size must be positive, got {action_size}")

        self.state_size: int = state_size
        self.action_size: int = action_size
        self.learning_rate: float = LEARNING_RATE
        self.gamma: float = DISCOUNT_FACTOR
        self.epsilon: float = EPSILON
        self.epsilon_decay: float = EPSILON_DECAY
        self.epsilon_min: float = EPSILON_MIN
        self.steps_done: int = 0
        self.model: LinearModel = LinearModel(state_size, action_size)

    def choose_action(self, state: np.ndarray) -> Action:
        """
        Selects an action based on the current state using an epsilon-greedy strategy.

        This method implements an epsilon-greedy action selection policy where with
        probability epsilon a random action is chosen for exploration, and with
        probability (1-epsilon) the action with the highest predicted Q-value is
        selected for exploitation. The input state is automatically reshaped to ensure
        compatibility with the model's expected input format.

        Args:
            state: The current environment state as a numpy array. Can be either 1-D
                or 2-D array. If 1-D, it will be automatically reshaped to 2-D with
                shape (1, -1).

        Returns:
            Action: The selected action index as an integer value from the available
                action space.

        Raises:
            ValueError: If the state array has more than 2 dimensions.
        """
        state = np.asarray(state, dtype=np.float64)

        if state.ndim == 1:
            state = state.reshape(1, -1)
        if state.ndim != 2:
            raise ValueError(f"state must be 1-D or 2-D, got {state.ndim}D")

        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))

        q_values: np.ndarray = self.model.predict(state)
        return int(np.argmax(q_values[0]))

    def train(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Trains the agent using a single transition from the environment.

        This method implements the Q-learning update rule by computing the target
        Q-value based on the observed reward and the maximum Q-value of the next
        state. The model is then updated via stochastic gradient descent to
        minimize the difference between predicted and target Q-values. After each
        training step, the exploration rate (epsilon) is decayed according to the
        configured decay rate, ensuring a gradual shift from exploration to
        exploitation.

        Args:
            state: The current state observation from the environment
            action: The action taken in the current state
            reward: The reward received after taking the action
            next_state: The resulting state after taking the action
            done: Flag indicating whether the episode has terminated

        Raises:
            FloatingPointError: If the computed Q-learning target value is
                non-finite (infinite or NaN)
        """
        state = np.asarray(state, dtype=np.float64)
        next_state = np.asarray(next_state, dtype=np.float64)

        if state.ndim == 1:
            state = state.reshape(1, -1)
        if next_state.ndim == 1:
            next_state = next_state.reshape(1, -1)

        if done:
            target: float = float(reward)
        else:
            next_q_values: np.ndarray = self.model.predict(next_state)
            target: float = float(reward + self.gamma * float(np.max(next_q_values[0])))

        if not np.isfinite(target):
            raise FloatingPointError(f"Non-finite Q-learning target computed: {target}")

        target_full: np.ndarray = self.model.predict(state)
        target_full[0, action] = target

        self.model.sgd_optimize_params(
            state,
            target_full,
            learning_rate=self.learning_rate,
        )

        self.steps_done += 1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_model(self, filepath: str | Path) -> None:
        """
        Saves the model to the specified file path.

        This method persists the current state of the model to disk at the given
        location. The saved model can be loaded later for inference or continued
        training. The method delegates the actual saving operation to the
        underlying model instance.

        Args:
            filepath: The destination path where the model will be saved. Can be
                either a string representing the file path or a Path object.
        """
        self.model.save_model(filepath)

    def load_model(self, filepath: str | Path) -> TradingAgent:
        """
        Loads a previously saved model from the specified file path and returns
        the current TradingAgent instance. This method delegates the loading
        operation to the underlying model and enables the agent to resume
        trading operations with the restored model parameters and weights.

        Args:
            filepath: The file system path pointing to the saved model file,
                which can be provided as either a string or a Path object

        Returns:
            TradingAgent: The current TradingAgent instance with the loaded
                model, allowing for method chaining
        """
        self.model.load_model(filepath)
        return self


# ---------------------------
# Play Episode
# ---------------------------
def play_one_episode(
    agent: TradingAgent,
    env: TradingEnvironment,
    scaler: StandardScaler,
    is_train: bool,
) -> tuple[float, float, List[float]]:
    """
    Executes a single trading episode with the given agent and environment.

    This function runs a complete trading episode from start to finish, where the
    agent interacts with the trading environment by choosing actions based on
    observed states. During training mode, the agent learns from each step by
    updating its model. The function tracks portfolio values throughout the episode
    and calculates the average loss per step when training is enabled.

    Args:
        agent: The trading agent that selects actions and optionally trains on
            experiences.
        env: The trading environment that simulates market conditions and executes
            trades.
        scaler: The standard scaler is used to normalize state observations before
            feeding them to the agent.
        is_train: Flag indicating whether the agent should train during this
            episode. If True, the agent updates its model after each step.

    Returns:
        tuple[float, float, List[float]]: A tuple containing three elements:
            - The final portfolio value at the end of the episode
            - The average loss per step during the episode (0.0 if not training
              or if no steps were taken)
            - A list of portfolio values recorded at each step of the episode,
              starting with the initial value
    """
    raw_state: Observation = env.reset_env()
    state: np.ndarray = scaler.transform([raw_state])

    portfolio_values: List[float] = [env.get_portfolio_value()]
    portfolio_value: float = portfolio_values[0]

    done: bool = False
    num_steps: int = 0
    total_loss: float = 0.0

    while not done:
        action: Action = agent.choose_action(state)

        raw_next_state, reward, done, info = env.step(action)
        next_state: np.ndarray = scaler.transform([raw_next_state])

        if is_train:
            agent.train(state, action, reward, next_state, done)

            if agent.model.losses:
                total_loss += agent.model.losses[-1]

            num_steps += 1

        state = next_state
        portfolio_value = float(info["portfolio_value"])
        portfolio_values.append(portfolio_value)

    loss_per_episode: float = total_loss / num_steps if num_steps > 0 else 0.0

    return portfolio_value, loss_per_episode, portfolio_values


# ---------------------------
# Create Test / Train Data
# ---------------------------
def create_train_test_data(data: Data, split: float = 0.8) -> tuple[Data, Data]:
    """
    Splits data into training and test sets based on a specified ratio.

    This function divides the input data into two separate sets for training
    and testing purposes. The split is performed sequentially, with the first
    portion of data used for training and the remainder for testing. The
    function validates that both resulting sets contain at least two rows and
    that the split ratio is valid.

    Args:
        data: The input data to be split into training and test sets.
        split: The proportion of data to allocate for training, expressed as a
            decimal between 0 and 1 (exclusive). Defaults to 0.8, which
            allocates 80% of data for training and 20% for testing.

    Returns:
        tuple[Data, Data]: A tuple containing two Data objects where the first
            element is the training data and the second element is the test
            data.

    Raises:
        ValueError: If the split parameter is not between 0 and 1 (exclusive).
        ValueError: If the resulting training data contains fewer than two
            rows.
        ValueError: If the resulting test data contains fewer than two rows.
    """
    if not 0.0 < split < 1.0:
        raise ValueError(f"split must be between 0 and 1, got {split}")

    train_size: int = int(len(data) * split)

    if train_size < 2:
        raise ValueError("Training data must contain at least two rows.")
    if len(data) - train_size < 2:
        raise ValueError("Test data must contain at least two rows.")

    train_data: Data = data[:train_size]
    test_data: Data = data[train_size:]

    return train_data, test_data


# ---------------------------
# Train Agent
# ---------------------------
def train_agent(
    agent: TradingAgent,
    env: TradingEnvironment,
    scaler: StandardScaler,
    num_episodes: int = 1000,
    model_file: str | Path = MODEL_FILE,
) -> List[float]:
    """
    Trains a trading agent over multiple episodes using reinforcement learning.

    Executes the training loop for a specified number of episodes, where the agent
    interacts with the trading environment to learn optimal trading strategies. During
    training, the agent's neural network is updated based on experiences, and epsilon
    decay is applied for exploration-exploitation balance. The function displays
    real-time progress with portfolio values, average losses, and epsilon values.
    After training completes, the trained model is saved to disk.

    Args:
        agent: The trading agent to be trained, containing the policy network
            and learning parameters.
        env: The trading environment that simulates market conditions and
            executes trades.
        scaler: Standard scaler for normalizing state observations before
            feeding them to the agent's network.
        num_episodes: Number of training episodes to run. Each episode represents
            a complete trading sequence from start to end.
        model_file: The file path where the trained model weights will be saved.

    Returns:
        List[float]: A list containing the average loss per episode for all
            training episodes, useful for analyzing training convergence.

    Raises:
        ValueError: If num_episodes is less than or equal to zero.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")

    train_losses: List[float] = []

    pbar = tqdm(
        range(num_episodes),
        desc="Training Episodes",
        position=0,
        unit="episode",
        leave=True,
    )

    for _ in pbar:
        final_portfolio_value, loss_per_episode, _ = play_one_episode(
            agent,
            env,
            scaler,
            is_train=True,
        )

        train_losses.append(loss_per_episode)

        pbar.set_postfix(
            {
                "Portfolio": f"{final_portfolio_value:,.0f}",
                "Avg Loss": f"{loss_per_episode:.6f}",
                "Epsilon": f"{agent.epsilon:.4f}",
            }
        )

    agent.save_model(model_file)
    return train_losses


# ---------------------------
# Test Agent
# ---------------------------
def evaluate_agent(
    agent: TradingAgent,
    env: TradingEnvironment,
    scaler: StandardScaler,
    model_file: str | Path = MODEL_FILE,
) -> List[float]:
    """
    Evaluates a trained trading agent on the trading environment without exploration.

    This function loads a pre-trained model, runs a single episode with exploration
    disabled (epsilon set to 0), and reports comprehensive performance metrics
    including initial and final portfolio values, best and worst values during the
    episode, and total return percentage. The agent's original epsilon value is
    restored after evaluation to maintain its state. If the portfolio value remains
    unchanged, a warning note is printed indicating the greedy policy never opened
    a position.

    Args:
        agent: The trading agent instance to be evaluated with its trained policy.
        env: The trading environment instance where the agent will operate during
            evaluation.
        scaler: The fitted standard scaler is used to normalize state observations
            during evaluation, ensuring consistency with training data preprocessing.
        model_file: Path to the saved model file to load for evaluation. Can be
            either a string or Path object. Defaults to MODEL_FILE constant.

    Returns:
        List[float]: A list of portfolio values at each step throughout the
            evaluation episode, tracking the portfolio's performance over time.

    Raises:
        FileNotFoundError: If the specified model file does not exist at the given
            path, indicating training must be performed first.
    """
    model_path: Path = Path(model_file).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run with --mode train first."
        )

    agent.load_model(model_path)

    previous_epsilon: float = agent.epsilon
    agent.epsilon = 0.0

    try:
        final_portfolio_value, _, portfolio_values = play_one_episode(
            agent,
            env,
            scaler,
            is_train=False,
        )
    finally:
        agent.epsilon = previous_epsilon

    initial_value: float = portfolio_values[0]
    total_return_pct: float = (final_portfolio_value / initial_value - 1.0) * 100.0

    print("\nEvaluation Summary:")
    print(f"  Initial Portfolio Value: {initial_value:,.2f}")
    print(f"  Final Portfolio Value: {final_portfolio_value:,.2f}")
    print(f"  Best Portfolio Value: {np.max(portfolio_values):,.2f}")
    print(f"  Worst Portfolio Value: {np.min(portfolio_values):,.2f}")
    print(f"  Total Return: {total_return_pct:.2f}%")

    if np.allclose(portfolio_values, initial_value):
        print(
            "  Note: Portfolio value did not change during evaluation. "
            "The greedy policy likely never opened a position."
        )

    return portfolio_values


# ---------------------------
# Plot Losses
# ---------------------------
def plot_losses(losses: List[float], train_test_flag: str) -> None:
    """
    Plots training or testing losses over episodes with automatic scaling.

    This function visualizes loss values across episodes, filtering out
    non-finite values (NaN, infinity) and applying logarithmic scaling when
    appropriate. If all losses are non-finite, a message is printed and no
    plot is generated. The function automatically determines whether to use
    logarithmic or linear scaling based on the presence of positive loss
    values.

    Args:
        losses: A list of loss values recorded across episodes. May contain
            non-finite values which will be filtered out.
        train_test_flag: A string identifier indicating whether the losses
            are from training or testing, used in the plot title and status
            messages.

    Returns:
        None

    Raises:
        None

    Note:
        The function uses matplotlib for visualization and numpy for array
        operations. Logarithmic scaling is applied only when positive loss
        values exist. The plot dimensions are fixed at 10x6 inches.
    """
    finite_losses: np.ndarray = np.asarray(losses, dtype=float)
    finite_losses = finite_losses[np.isfinite(finite_losses)]

    if finite_losses.size == 0:
        print(f"No finite {train_test_flag.lower()} losses to plot.")
        return

    positive_losses: np.ndarray = finite_losses[finite_losses > 0]
    losses_to_plot: np.ndarray = positive_losses if positive_losses.size else finite_losses

    plt.figure(figsize=(10, 6))
    plt.plot(losses_to_plot)
    plt.xlabel("Episode")
    plt.ylabel("Loss")

    if positive_losses.size:
        plt.yscale("log")

    plt.title(f"{train_test_flag} Loss")
    plt.tight_layout()
    plt.show()


# ---------------------------
# Plot Portfolio Values
# ---------------------------
def plot_portfolio_values(
    portfolio_values: List[float],
    title: str = "Portfolio Performance",
) -> None:
    """
    Plots the portfolio performance over time with the initial investment baseline.

    This function creates a line chart visualizing how a portfolio's value changes
    over time, displaying both the actual portfolio value at each timestep and a
    horizontal line representing the initial investment amount for comparison. The
    plot includes labeled axes, a title, and a legend to distinguish between the
    portfolio value line and the initial investment baseline.

    Args:
        portfolio_values: A list of float values representing the portfolio's total
            value at each timestep in chronological order, where the first element
            is the initial portfolio value
        title: The title to display at the top of the plot, defaults to
            "Portfolio Performance"

    Returns:
        None: This function displays a matplotlib plot and does not return a value

    Notes:
        If the portfolio_values list is empty, the function prints a message and
        exits without generating a plot. The plot is displayed using matplotlib's
        interactive mode and includes automatic layout adjustment for optimal
        spacing.
    """
    if not portfolio_values:
        print("No portfolio values to plot.")
        return

    initial_value: float = portfolio_values[0]

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label="Portfolio Value")
    plt.axhline(
        y=initial_value,
        linestyle="--",
        label="Initial Investment",
    )
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------
# Parse Arguments
# ---------------------------
def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments for the trading linear Q-learning application.

    This function creates and configures an argument parser to handle command line
    inputs for a trading system that uses linear Q-learning. It defines three main
    parameters: operation mode (train or test), data file path for stock prices,
    and number of training episodes. The parser provides default values for all
    arguments and includes help text for user guidance.

    Returns:
        argparse.Namespace: A namespace object containing the parsed command line
            arguments with the following attributes: mode (str) for operation mode,
            data_file (Path) for the CSV file path, and episodes (int) for the
            number of training episodes.

    Raises:
        SystemExit: When argument parsing fails or when help is requested via
            the command line.
    """
    p = argparse.ArgumentParser(description="Trading Linear Q-Learning.")

    p.add_argument(
        "-m",
        "--mode",
        choices=("train", "test"),
        default="train",
        help='Mode of operation. Defaults to "train".',
    )

    p.add_argument(
        "-df",
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="CSV file containing stock prices.",
    )

    p.add_argument(
        "-e",
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes to run in train mode.",
    )

    return p.parse_args()


# ---------------------------
# Main Function
# ---------------------------
def main() -> None:
    """
    Main function for training and evaluating a trading agent using reinforcement
    learning.

    This function orchestrates the complete workflow for either training a new
    trading agent or evaluating an existing one. It handles data loading,
    preprocessing, environment setup, agent initialization, and execution of the
    specified mode. The function uses a fixed random seed for reproducibility and
    splits the data into training and testing sets. In training mode, it creates
    a trading environment, trains an agent, and saves the trained model along with
    data scaling parameters. In evaluation mode, it loads a pre-trained model and
    evaluates its performance on test data.

    Returns:
        None

    Raises:
        Any exceptions raised by called functions such as data loading errors,
        model training failures, or file I/O errors.

    Note:
        The function relies on command-line arguments parsed by parse_args() to
        determine the operating mode and data file location. It uses global
        constants for configuration including RANDOM_SEED, START_INVESTMENT,
        MODEL_FILE, and SCALER_FILE.
    """
    args = parse_args()

    np.random.seed(RANDOM_SEED)

    data: Data = get_data(args.data_file)
    print(f"Loaded data shape: {data.shape}")

    train_data, test_data = create_train_test_data(data)

    if args.mode == "train":
        train_env = TradingEnvironment(
            train_data,
            start_investment=START_INVESTMENT,
        )

        agent = TradingAgent(
            state_size=train_env.state_dim,
            action_size=train_env.action_space.size,
        )

        train_scaler = scale_data(train_env)
        save_scaler(train_scaler, SCALER_FILE)

        train_losses = train_agent(
            agent,
            train_env,
            train_scaler,
            num_episodes=args.episodes,
            model_file=MODEL_FILE,
        )

        plot_losses(train_losses, "Training")

    else:
        test_env = TradingEnvironment(
            test_data,
            start_investment=START_INVESTMENT,
        )

        agent = TradingAgent(
            state_size=test_env.state_dim,
            action_size=test_env.action_space.size,
        )

        test_scaler = load_scaler(SCALER_FILE)

        test_portfolio_values = evaluate_agent(
            agent,
            test_env,
            test_scaler,
            model_file=MODEL_FILE,
        )

        plot_portfolio_values(test_portfolio_values, "Testing Portfolio Value")


# ---------------------------
# Main Module
# ---------------------------
if __name__ == "__main__":
    main()

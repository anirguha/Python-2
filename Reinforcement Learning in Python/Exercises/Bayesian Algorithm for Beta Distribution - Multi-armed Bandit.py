from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from copy import deepcopy

from numpy import dtype, float64, ndarray

# Define Hyperparameters
NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

# Define Bandit Class
class Bandit:
    """
    Represents a bandit used in the context of reinforcement learning and decision-making
    problems.

    This class models a bandit with a Beta distribution to represent its underlying
    probability distribution. It supports actions such as pulling the bandit, updating
    its parameter estimates, and sampling from its current belief. The initial parameters
    can be customized to fit specific problem setups.

    Attributes:
        p (float): The true probability of success for the bandit.
        Mean (float): The estimated mean of the bandit based on prior information.
        Alpha (float): The alpha parameter (shape) of the bandit's Beta distribution.
        Beta (float): The beta parameter (shape) of the bandit's Beta distribution.
        N (int): The number of times the bandit has been pulled.
    """
    def __init__(self,
                 p: float,
                 initial_mean: float = 0.0,
                 initial_alpha: float = 1.0,
                 initial_beta: float = 1.0):
        """
        Initialize a Bandit instance with specified parameters.

        Args:
            p (float): The true probability of success for the bandit.
            Initial_mean (float): Initial estimated mean of the bandit.
            Initial_alpha (float): Initial alpha parameter for the Beta distribution.
            Initial_beta (float): Initial beta parameter for the Beta distribution.
        """
        self.p = p
        self.mean = initial_mean
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.N = 0

    def pull(self) -> bool:
        """Simulates the action of pulling a Bernoulli arm in a multi-armed bandit problem."""
        return np.random.random() < self.p

    def update(self, x: float) -> None:
        """Updates the parameter estimates of the bandit based on new information."""
        self.alpha += x
        self.beta += 1 - x
        self.N += 1

        return None

    def sample(self) -> float | ndarray[tuple[Any, ...], dtype[float64]]:
        """Draws a sample from the bandit's current Beta distribution, which represents the belief about the probability of success.
        """
        return np.random.beta(self.alpha, self.beta)

# Function to plot the bandit distributions
def plot_distributions(bandit_lst: list[Bandit], axes: plt.Axes) -> None:
    """
    Plots the probability distribution of each bandit's success rate based on their
    current alpha and beta parameters. Each bandit is represented by a line plot
    showing its beta distribution.

    Args:
        axes: plt.axes object to plot on.
        bandit_lst (list[Bandit]): A list of Bandit objects, each providing alpha,
            beta, p, and N attributes for distribution computation and labeling.

    Returns:
        None
    """
    x = np.linspace(0, 1, 200)
    for b in bandit_lst:
        y = stats.beta.pdf(x, b.alpha, b.beta)
        wins = int(b.alpha - 1)
        axes.plot(x, y, label=f"p={b.p:.2f}, wins={wins}/{b.N}")

    return None

def run_experiment(sample_point_lst: list[int] | None = None) -> tuple[list[Bandit], dict[int, list[Bandit]]]:
    """
    Executes a multi-armed bandit experiment, simulating the iterative process of selecting bandits,
    pulling arms, and updating beliefs based on observed rewards. Captures and returns detailed
    snapshots of the bandit states at specified trial numbers.

    Args:
        sample_point_lst (list[int] | None, optional): A list of trial numbers at which to capture snapshots
            of the bandit states. If None, no snapshots are captured. Defaults to None.

    Returns:
        tuple[list[Bandit], dict[int, list[Bandit]]]: A tuple containing the final state of all bandits
            and a dictionary mapping trial numbers (as specified in `sample_points`) to the states of
            the bandits at those trials.
    """
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    sample_point_lst = sorted(sample_point_lst or [])
    snaps: dict[int, list[Bandit]] = {}

    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])

        x = bandits[j].pull()

        rewards[i] = x

        bandits[j].update(x)

        trial_num = i + 1
        if trial_num in sample_point_lst:
            snaps[trial_num] = deepcopy(bandits)

    # Print the average reward for each bandit
    print(f"Total reward for all bandits together: {rewards.sum()}")
    print(f"Overall win rate: {rewards.mean()}")
    selection_percentages = [f"{b.N / NUM_TRIALS * 100:.2f}%" for b in bandits]
    print(f"Number of times each bandit was selected: {', '.join(selection_percentages)}")

    return bandits, snaps

if __name__ == "__main__":
    """
    Run the multi-armed bandit experiment with Bayesian algorithm and visualize the results.
    """
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]

    _, snapshots = run_experiment(sample_points)

    # Plot
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.ravel()

    for ax, trial in zip(axs, sample_points):
        plot_distributions(snapshots[trial], ax)
        ax.set_title(f"After {trial} trials")
        ax.legend(fontsize="small")

    fig.tight_layout(pad=2.0)

    plt.show()

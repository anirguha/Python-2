import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from copy import deepcopy

# Set up the parameters
NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]
# np.random.seed(1)

# Create a Bandit class
class Bandit:
    """
    Represents a bandit in a multi-armed bandit problem.

    The Bandit class models a single arm of a multi-armed bandit problem. It tracks
    the true mean of rewards for the arm and maintains a Bayesian posterior
    distribution of the mean. The class supports pulling the bandit arm, updating
    the posterior based on observed rewards, and sampling from the posterior
    distribution to predict the mean reward.

    Attributes:
        true_mean (float): The true mean reward of the bandit arm.
        prior_mean (float): The mean of the prior distribution over the reward mean.
        N (int): The number of observations made from the bandit arm.
        tau_ (float): The precision (inverse variance) associated with the prior.
        lambda_ (float): The precision (inverse variance) used for updating the
            posterior distribution.
    """
    def __init__(self, true_mean):
        self.true_mean = true_mean

        # Parameters for the prior distribution of the mean (mu)
        self.prior_mean = 0
        self.N = 0
        self.tau_ = 1
        self.lambda_ = 1

    def pull(self) -> float | np.ndarray:
        """
        Pull the bandit arm and update the posterior distribution of the mean.
        """
        return np.random.randn() / np.sqrt(self.tau_) + self.true_mean

    def update(self, x) -> None:
        """
        Update the posterior distribution of the mean based on the observed reward.
        """
        self.N += 1
        self.lambda_ += self.tau_
        self.prior_mean = (self.lambda_ * self.prior_mean + self.N * x) / (self.lambda_ + self.N)

        return None

    def sample(self) -> float | np.ndarray:
        """
        Sample a mean from the current posterior distribution.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.prior_mean

    def __repr__(self) -> str:
        """
        Provides a string representation of the Bandit object including details about its
        true mean, sample count, and prior distribution parameters.

        Returns:
            str: A string representation of the Bandit object, showing the true mean,
            the number of samples observed, the prior mean, the scaling factor for the
            prior variance, and the standard deviation of the sampling distribution.
        """
        return f"Bandit(true_mean={self.true_mean}, N={self.N}, prior_mean={self.prior_mean}, lambda_={self.lambda_}, tau_={self.tau_})"

def plot_distribution(bandit_lst: list[Bandit], axes: plt.Axes) -> None:
    """
    Visualizes the probability distributions of bandits' prior beliefs after a certain number
    of trials, superimposing the plots on the provided axes.

    This function iterates through a list of Bandit objects and plots the probability density
    functions (PDF) of their priors. The priors are modeled as normal distributions based on
    each bandit's mean and inverse variance (lambda_). This visualization helps to assess how
    each bandit’s belief about its mean reward changes over time, based on the number of plays
    and trials.

    Args:
        bandit_lst (list[Bandit]): A list of Bandit objects, each containing the parameters for
            its prior distribution and tracking its true mean and play count.
        axes (plt.Axes): Matplotlib axes object where the distribution plots will be drawn.

    Returns:
        None
    """
    x = np.linspace(-3, 6, 200)

    for b in bandit_lst:
        y = norm.pdf(x, b.prior_mean, np.sqrt(1. / b.lambda_))
        axes.plot(x, y, label=f"real mean: {b.true_mean:.2f}, num plays: {b.N}")

def run_experiment(sample_point_lst: list[int] | None = None) -> tuple[list[Bandit], dict[int, list[Bandit]]]:
    """
    Runs a bandit experiment based on the provided sample points.

    This function simulates a bandit experiment by optionally using a list of
    sample points to configure the experiment. It returns the results including
    a list of bandit objects used in the experiment and a dictionary mapping
    integer keys to lists of bandit objects.

    Args:
        sample_point_lst (list[int] or None): A list of sample points to configure
            the experiment. If None, the function will use a default configuration.

    Returns:
        tuple[list[Bandit], dict[int, list[Bandit]]]: A tuple containing a list of
        bandit objects used in the experiment and a dictionary where the keys are
        integers and the values are lists of associated bandit objects.
    """
    bandits = [Bandit(m) for m in BANDIT_MEANS]
    sample_point_lst = sorted(sample_point_lst or [])
    snaps: dict[int, list[Bandit]] = {}

    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])
        rewards[i] = bandits[j].pull()
        bandits[j].update(rewards[i])

        if i in sample_point_lst:
            # Freeze per-trial posterior state for plotting later.
            snaps[i] = deepcopy(bandits)

    cumulative_average = np.cumsum(rewards) / (np.arange(i + 1) + 1)
    plt.plot(cumulative_average, label="Cumulative Average")
    for m in BANDIT_MEANS:
        plt.plot(np.ones(NUM_TRIALS) * m, label=f"true mean: {m:.3f}")
    plt.show()

    # Print the average reward for each bandit
    print(f"Total reward for all bandits together: {rewards.sum():.2f}")
    print(f"Overall win rate: {rewards.mean():.2f}")
    selection_percentages = [f"{b.N / NUM_TRIALS * 100:.2f}%" for b in bandits]
    print(f"Number of times each bandit was selected: {', '.join(selection_percentages)}")

    return bandits, snaps

if __name__ == '__main__':
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    bandits, snaps = run_experiment(sample_points)
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Bandit Prior Distributions for True Means {BANDIT_MEANS} for Multi-Armed Bandit for {NUM_TRIALS} trials', fontweight='bold')
    axs = axs.ravel()
    for ax, trial in zip(axs, sample_points):
        plot_distribution(snaps[trial], ax)
        ax.set_title(f"After {trial} trials")
        ax.legend(fontsize="small")
    fig.tight_layout(pad=2.0)
    plt.show()
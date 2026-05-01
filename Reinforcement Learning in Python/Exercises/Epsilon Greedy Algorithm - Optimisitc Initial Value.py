from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype, float64, ndarray


# Define Bandit Class
class Bandit:
    def __init__(self, m: float, init_mean: float = 0.0):
        """
        Initializes an instance of the class with specific parameters for statistical
        calculation. This constructor sets up the initial value of an observed mean
        and allows for the initialization of a custom starting value for the mean.

        Args:
            m (float): The initial value used for calculations or as a reference
                for the observed mean.
            init_mean (float, optional): The initial mean value to start the
                calculations with. Defaults to 0.0.
        """
        self.m = m
        self.estimated_mean = init_mean
        self.N = 1

    def pull(self) -> float | ndarray[tuple[Any, ...], dtype[float64]]:
        """Pulls a sample from the bandit's distribution and returns it."""
        return np.random.randn() + self.m

    def update_estimate(self, x: float):
        """
        Updates the estimated mean of a sequence of values incrementally.

        This function updates the mean estimate when a new value is provided.
        The estimate is updated using an incremental formula to avoid recalculating
        the mean from scratch.

        Args:
            x (float): The new value to be incorporated into the mean estimate.
        """
        self.N += 1
        self.estimated_mean = self.estimated_mean + (x - self.estimated_mean) / self.N

def run_experiment(m1: float, m2: float, m3: float, initial_mean: float, num_trials: int) -> np.ndarray:
    """
    Runs an experiment with three bandits and returns the rewards obtained.

    Args:
        m1 (float): Mean of the first bandit's distribution.
        m2 (float): Mean of the second bandit's distribution.
        m3 (float): Mean of the third bandit's distribution.
        initial_mean (float): Initial mean estimate for each bandit.
        num_trials (int): Number of trials to run the experiment.

    Returns:
        np.ndarray: Array of rewards obtained from each trial.

    """
    bandits = [
        Bandit(m1, initial_mean),
        Bandit(m2, initial_mean),
        Bandit(m3, initial_mean),
    ]
    rewards = np.empty(num_trials)
    selection_counts = np.zeros(len(bandits), dtype=int)

    for trial in range(num_trials):
        best_bandit_index = np.argmax([bandit.estimated_mean for bandit in bandits])
        selected_bandit = bandits[best_bandit_index]

        reward = selected_bandit.pull()
        selected_bandit.update_estimate(reward)

        selection_counts[best_bandit_index] += 1
        rewards[trial] = reward

    cumulative_average = np.cumsum(rewards) / (np.arange(num_trials) + 1)

    print("Results for Optimistic Initial Values:")
    for index, bandit in enumerate(bandits, start=1):
        print(
            f"Bandit {index} - Estimated Mean: {bandit.estimated_mean:.4f}, "
            f"True Mean: {bandit.m:.4f}"
        )

    selection_percentages = (selection_counts / num_trials) * 100
    formatted_percentages = ", ".join(f"{percentage:.2f}%" for percentage in selection_percentages)
    print(f"Percent of times each bandit was selected: {formatted_percentages}")

    return cumulative_average

def run_experiment_eps(m1: float, m2: float, m3: float, eps: float, n_trials: int) -> np.ndarray:
    """
    Executes an epsilon-greedy experiment on a set of bandit instances over a specified number
    of trials and returns the cumulative average rewards.

    The function uses the epsilon-greedy strategy to balance exploration and exploitation
    during the experiment. It collects data on the rewards received from selecting each bandit
    and updates their estimated means accordingly. It outputs details about the true best bandit,
    the estimated means of each bandit, and the selection percentages.

    Args:
        m1 (float): The true mean reward of the first bandit.
        m2 (float): The true mean reward of the second bandit.
        m3 (float): The true mean reward of the third bandit.
        eps (float): The epsilon value for the epsilon-greedy strategy, indicating the
            probability of choosing a random bandit for exploration.
        n_trials (int): The number of trials to run in the experiment.

    Returns:
        np.ndarray: An array containing the cumulative average rewards for each trial
        in the experiment.
    """
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    means = np.array([m1, m2, m3])
    true_best = np.argmax(means)
    data = np.empty(n_trials)

    num_bandit_selections = np.zeros(len(bandits))

    for i in range(n_trials):
        p = np.random.rand()

        if p < eps:
            j = np.random.randint(len(bandits))
        else:
            j = np.argmax([b.estimated_mean for b in bandits])

        x = bandits[j].pull()
        bandits[j].update_estimate(x)
        data[i] = x
        num_bandit_selections[j] += 1

    cum_avg = np.cumsum(data) / (np.arange(n_trials) + 1)

    print("Results for Epsilon-Greedy:")
    print(f"True best bandit: {true_best + 1}")
    for b in bandits:
        print(f"Bandit {bandits.index(b) + 1} - Estimated Mean: {b.estimated_mean:.4f}, True Mean: {b.m:.4f}")

    selection_percentages = num_bandit_selections / n_trials * 100
    formatted_percentages = ", ".join(f"{pct:.2f}%" for pct in selection_percentages)
    print(f"Percent of times each bandit was selected: {formatted_percentages}")

    return cum_avg



# Function to plot the results
def plot_results(cum_avg: np.ndarray, m1: float, m2: float, m3: float, n: int, ax=None, title: str = 'Cumulative Average Reward vs True Mean'):
    """Plots the cumulative average reward over time."""
    ax = ax or plt.gca()
    ax.plot(cum_avg, label='Cumulative Average')
    ax.plot(np.ones(n) * m1, 'k--', label='True Mean Bandit 1')
    ax.plot(np.ones(n) * m2, 'b--', label='True Mean Bandit 2')
    ax.plot(np.ones(n) * m3, 'r--', label='True Mean Bandit 3')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Average Reward')
    ax.set_title(title)
    ax.legend(fontsize="small")

if __name__ == '__main__':
    mu1, mu2, mu3 = 0.2, 0.5, 0.75
    initial_mean = 10.0
    num_trials = 100000
    EPSILON = 0.1

    cumulative_average_opt_mean = run_experiment(mu1, mu2, mu3, initial_mean, num_trials)
    cumulative_average_eps = run_experiment_eps(mu1, mu2, mu3, EPSILON, num_trials)


    fig_compare, axs = plt.subplots(1, 2, figsize=(14, 5))
    plot_results(cumulative_average_opt_mean, mu1, mu2, mu3, num_trials, axs[0], f'Optimistic Initial Values with Mean {initial_mean:.2f}')
    plot_results(cumulative_average_eps, mu1, mu2, mu3, num_trials, axs[1], f'Epsilon-Greedy with EPSILON = {EPSILON:.2f}')
    fig_compare.tight_layout(pad=2.0)
    plt.show()

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype, float64, ndarray


# Define Bandit Class
class Bandit:
    def __init__(self, m: float, init_mean: float = 0.0):
        self.m = m
        self.estimated_mean = init_mean
        self.N = 1

    def pull(self) -> float | ndarray[tuple[Any, ...], dtype[float64]]:
        return np.random.randn() + self.m

    def update_estimate(self, x: float):
        self.N += 1
        self.estimated_mean = self.estimated_mean + (x - self.estimated_mean) / self.N

def run_experiment(m1: float, m2: float, m3: float, initial_mean: float, num_trials: int) -> np.ndarray:
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
    ax = ax or plt.gca()
    ax.plot(cum_avg, label='Cumulative Average')
    ax.plot(np.ones(n) * m1, 'k--', label='True Mean Bandit 1')
    ax.plot(np.ones(n) * m2, 'b--', label='True Mean Bandit 2')
    ax.plot(np.ones(n) * m3, 'r--', label='True Mean Bandit 3')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Average Reward')
    ax.set_title(title)
    ax.legend()

if __name__ == '__main__':
    mu1, mu2, mu3 = 0.2, 0.5, 0.75
    initial_mean = 10.0
    num_trials = 100000
    EPSILON = 0.1

    cumulative_average_opt_mean = run_experiment(mu1, mu2, mu3, initial_mean, num_trials)
    cumulative_average_eps = run_experiment_eps(mu1, mu2, mu3, EPSILON, num_trials)


    fig_compare, axs = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    plot_results(cumulative_average_opt_mean, mu1, mu2, mu3, num_trials, axs[0], f'Optimistic Initial Values with Mean {initial_mean:.2f}')
    plot_results(cumulative_average_eps, mu1, mu2, mu3, num_trials, axs[1], f'Epsilon-Greedy with EPSILON = {EPSILON:.2f}')
    plt.show()

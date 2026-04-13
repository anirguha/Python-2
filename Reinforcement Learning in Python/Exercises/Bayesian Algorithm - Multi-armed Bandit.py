import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from copy import deepcopy

# Define Hyperparameters
NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

# Define Bandit Class
class Bandit:
    def __init__(self,
                 p: float,
                 initial_mean: float = 0.0,
                 initial_alpha: float = 1.0,
                 initial_beta: float = 1.0):
        self.p = p
        self.mean = initial_mean
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.N = 0

    def pull(self) -> bool:
        return np.random.random() < self.p

    def update(self, x: float) -> None:
        self.alpha += x
        self.beta += 1 - x
        self.N += 1

        return None

    def sample(self) -> float:
        return np.random.beta(self.alpha, self.beta)

# Function to plot the bandit distributions
def plot_distributions(bandit_lst: list[Bandit], ax: plt.Axes) -> None:
    x = np.linspace(0, 1, 200)
    for b in bandit_lst:
        y = stats.beta.pdf(x, b.alpha, b.beta)
        wins = int(b.alpha - 1)
        ax.plot(x, y, label=f"p={b.p:.2f}, wins={wins}/{b.N}")

    return None

def run_experiment(sample_points: list[int] | None = None) -> tuple[list[Bandit], dict[int, list[Bandit]]]:
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    sample_points = sorted(sample_points or [])
    snapshots: dict[int, list[Bandit]] = {}

    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])

        x = bandits[j].pull()

        rewards[i] = x

        bandits[j].update(x)

        trial_num = i + 1
        if trial_num in sample_points:
            snapshots[trial_num] = deepcopy(bandits)

    # Print the average reward for each bandit
    print(f"Average reward for each bandit: {rewards.mean()}")
    print(f"Overall win rate: {rewards.mean()}")
    selection_percentages = [f"{b.N / NUM_TRIALS * 100:.2f}%" for b in bandits]
    print(f"Number of times each bandit was selected: {', '.join(selection_percentages)}")

    return bandits, snapshots

if __name__ == "__main__":
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]

    bandits, snapshots = run_experiment(sample_points)

    # Plot
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.ravel()

    for ax, trial in zip(axs, sample_points):
        plot_distributions(snapshots[trial], ax)
        ax.set_title(f"After {trial} trials")
        ax.legend(fontsize="small")

    fig.tight_layout(pad=2.0)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define Bandit class for Gaussian rewards
class Bandit:
    """Represents one arm in an epsilon-greedy multi-armed bandit experiment with Gaussian rewards.

    The class stores the arm's true mean reward and maintains an online
    estimate of that mean as rewards are observed.

    Args:
        mu (float): True mean reward for this arm.
        initial_estimate (float, optional): Initial value for the estimated mean reward. Defaults to 0.0.
    """
    def __init__(self, mu, initial_estimate=0.0):
        self.mu = mu
        self.estimate = initial_estimate
        self.N = 0

    def pull(self):
        """Samples a reward from this bandit arm.

        Returns:
            float: Sampled reward from the Gaussian distribution.
        """
        return np.random.normal(0, 1) + self.mu

    def update_estimate(self, x):
        """Updates the running mean reward estimate using one new observation.

        Args:
            x (float): Observed reward for the latest pull.
        """
        self.N += 1
        self.estimate = self.estimate + (x - self.estimate) / self.N

# Function to run the experiment
def run_experiment(m1: float, m2: float, m3: float, e: float, n: int, decaying_epsilon: bool = False):
    """Runs an epsilon-greedy multi-armed bandit experiment with Gaussian rewards.

    Args:
        m1 (float): True mean reward for arm 1.
        m2 (float): True mean reward for arm 2.
        m3 (float): True mean reward for arm 3.
        e (float): Initial epsilon value for the epsilon-greedy strategy.
        n (int): Number of trial runs.
        decaying_epsilon (bool, optional): If True, epsilon decays as 1/t so
            exploration fades over time and the cumulative average rises toward
            the optimal. If False, epsilon is fixed. Defaults to False.
    """
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    # Count the number of suboptimal choices
    means = np.array([m1, m2, m3])
    true_best = np.argmax(means)
    cnt_suboptimal: int = 0

    data = np.empty(n)

    for i in range(n):
        # Epsilon greedy strategy
        # With decaying epsilon, exploration rate shrinks as 1/t so the
        # algorithm gradually shifts from exploration to exploitation and the
        # cumulative average converges upward to the optimal mean.
        eps_t = 1 / (i + 1) if decaying_epsilon else e
        p = np.random.random()

        if p < eps_t:
            # Explore: select a random bandit
            j = np.random.randint(len(bandits))
        else:
            # Exploit: select the bandit with the highest estimated mean reward
            j = np.argmax([b.estimate for b in bandits])
        x = bandits[j].pull()
        bandits[j].update_estimate(x)

        if j != true_best:
            cnt_suboptimal += 1

        # For the plot
        data[i] = x

    cum_avg = np.cumsum(data) / (np.arange(n) + 1)

    return cum_avg, cnt_suboptimal, bandits

# Function to plot the results
def plot_results(cum_avg, e: float):
    """Plots the cumulative average reward over time.

    Args:
        cum_avg (numpy.ndarray): Array of cumulative average rewards.
        e (float): Epsilon value used in the experiment, for labeling the plot.

    """
    plt.plot(cum_avg, label=f"epsilon = {e}")

    return None

# Main function
if __name__ == '__main__':
    mu1, mu2, mu3 = 1.5, 2.5, 3.5
    N = 100_000
    eps = [0.1, 0.05, 0.01]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Epsilon-Greedy Multi-Armed Bandit with Gaussian Rewards', fontweight='bold')

    # --- Fixed epsilon (curves descend toward their asymptote) ---
    plt.sca(axes[0])
    axes[0].set_title('Fixed Epsilon')
    for eps_value in eps:
        print(f"[Fixed] Running experiment with epsilon = {eps_value}...")
        cumulative_average, count_suboptimal, bandits = run_experiment(mu1, mu2, mu3, eps_value, N)
        plot_results(cumulative_average, eps_value)
        print(f"  Suboptimal choices: {count_suboptimal}")
    plt.plot(np.ones(N) * max(mu1, mu2, mu3), 'k--', label='True Best Mean')
    plt.xscale('log')
    plt.xlabel('Number of trials')
    plt.ylabel('Cumulative Average Reward')
    plt.legend()

    # --- Decaying epsilon 1/t (curves rise toward the optimal) ---
    plt.sca(axes[1])
    axes[1].set_title('Decaying Epsilon (1/t)')
    for eps_value in eps:
        print(f"[Decaying] Running experiment with initial epsilon = {eps_value}...")
        cumulative_average, count_suboptimal, bandits = run_experiment(
            mu1, mu2, mu3, eps_value, N, decaying_epsilon=True
        )
        plot_results(cumulative_average, eps_value)
        print(f"  Suboptimal choices: {count_suboptimal}")
    plt.plot(np.ones(N) * max(mu1, mu2, mu3), 'k--', label='True Best Mean')
    plt.xscale('log')
    plt.xlabel('Number of trials')
    plt.ylabel('Cumulative Average Reward')
    plt.legend()

    plt.tight_layout()
    plt.show()

    for b in bandits:
        print(f"Estimated mean for bandit with true mean {b.mu}: {b.estimate:.2f}")

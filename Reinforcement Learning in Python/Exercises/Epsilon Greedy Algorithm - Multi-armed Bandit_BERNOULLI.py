

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    """Represents one arm in an epsilon-greedy multi-armed bandit experiment.

    The class stores the arm's true win probability and maintains an online
    estimate of that probability as rewards are observed.

    Args:
        p (float): True probability of returning a reward of 1.
        Initial_estimate (float, optional): Initial value for the estimated
            win probability. Defaults to 0.5.

    Attributes:
        p (float): True probability of reward for this arm.
        p_estimate (float): Running estimate of the arm's win probability.
        N (int): Number of times this arm has been pulled.
    """
    def __init__(self, p: float, initial_estimate: float = 0.5):
        self.p = p
        self.p_estimate = initial_estimate # small non-zero value to encourage exploration
        self.N = 0

    def pull(self):
        """Samples a reward from this bandit arm.

        Returns:
            int: 1 for a win and 0 for a loss.
        """
        return int(np.random.random() < self.p)

    def update_estimate(self, x: float):
        """Updates the running win-rate estimate using one new observation.

        Args:
            x (float): Observed reward for the latest pull (typically 0 or 1).
        """
        self.N += 1
        self.p_estimate = self.p_estimate + (x - self.p_estimate) / self.N

# Main
NUM_TRIALS = 10_000
EPSILON = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

# Instantiate bandits with the given probabilities
bandits = [Bandit(p, initial_estimate=0) for p in BANDIT_PROBABILITIES]
rewards = np.zeros(NUM_TRIALS) # array to store the reward for each trial
num_times_explored = 0
num_times_exploited = 0
num_optimal_bandit_selected = 0
print('\n'.join(f"Bandit {i}: p = {b.p}" for i, b in enumerate(bandits)))
optimal_bandit_index = np.argmax([b.p for b in bandits])
print("Optimal Bandit Index:", optimal_bandit_index)

for trial in range(NUM_TRIALS):
    random_value: float = np.random.random()
    # use epsilon greedy to choose the next bandit. If probability < EPSILON explore, else exploit
    if random_value < EPSILON:
        selected_bandit_index = np.random.choice(len(bandits)) # randomly select from one of the bandits
        num_times_explored += 1
    else:
        # Select the bandit with highest estimated probability of winning
        selected_bandit_index = np.argmax([b.p_estimate for b in bandits])
        num_times_exploited += 1


    if selected_bandit_index == optimal_bandit_index:
        num_optimal_bandit_selected += 1 # Updates the number of times the optimal bandit is selected

    # Pull the arm for the bandit for the selected bandit and get the reward for this trial
    trial_reward = bandits[selected_bandit_index].pull() # returns reward: 1 if random value < bandit probability, 0 otherwise


    rewards[trial] = trial_reward # log the reward for this trial
    bandits[selected_bandit_index].update_estimate(trial_reward) # update the estimated probability of winning for the selected bandit

# Print estimated probabilities for each bandit
print("*" * 10 + "End of Experiment" + "*" * 10)
print('\n'.join(f"Bandit {i}: p estimate = {b.p_estimate:.2f}, true p = {b.p}" for i, b in enumerate(bandits)))
print(f"Number of times explored: {num_times_explored}")
print(f"Number of times exploited: {num_times_exploited}")
print(f"Number of times optimal bandit selected: {num_optimal_bandit_selected}")
print(f"Average Reward: {rewards.sum() / NUM_TRIALS}")
print(f"Total reward: {np.sum(rewards)}")

# Plot the rewards
cumulative_rewards = np.cumsum(rewards)
average_reward = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
plt.plot(average_reward, label='Win Ratio')
plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES), label='Optimal Probability')
plt.gca().set(xlabel='Trial', ylabel='Win Ratio', title='Win Ratios by trial run')

plt.legend()
plt.show()




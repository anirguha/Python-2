import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 10000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
rewards = np.zeros(NUM_TRIALS)
EPS = 0.1

# Define Bandit class
class Bandit:
    """
    Represents a single-armed bandit as part of a multi-armed bandit problem.

    This class models a single slot machine (bandit) with a true win rate (p)
    and maintains an estimate of the win rate based on the outcomes of
    interactions with the bandit. It allows for simulating pulls on the bandit
    and updating its estimated win rate.

    Attributes:
        p (float): The true win rate of the bandit, a probability between 0 and 1.
        p_estimate (float): The estimated win rate of the bandit based on interaction history.
        N (int): The number of times this bandit has been played.
    """
    def __init__(self, p):
        """
        Initializes the bandit with its true win rate and tracks its estimated win
        rate and the number of times it has been played.

        Args:
            p: True win rate of the bandit.

        Attributes:
            p_estimate: Float representing the current estimated win rate of the bandit.
            N: Integer representing the number of times the bandit has been played.
        """
        self.p = p  # true win rate
        self.p_estimate = 0.0  # estimated win rate
        self.N = 0  # number of times this bandit has been played

    def pull(self):
        """
        Simulates the action of pulling a Bernoulli arm in a multi-armed bandit problem.

        The method determines whether the pull of the arm is successful based on the
        probability associated with the arm. A random value is generated and compared
        against the success probability `p` to decide the result.

        Returns:
            bool: True if the pull is successful (random value is less than `p`), False otherwise.
        """
        return np.random.random() < self.p

    def update(self, x):
        """
        Updates the estimate of a probability value with the given input.

        The method incrementally updates the probability estimate (p_estimate)
        based on new input data (x) while keeping track of the total count (N).

        Args:
            x: New input value used to update the probability estimate (float or int).
        """
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

# Define UCB1 algorithm
def ucb1(mean, n, nj):
    """
    Calculates the Upper Confidence Bound (UCB1) value for a given input.

    The UCB1 algorithm is used in multi-armed bandit problems to balance exploration and exploitation.
    It computes an upper bound score considering the average reward, the total number of trials, and
    the number of times a specific arm has been selected.

    Args:
        mean: The mean reward obtained from the selected arm.
        n: The total number of trials conducted.
        nj: The number of times the specific arm has been selected.

    Returns:
        float: The computed UCB1 value.
    """
    return mean + np.sqrt(2 * np.log(n) / nj)

ucb_values = np.zeros(NUM_TRIALS)
total_plays = 0
bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

# Initialization: play each bandit once
for b in bandits:
    x = b.pull()
    b.update(x)
    total_plays += 1

# Main loop: play each bandit NUM_TRIALS times
for t in range(NUM_TRIALS):
    ucb_scores = [ucb1(b.p_estimate, total_plays, b.N) for b in bandits]
    j = np.argmax(ucb_scores)
    x = bandits[j].pull()
    ucb_values[t] = ucb_scores[j]
    bandits[j].update(x)
    total_plays += 1
    rewards[t] = x

cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

plt.plot(cumulative_average, label='UCB1')
plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES), label='Optimal')
plt.gca().set(xlabel='Trial', ylabel='Win Ratio', title='Win Ratios by trial run')
plt.xscale('log')
plt.legend()
plt.show()

for b in bandits:
    print(b.p_estimate)

print("total reward earned:", rewards.sum())
print("overall win rate:", rewards.sum() / NUM_TRIALS)
selection_percentages = ",".join([f"{b.N / total_plays * 100:.2f}" for b in bandits])
print(f"Percentage times selected each bandit: {selection_percentages}")
print("ucb values:", ucb_values)
print("total plays:", total_plays)



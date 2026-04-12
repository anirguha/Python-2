import numpy as np
import matplotlib.pyplot as plt

NUM_TRIALS = 10000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
rewards = np.zeros(NUM_TRIALS)
EPS = 0.1

# Define Bandit class
class Bandit:
    def __init__(self, p):
        self.p = p  # true win rate
        self.p_estimate = 0.0  # estimated win rate
        self.N = 0  # number of times this bandit has been played

    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

# Define UCB1 algorithm
def ucb1(mean, n, nj):
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



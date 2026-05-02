from typing import Dict, Tuple, List
from random import choice

from sklearn.kernel_approximation import RBFSampler


import numpy as np
import matplotlib.pyplot as plt
from gridworld_standard_windy import(
WindyGridworld,
standard_gridworld,
# negative_reward_gridworld,
State
)

from pretty_printing import print_values, print_policy

# Define types
Action = str
Reward = float
Policy = Dict[State, Action]
ActionValueTable = Dict[State, Dict[Action, Reward]]
SampleCountTable = Dict[State, Dict[Action, int]]
StateSampleCounts = Dict[State, float]
ValueTable = Dict[State, Reward]

# Hyperparameters
GRID_SIZE: Tuple[int, int] = (3, 4)
START_STATE: State = (2, 0)
TERMINAL_STATES: Tuple[State, ...] = ((0, 3), (1, 3))
STEP_COST: float = -0.5
POLICY: Policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }

def epsilon_greedy(p: Policy,
                   g: WindyGridworld,
                   s: State,
                   epsilon: float) -> Action:

    action_map: Dict[State, Tuple[Action, ...]] = g.get_action_space()
    if np.random.random() < epsilon:
        return choice(action_map[s]) # Explore
    else:
        return p[s] # Exploit

def collect_samples(p: Policy,
                    g: WindyGridworld,
                    num_samples: int,
                    epsilon: float) -> List[State]:
    samples: List[State] = []

    for _ in range(num_samples):
        start_state = START_STATE
        g.set_state(start_state)
        samples.append(start_state)

        while not g.end_episode():
            s: State = g.current_state()
            a: Action = epsilon_greedy(p, g, s, epsilon)
            g.move(a)
            s_next: State = g.current_state()
            samples.append(s_next)

    return samples

class ValueFunctionApproximator:
    def __init__(self,
                 p: Policy,
                 g: WindyGridworld,
                 num_samples: int,
                 epsilon: float,
                 lr: float,
                 gamma: float):
        self.p = p
        self.g = g
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.samples = collect_samples(p, g, num_samples, epsilon)

        self.sampler = RBFSampler()
        self.sampler.fit(self.samples)
        dims = self.sampler.n_components

        self.w = np.zeros(dims)

    def predict(self, s: State) -> Reward:
        x = self.sampler.transform([s])[0]
        return x @ self.w

    def grad(self, s: State) -> np.ndarray:
        x = self.sampler.transform([s])[0]
        return x

def run_td0_value_approximation(p: Policy,
               g: WindyGridworld,
               num_samples: int,
               epsilon: float,
               lr: float,
               gamma: float,
               num_iterations: int) -> Tuple[List[float], ValueFunctionApproximator]:
    approx = ValueFunctionApproximator(p, g, num_samples, epsilon, lr, gamma)
    mse_per_episode: List[float] = []

    for _ in range(num_iterations):
        s: State = START_STATE
        g.set_state(s)

        Vs = approx.predict(s)
        n_steps: int = 0
        episode_err: float = 0.0

        while not g.end_episode():
            a: Action = epsilon_greedy(p, g, s, epsilon)
            r: Reward = g.move(a)
            s_next: State = g.current_state()
            n_steps += 1

            if g.is_terminal(s_next):
                target = r
                Vs_next = 0.0
            else:
                Vs_next = approx.predict(s_next)
                target = r + gamma * Vs_next

            # Update model weights
            grad = approx.grad(s)
            err = target - Vs
            approx.w += approx.lr * err * grad

            # Accumulate error
            episode_err += err*err

            # Update state
            s = s_next
            Vs = Vs_next
        if n_steps > 0:
            mse_per_episode.append(episode_err / n_steps)
        else:
            mse_per_episode.append(0.0)

    return mse_per_episode, approx


def get_predicted_state_values(g: WindyGridworld, approx: ValueFunctionApproximator) -> ValueTable:
    """Build V(s) from the trained approximator for all known states."""
    values: ValueTable = {}
    for s in g.get_all_states():
      if s in g.actions:
          values[s] = approx.predict(s)
      else:
          # keep terminal values explicit from environment rewards
          values[s] = g.rewards.get(s, 0.0)
    return values

def plot_mse(mse_per_episode: List[float]) -> None:
    plt.plot(mse_per_episode)
    plt.title("MSE per episode")
    backend = plt.get_backend().lower()
    if 'agg' in backend:
        plt.savefig('mse_per_episode.png')
        plt.close()
    else:
        plt.show()

def main() -> None:
    rows, cols = GRID_SIZE
    g: WindyGridworld = standard_gridworld(rows, cols, START_STATE, TERMINAL_STATES)

    print("Initial Policy:")
    print_policy(POLICY, g)
    print()

    print("Initial Rewards:")
    print_values(g.rewards, g)
    print()

    p: Policy = POLICY
    num_samples: int = 10000
    epsilon: float = 0.1
    lr: float = 0.1
    gamma: float = 0.9
    num_iterations: int = 10000

    mse_per_episode, approx = run_td0_value_approximation(p, g, num_samples, epsilon, lr, gamma, num_iterations)
    plot_mse(mse_per_episode)

    print("Final Policy:")
    print_policy(POLICY, g)
    print()

    print("Predicted State Values:")
    predicted_values = get_predicted_state_values(g, approx)
    print_values(predicted_values, g)


if __name__ == "__main__":
    main()

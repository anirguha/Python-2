import numpy as np
import matplotlib.pyplot as plt

input_values = np.random.randn(10)
print(f'a: {input_values} | Max value: {np.max(input_values)}')

exp_a = np.exp(input_values)
naive_softmax = exp_a / np.sum(exp_a)
print(naive_softmax)
exp_a_stable = np.exp(input_values - np.max(input_values))
c = exp_a_stable / np.sum(exp_a_stable)
stable_softmax = np.exp(input_values - np.max(input_values)) / np.sum(np.exp(input_values - np.max(input_values)))
print(stable_softmax)

# rng = np.random.default_rng()
# rng.multinomial(1, naive_softmax)  # Sample from the naive softmax distribution

fig, axs = plt.subplots(1,2, figsize=(12, 4))
axs[0].plot(input_values, 'o-')
axs[0].set_title('Input')
axs[1].plot(naive_softmax, 'o-', label='Naive')
axs[1].plot(stable_softmax, 'x-', label='Stable')
axs[1].set_title('Softmax Output')
axs[1].legend()
plt.show()
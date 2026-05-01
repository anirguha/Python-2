import numpy as np
import matplotlib
# matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

print(matplotlib.get_backend())
# Reproducible synthetic data for y = m*x + b + noise
rng = np.random.default_rng(42)
n_samples = 100
true_slope = 2.5
true_intercept = 4.0
noise_std = 2.0

x = rng.uniform(0, 10, n_samples)
noise = rng.normal(0, noise_std, n_samples)
y = true_slope * x + true_intercept + noise


def predict(x_values, slope, intercept):
	return slope * x_values + intercept


def train_gradient_descent(x_values, y_values, learning_rate, epochs):
	m = 0.0
	b = 0.0
	loss_history = []
	n = len(x_values)

	for _ in range(epochs):
		y_pred = predict(x_values, m, b)
		error = y_pred - y_values

		dm = (2 / n) * np.sum(error * x_values)
		db = (2 / n) * np.sum(error)

		m -= learning_rate * dm
		b -= learning_rate * db

		loss = np.mean(error ** 2)
		loss_history.append(loss)

	return m, b, loss_history


# Standardize x to compare convergence behavior
x_mean = np.mean(x)
x_std = np.std(x)
x_scaled = (x - x_mean) / x_std

learning_rate = 0.01
epochs = 1000

m_unscaled, b_unscaled, loss_history_unscaled = train_gradient_descent(
	x, y, learning_rate, epochs
)

m_scaled, b_scaled, loss_history_scaled = train_gradient_descent(
	x_scaled, y, learning_rate, epochs
)

# Convert scaled-model parameters back to original x units
m_scaled_original = m_scaled / x_std
b_scaled_original = b_scaled - (m_scaled * x_mean / x_std)

print(f"True slope: {true_slope:.3f}, True intercept: {true_intercept:.3f}")
print(
	f"Unscaled fit -> slope: {m_unscaled:.3f}, intercept: {b_unscaled:.3f}, "
	f"Final MSE: {loss_history_unscaled[-1]:.3f}"
)
print(
	f"Scaled fit   -> slope: {m_scaled_original:.3f}, intercept: {b_scaled_original:.3f}, "
	f"Final MSE: {loss_history_scaled[-1]:.3f}"
)

plt.scatter(x, y, alpha=0.7, label="Data points")

# Show true line and learned line
x_line = np.array([x.min(), x.max()])
y_line = true_slope * x_line + true_intercept
plt.plot(x_line, y_line, color="red", label="True relationship")
y_fit_unscaled = predict(x_line, m_unscaled, b_unscaled)
plt.plot(x_line, y_fit_unscaled, color="green", linestyle="--", label="Fit (unscaled x)")
y_fit_scaled = predict(x_line, m_scaled_original, b_scaled_original)
plt.plot(x_line, y_fit_scaled, color="blue", linestyle=":", label="Fit (scaled x)")

plt.title("Synthetic Data for Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.legend()

plt.figure()
plt.plot(loss_history_unscaled, label="Unscaled x")
plt.plot(loss_history_scaled, label="Scaled x")
plt.title("Gradient Descent Loss Comparison (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()

# if "agg" in plt.get_backend().lower():
# 	plt.close("all")
# else:
# 	plt.show()

plt.show()

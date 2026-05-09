# AGENTS.md - Python Learning Repository

## Repository Purpose & Architecture

This is a **learning/course repository** containing implementations of ML/AI concepts across multiple domains:
- **Reinforcement Learning** (Gridworld, Monte Carlo, Q-Learning, Policy Iteration)
- **Data Science** (Classification, Regression, NLP, Unsupervised Learning)
- **Deep Learning** (PyTorch, Vision Transformers, LLM Mechanisms)
- **Data Engineering** (Pandas, PySpark, CSV processing)

Each topic is in its own top-level directory. Key directories:
- `Reinforcement Learning in Python/Exercises/` - Primary RL exercise implementations
- `tests/` - Unit test suite using pytest
- `Helper Functions/` - Shared utility modules (model saving, image processing, etc.)

## Critical Code Patterns & Conventions

### Type Hints & Custom Type Aliases (Mandatory)
All RL files extensively use type hints. Define custom types at module top:
```python
from typing import Dict, Tuple, List, Literal, Mapping, Sequence

State = Tuple[int, int]  # Grid coordinates (row, col)
Action = Literal['U', 'D', 'L', 'R']  # Directional movement
ValueTable = Mapping[State, float]  # Value function mapping
Policy = Dict[State, Action]  # Deterministic policy mapping
```

Import from `Reinforcement_Learning_in_Python.Exercises` (module path), not relative imports.
See: `Reinforcement Learning in Python/Exercises/gridworld_standard_windy.py`, `pretty_printing.py`

### Gridworld Environment Pattern
Never hardcode grids. Use the `WindyGridworld` class with factory functions:
```python
from gridworld_standard_windy import WindyGridworld, negative_reward_gridworld

grid = negative_reward_gridworld(
    rows=3, cols=4,
    start=(2, 0),
    terminal_states=((0, 3), (1, 3)),
    step_cost=-0.5
)
```
Key methods: `get_all_states()`, `get_action_space()`, `get_transitions()`, `get_reward()`
See: `Reinforcement Learning in Python/Exercises/gridworld_monte_carlo_epsilon_greedy.py`

### Visualization with pretty_printing Module
Use consistent pretty-printing functions instead of manual printing:
```python
from pretty_printing import print_policy, print_values

print_values(V=value_table, g=grid)  # Prints grid with state values
print_policy(p=policy, g=grid)  # Prints policy as grid of actions
```
See: `Reinforcement Learning in Python/Exercises/pretty_printing.py`

### Optional Dependencies & Graceful Degradation
Files using optional libraries must wrap imports in try/except blocks with fallback implementations:
```python
try:
    from sklearn.kernel_approximation import RBFSampler
except ModuleNotFoundError:
    RBFSampler = None

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    class _MissingPyplot:
        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                f"matplotlib is required for plotting. Install matplotlib to use plt.{name}()."
            )
    plt = _MissingPyplot()

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Any, **_kwargs: Any) -> Any:
        return iterable
```
This allows optional dependencies (sklearn, matplotlib, tqdm, gymnasium, pandas) to be imported conditionally.
See: `Reinforcement Learning in Python/Exercises/gridworld_policy_control_approximation_algorithm.py`, `cartpole_control_approximation_algorithm.py`

### Module-Level Comprehensive Docstrings
Complex modules should include a module-level docstring describing the algorithm, key components, and dependencies:
```python
"""
CartPole Control with Function Approximation using RBF Features

This module implements a Q-learning agent with value-function approximation
for solving the CartPole-v1 environment from OpenAI Gymnasium. The agent uses
Random Fourier Features (via scikit-learn's RBFSampler) to approximate the
Q-value function.

Key Parts:
- epsilon_greedy(): Action selection policy balancing exploration/exploitation
- ValueFunctionApproximator: Linear approximator over RBF features
- train_agent(): Q-learning training loop

Dependencies:
- gymnasium: Environment simulation
- numpy: Numerical computations
- scikit-learn: RBF feature transformation
- matplotlib: Optional visualization
"""
```
See: `Reinforcement Learning in Python/Exercises/cartpole_control_approximation_algorithm.py`

### Module-Level Constants for Experiments
Always define experiment hyperparameters as module-level constants:
```python
GRID_SIZE: Tuple[int, int] = (3, 4)
START_STATE: State = (2, 0)
TERMINAL_STATES: Tuple[State, ...] = ((0, 3), (1, 3))
STEP_COST: float = -0.5
LEARNING_RATE: float = 0.1
EPSILON: float = 0.1
```
This enables easy configuration without function parameter hunting.

### Function Approximation Pattern
For value-function approximation using RBF features, implement a `ValueFunctionApproximator` class:
```python
from sklearn.kernel_approximation import RBFSampler

class ValueFunctionApproximator:
    def __init__(self, samples: List[State], n_components: int = 100, gamma: float = 1.0):
        self.sampler = RBFSampler(n_components=n_components, gamma=gamma)
        self.sampler.fit(samples)
        self.w = np.zeros(self.sampler.n_components)
    
    def predict(self, state: State) -> float:
        features = self.sampler.transform([state])[0]
        return features @ self.w
    
    def update(self, state: State, target: float, learning_rate: float) -> None:
        features = self.sampler.transform([state])[0]
        error = target - (features @ self.w)
        self.w += learning_rate * error * features
```
Use RBFSampler for feature transformation and linear weight updates for rapid prototyping.
See: `Reinforcement Learning in Python/Exercises/gridworld_prediction_approx_algorithm.py`, `gridworld_policy_control_approximation_algorithm.py`

## Testing & Execution

### Run Tests
```bash
cd "/Users/AnirbanGuha/Library/CloudStorage/GoogleDrive-guhaa1@gmail.com/My Drive/GitHub/Python-2"
pytest tests/ -v
```

Test files follow convention: `tests/test_<module_name>.py`
Tests use `unittest.TestCase` with `setUp()` method for common fixtures.
Mock GridWorld instances with `negative_reward_gridworld()` factory.
See: `tests/test_gridworld_policy_control_q_learning.py`

### Run Individual Exercise Scripts
```bash
# Set PYTHONPATH to include project root
cd "Reinforcement Learning in Python/Exercises"
python gridworld_monte_carlo_epsilon_greedy.py
```

### Dependencies
Core dependencies (see `Reinforcement Learning in Python/Exercises/requirements.txt`):
- `pytest >= 9.0.0` - Testing framework
- `numpy`, `pandas`, `matplotlib` - Data & visualization
- `gymnasium` - RL environment interface (CartPole, et al. - modern replacement for gym)
- `scipy`, `seaborn` - Analysis & plotting
- `scikit-learn` - ML preprocessing & RBF feature sampling for approximation algorithms
- `tqdm` - Progress bars (optional, gracefully degraded if not installed)

## Debugging & Common Issues

### Import Path Issues
The project uses an import bridge pattern: The `Reinforcement_Learning_in_Python` package (with underscores) wraps the legacy `Reinforcement Learning in Python` directory (with spaces) via sys.path manipulation in `Reinforcement_Learning_in_Python/Exercises/__init__.py`. This allows:
```python
from Reinforcement_Learning_in_Python.Exercises import gridworld_standard_windy
from Reinforcement_Learning_in_Python.Exercises.gridworld_monte_carlo_epsilon_greedy import get_all_states
```

If imports fail, ensure `PYTHONPATH` includes the project root AND that the underscore-named directory is a proper Python package:
```bash
export PYTHONPATH="/full/path/to/Python-2:$PYTHONPATH"
cd "/full/path/to/Python-2"
python -c "from Reinforcement_Learning_in_Python.Exercises import gridworld_standard_windy"
```

### Relative vs Absolute Imports
- **Within exercises**: Use `from gridworld_standard_windy import ...`
- **From outside**: Use `from Reinforcement_Learning_in_Python.Exercises import gridworld_monte_carlo_epsilon_greedy`
- **Never use**: `from ..Exercises import` - leads to import errors

### Data File Paths & Environments
Exercise scripts expect `data/` directory relative to execution directory: 
Example: `pd.read_csv("data/aapl_msi_sbux.csv")`

For gymnasium-based environments (CartPole, et al.), import from the `gymnasium` package:
```python
import gymnasium as gym

env = gym.make('CartPole-v1')
state, info = env.reset()
action = env.action_space.sample()
next_state, reward, terminated, truncated, info = env.step(action)
```
See: `Reinforcement Learning in Python/Exercises/cartpole_control_approximation_algorithm.py`

## Workflow Recommendations for AI Agents

1. **When modifying exercises**: Check test file first (`tests/test_*.py`) to understand expected behavior
2. **When adding algorithms**: Use `WindyGridworld` + `pretty_printing` for consistency
3. **Type hints required**: Always add full type hints - no ambiguous `Any` types in algorithm implementations
4. **Docstrings format**: Use Google-style docstrings with Args/Returns sections (see `pretty_printing.py`)
5. **Keep experiments reproducible**: Initialize random seeds and use module-level constants for parameters
6. **Optional dependencies**: Wrap external library imports in try/except blocks with fallbacks. Check if dependency is None before using.
7. **For new RL code**: Look at `gridworld_monte_carlo_epsilon_greedy.py` as the basic reference, or `gridworld_policy_control_approximation_algorithm.py` for function approximation patterns
8. **For approximation algorithms**: Start with `RBFSampler` for feature transformation and implement a `ValueFunctionApproximator` class with `predict()` and `update()` methods
9. **Module docstrings**: If implementing complex algorithms (approximation, CartPole environments), include comprehensive module-level docstrings describing Key Parts and Dependencies

## Key Files to Review First
- `Reinforcement_Learning_in_Python/Exercises/__init__.py` - Import bridge pattern (sys.path manipulation for space-named directories)
- `Reinforcement Learning in Python/Exercises/__init__.py` - Not a package; legacy working directory
- `Reinforcement Learning in Python/Exercises/gridworld_standard_windy.py` - Core gridworld environment (120+ LOC with detailed docs)
- `Reinforcement Learning in Python/Exercises/pretty_printing.py` - Visualization utilities with Google-style docstrings
- `Reinforcement Learning in Python/Exercises/gridworld_monte_carlo_epsilon_greedy.py` - Reference RL algorithm implementation
- `Reinforcement Learning in Python/Exercises/gridworld_policy_control_approximation_algorithm.py` - Function approximation pattern with optional imports
- `Reinforcement Learning in Python/Exercises/cartpole_control_approximation_algorithm.py` - Gymnasium environment + approximation (900+ LOC reference)
- `tests/test_gridworld_monte_carlo_epsilon_greedy.py` - Reference test patterns with mocking
- `tests/test_gridworld_policy_control_approximation_algorithm.py` - Approximation algorithm testing with mocks


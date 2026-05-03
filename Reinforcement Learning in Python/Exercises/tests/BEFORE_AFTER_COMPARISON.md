# Before vs After Code Comparison

## Problem Statement
You wanted to know how to achieve `collect_samples()` calling `run_episode()` while still collecting state-action pair samples. The original code had duplicated environment-stepping logic.

## Solution: Shared Helper Function

### BEFORE: Duplicated Code
```python
# collect_samples had its own loop
def collect_samples(env: gymnasium.Env, num_samples: int) -> Samples:
    samples: Samples = []
    for _ in tqdm(range(num_samples)):
        s, info = env.reset()
        done: bool = False
        truncated: bool = False

        while not (done or truncated):
            a = env.action_space.sample()
            sa = np.concatenate((s, [a]))  # Collect state-action pairs
            samples.append(sa)

            s, r, done, truncated, info = env.step(a)

    return samples


# run_episode had DUPLICATE stepping logic
def run_episode(env: gymnasium.Env,
                eps: float = EPSILON) -> float:
    s, info = env.reset()
    done: bool = False
    truncated: bool = False
    episode_reward: float = 0.0

    while not (done or truncated):
        a = epsilon_greedy(s, eps)  # Different action selection
        s, r, done, truncated, info = env.step(a)
        episode_reward += r

    return episode_reward
```

**Problems:**
- ❌ Almost identical stepping logic in two places
- ❌ Bug in one would need to be fixed in both
- ❌ No code reuse
- ❌ Harder to maintain

### AFTER: Unified Helper + Refactored Functions

```python
# Step 1: Create unified helper function that handles the core logic
def run_episode_with_collection(env: gymnasium.Env,
                                 action_selector,
                                 collect_sa_pairs: bool = False) -> Tuple[float, Samples]:
    """
    Run an episode with a given action selector and optionally collect samples.
    
    Key design:
    - action_selector: A function that takes state and returns action
    - collect_sa_pairs: Toggle sample collection on/off
    """
    s, info = env.reset()
    done: bool = False
    truncated: bool = False
    episode_reward: float = 0.0
    samples: Samples = []

    while not (done or truncated):
        a = action_selector(s)  # Flexible action selection
        
        # Conditionally collect state-action pairs
        if collect_sa_pairs:
            sa = np.concatenate((s, [a]))
            samples.append(sa)
        
        s, r, done, truncated, info = env.step(a)
        episode_reward += r

    return episode_reward, samples


# Step 2: Refactor collect_samples to use the helper
def collect_samples(env: gymnasium.Env, num_samples: int) -> Samples:
    """Now calls run_episode_with_collection internally!"""
    all_samples: Samples = []
    
    for _ in tqdm(range(num_samples)):
        # Random action selector for exploration
        action_selector = lambda s: env.action_space.sample()
        
        # Run episode and collect samples
        _, samples = run_episode_with_collection(
            env, 
            action_selector, 
            collect_sa_pairs=True  # Enable collection
        )
        
        all_samples.extend(samples)
    
    return all_samples


# Step 3: Refactor run_episode to use the helper
def run_episode(env: gymnasium.Env,
                eps: float = EPSILON) -> float:
    """Now calls run_episode_with_collection internally!"""
    # Epsilon-greedy action selector
    action_selector = lambda s: epsilon_greedy(s, eps)
    
    # Run episode without collecting samples
    episode_reward, _ = run_episode_with_collection(
        env, 
        action_selector, 
        collect_sa_pairs=False  # Disable collection
    )
    return episode_reward
```

**Benefits:**
- ✅ Single implementation of stepping logic
- ✅ DRY (Don't Repeat Yourself) principle
- ✅ Easy to maintain - fix once, works everywhere
- ✅ Flexible - different action selectors for different strategies
- ✅ Testable - one function to test thoroughly
- ✅ Extensible - easy to add new features

## Key Difference: Action Selector Pattern

The main innovation is using **action_selector** as a parameter:

```python
# For collect_samples: use random actions
action_selector = lambda s: env.action_space.sample()

# For run_episode: use epsilon-greedy policy
action_selector = lambda s: epsilon_greedy(s, eps)

# For any new policy: just pass a different selector
action_selector = lambda s: my_new_policy(s, params)
```

## Architecture Diagram

```
                    run_episode_with_collection()
                    ├─ Resets environment
                    ├─ Runs episode loop
                    ├─ Collects state-action pairs (if enabled)
                    ├─ Returns (reward, samples)
                    └─ Uses action_selector function (pluggable!)
                           ↓
        ┌─────────────────┴─────────────────┐
        ↓                                    ↓
collect_samples()                    run_episode()
├─ Calls with random                 ├─ Calls with epsilon-greedy
├─ collect_sa_pairs=True             ├─ collect_sa_pairs=False
└─ Returns samples                   └─ Returns reward
```

## Usage Examples

### Using collect_samples (unchanged API)
```python
env = gymnasium.make("CartPole-v1")

# Collect 10,000 samples for training the RBFSampler
samples = collect_samples(env, num_samples=10_000)
print(f"Collected {len(samples)} state-action pairs")
# Output: Collected ~500,000 state-action pairs
```

### Using run_episode (unchanged API)
```python
# Initialize model and run an episode
MODEL = ValueFunctionApproximator(train_env)

# Run episode with 10% exploration
reward = run_episode(train_env, eps=0.1)
print(f"Episode reward: {reward}")
# Output: Episode reward: 195.0
```

### Using the helper directly (new capability)
```python
# You can now use the helper for custom policies
def custom_policy(state):
    # Your custom logic here
    return action

reward, collected_samples = run_episode_with_collection(
    env,
    action_selector=custom_policy,
    collect_sa_pairs=True
)

print(f"Reward: {reward}, Samples: {len(collected_samples)}")
```

## Summary Table

| Aspect | Before | After |
|--------|--------|-------|
| **Code Duplication** | 2 stepping loops | 1 unified loop |
| **Maintenance** | Fix 2 places | Fix 1 place |
| **Flexibility** | Action hardcoded | Action selector parameter |
| **Sample Collection** | Special case | Toggleable flag |
| **API Changes** | None | None (backward compatible) |
| **Testability** | Need to test both | Test once, both inherit tests |

## Testing Strategy

The helper function is tested thoroughly with 5 new test cases:

```python
def test_returns_tuple_with_reward_and_samples()
def test_collects_sa_pairs_when_enabled()
def test_no_sample_collection_when_disabled()
def test_accumulates_reward_correctly()
def test_action_selector_called_correctly()
```

Because both `collect_samples()` and `run_episode()` use this helper, they automatically inherit its reliability.


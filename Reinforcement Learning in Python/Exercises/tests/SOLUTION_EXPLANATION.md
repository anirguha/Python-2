# Solution Summary: How to Make `collect_samples()` Call `run_episode()`

## Your Question
> "In the function collect_samples the run episode function is being called. Only difference is samples have been concatenated together with (s,[a]). How to achieve that, i.e. collect_samples will call run_episode and collect samples at the same time?"

## The Answer: **Create a Shared Helper Function**

The key insight is to extract the common episode-stepping logic into a helper function that **both functions can use**.

### The Core Solution

```python
def run_episode_with_collection(env: gymnasium.Env,
                                 action_selector,
                                 collect_sa_pairs: bool = False) -> Tuple[float, Samples]:
    """
    Universal episode runner with optional sample collection.
    
    Args:
        env: Gymnasium environment
        action_selector: Function that takes state → action (pluggable!)
        collect_sa_pairs: Toggle to collect state-action pairs
    
    Returns:
        (episode_reward, samples)
    """
    s, info = env.reset()
    done, truncated = False, False
    episode_reward, samples = 0.0, []

    while not (done or truncated):
        a = action_selector(s)  # Flexible action selection!
        
        if collect_sa_pairs:  # Toggle collection on/off
            sa = np.concatenate((s, [a]))  # This is what you wanted!
            samples.append(sa)
        
        s, r, done, truncated, info = env.step(a)
        episode_reward += r

    return episode_reward, samples
```

### How Both Functions Use It

**`collect_samples()` - Collects samples with random actions**
```python
def collect_samples(env: gymnasium.Env, num_samples: int) -> Samples:
    all_samples = []
    
    for _ in tqdm(range(num_samples)):
        # Pass RANDOM action selector
        action_selector = lambda s: env.action_space.sample()
        
        # Call helper with collection ENABLED
        _, samples = run_episode_with_collection(
            env, action_selector, collect_sa_pairs=True
        )
        all_samples.extend(samples)
    
    return all_samples  # Returns state-action samples!
```

**`run_episode()` - Runs episodes with epsilon-greedy**
```python
def run_episode(env: gymnasium.Env, eps: float = EPSILON) -> float:
    # Pass EPSILON-GREEDY action selector
    action_selector = lambda s: epsilon_greedy(s, eps)
    
    # Call helper with collection DISABLED
    episode_reward, _ = run_episode_with_collection(
        env, action_selector, collect_sa_pairs=False
    )
    return episode_reward  # Returns total reward!
```

## Key Design Pattern: **Action Selector**

The secret is using a pluggable `action_selector` function:

```python
# For collect_samples: random exploration
action_selector = lambda s: env.action_space.sample()

# For run_episode: epsilon-greedy exploitation  
action_selector = lambda s: epsilon_greedy(s, eps)

# For any new policy: just plug in your function!
action_selector = lambda s: your_policy(s, params)
```

## What Changed in Your Code

### Before (Duplicated Logic)
```
collect_samples()          run_episode()
    ├─ reset()                ├─ reset()
    ├─ loop until done         ├─ loop until done
    ├─ random action           ├─ epsilon_greedy action
    ├─ concatenate s,a         └─ accumulate reward (no sampling!)
    └─ accumulate samples
```

### After (Unified Logic)
```
                    run_episode_with_collection()
                    ├─ reset()
                    ├─ loop until done
                    ├─ USE ACTION_SELECTOR (generic!)
                    ├─ IF collect_sa_pairs: concatenate s,a
                    ├─ accumulate reward
                    └─ return (reward, samples)
                            ↙        ↘
        collect_samples()           run_episode()
        (collect=True)              (collect=False)
```

## Benefits

| | Before | After |
|---|--------|-------|
| **Code Duplication** | 2 loops | 1 loop |
| **Maintenance** | Fix 2 places | Fix 1 place |
| **Flexibility** | Fixed logic | Pluggable actions |
| **Sample Collection** | Special case | Toggleable |
| **Testability** | Test both | Test once |
| **Extensibility** | Hard to extend | Easy to extend |

## Files Modified

1. **`cartpole_control_approximation_algorithm.py`**
   - Added `run_episode_with_collection()` helper
   - Refactored `collect_samples()` to use helper
   - Refactored `run_episode()` to use helper

2. **`tests/test_cartpole_control_approximation_algorithm.py`**
   - 34 comprehensive tests (all passing ✅)
   - Tests for the new helper function
   - Tests for refactored functions

## Verification

All 34 tests passing:
```
✅ 4 tests: Epsilon-Greedy
✅ 5 tests: New Helper Function (run_episode_with_collection)
✅ 6 tests: Collect Samples
✅ 3 tests: Run Episode
✅ 5 tests: ValueFunctionApproximator
✅ 3 tests: Evaluate Agent
✅ 3 tests: Watch Agent
✅ 2 tests: Integration
✅ 3 tests: Edge Cases
```

## How to Run Tests

```bash
cd "Reinforcement Learning in Python/Exercises"
python -m pytest tests/test_cartpole_control_approximation_algorithm.py -v
```

## Summary

**The solution is: Create a helper function (`run_episode_with_collection`) that:**
1. Contains the core episode-stepping logic (once)
2. Accepts a pluggable `action_selector` function
3. Has a toggle for `collect_sa_pairs` 
4. Returns both reward AND samples

**Then both functions use it:**
- `collect_samples()`: calls with random actions + collection enabled
- `run_episode()`: calls with epsilon-greedy + collection disabled

This follows the **DRY (Don't Repeat Yourself)** principle and makes your code more maintainable, testable, and extensible!


import torchvision.transforms as T
import torchvision.transforms.v2 as v2

def extract_mean_std(tf):
    """Find Normalize in a (possibly nested) v1/v2 transform; return (mean, std) or (None, None)."""
    stack = [tf]
    while stack:
        t = stack.pop()
        if hasattr(t, "transforms"):  # Compose-like
            stack.extend(t.transforms)
            continue
        if isinstance(t, (T.Normalize, v2.Normalize)):
            return t.mean, t.std
        for attr in ("transform", "augmentation", "random_transform"):
            if hasattr(t, attr):
                stack.append(getattr(t, attr))
    return None, None

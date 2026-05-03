"""Import bridge for the legacy exercises directory."""

from __future__ import annotations

import sys
from pathlib import Path


_LEGACY_EXERCISES_DIR = (
    Path(__file__).resolve().parents[2]
    / "Reinforcement Learning in Python"
    / "Exercises"
)

if not _LEGACY_EXERCISES_DIR.is_dir():
    raise ImportError(f"Legacy exercises directory not found: {_LEGACY_EXERCISES_DIR}")

_legacy_path = str(_LEGACY_EXERCISES_DIR)
if _legacy_path not in sys.path:
    sys.path.insert(0, _legacy_path)

# Let imports such as
# Reinforcement_Learning_in_Python.Exercises.gridworld_standard_windy
# resolve modules from the legacy directory.
__path__.append(_legacy_path)
